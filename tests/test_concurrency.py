"""Concurrency and failure-mode tests.

These cover the QC complaints these fixes were made for:

* several submissions of the same new furnace run must yield exactly one
  ``replaced: false`` receipt -- the first that commits -- and every later one
  must say it replaced an archive;
* each receipt (archive, over-temperature periods) must describe the data
  *that request wrote*, even if another request replaced it again before the
  response was sent;
* the same holds when two independent service processes share one SQLite file;
* one process compressing a long curve must still answer lookups and short
  archives for other runs;
* a rejected or failed write must never clobber the archive already on disk.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import threading
import time
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.compression import above_limit_intervals
from app.main import create_app


def _body(run_id: str, *, limit: int = 50, offset: int = 0, max_points: int = 100):
    return {
        "run_id": run_id,
        "limit": limit,
        "tolerance": 0,
        "max_points": max_points,
        "segments": [
            [
                [0, 10 + offset],
                [10, 90 + offset],
                [20, 10 + offset],
            ]
        ],
    }


def _assert_receipt_matches_its_own_write(response, body) -> None:
    """The receipt must describe exactly the request's own points and limit."""
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["archived"] is True
    own_points = [{"t": t, "v": v} for t, v in body["segments"][0]]
    assert payload["segments"] == [{"points": own_points, "point_count": len(own_points)}]

    archive = payload["archive"]
    assert archive["limit"] == body["limit"]
    assert archive["segments"][0]["points"] == own_points
    # Over-temperature periods are derived from the same archived polyline and
    # the same limit -- never from a concurrent writer's archive.
    times = [p["t"] for p in own_points]
    values = [p["v"] for p in own_points]
    expected = above_limit_intervals(times, values, body["limit"])
    reported = payload["over_temperature_periods"]
    assert len(reported) == len(expected)
    for interval, item in zip(expected, reported):
        assert float(interval.start) == pytest.approx(item["start_ms"], abs=1e-9)
        assert float(interval.end) == pytest.approx(item["end_ms"], abs=1e-9)
        assert interval.start_closed == item["start_closed"]
        assert interval.end_closed == item["end_closed"]
    assert payload["over_temperature_periods"] == archive["over_temperature_periods"]


def test_concurrent_same_run_has_one_first_and_consistent_receipts(tmp_path):
    """A burst of same-run submissions: one first archive, rest replacements."""
    db_path = str(tmp_path / "burst.db")
    app = create_app(db_path)
    count = 12
    bodies = [_body("same-heat", limit=40 + i, offset=i) for i in range(count)]

    def submit(body):
        # A fresh client per thread; the app (and the SQLite file) is shared.
        with TestClient(app) as worker:
            return worker.post("/runs", json=body)

    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        responses = list(pool.map(submit, bodies))

    for response, body in zip(responses, bodies):
        _assert_receipt_matches_its_own_write(response, body)

    replaced_flags = [response.json()["replaced"] for response in responses]
    assert replaced_flags.count(False) == 1, replaced_flags

    # The final stored archive is exactly one of the submissions, intact.
    with TestClient(app) as reader:
        stored = reader.get("/runs/same-heat").json()
    assert stored["limit"] in {body["limit"] for body in bodies}
    match = next(body for body in bodies if body["limit"] == stored["limit"])
    assert stored["segments"][0]["points"] == [
        {"t": t, "v": v} for t, v in match["segments"][0]
    ]


# ---------------------------------------------------------------------------
# Two service processes sharing one SQLite database
# ---------------------------------------------------------------------------


def _process_submit(db_path, barrier, queue, index):
    # Runs in a forked child: a completely separate app instance and SQLite
    # connection set, sharing only the database file.
    app = create_app(db_path)
    body = _body("shared-heat", limit=40 + index, offset=index)
    with TestClient(app) as client:
        barrier.wait(timeout=30)
        response = client.post("/runs", json=body)
        queue.put((index, response.status_code, response.json(), body))


def test_two_processes_sharing_existing_database(tmp_path):
    db_path = str(tmp_path / "shared.db")
    # The file already exists and already holds an unrelated run.
    with TestClient(create_app(db_path)) as seed:
        assert seed.post("/runs", json=_body("older-run")).status_code == 200

    workers = 4
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(workers)
    queue = context.Queue()
    processes = [
        context.Process(
            target=_process_submit, args=(db_path, barrier, queue, i)
        )
        for i in range(workers)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=60) for _ in range(workers)]
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    for _index, status, payload, body in results:
        assert status == 200, payload
        assert payload["archived"] is True
        own_points = [{"t": t, "v": v} for t, v in body["segments"][0]]
        assert payload["archive"]["limit"] == body["limit"]
        assert payload["archive"]["segments"][0]["points"] == own_points
        assert payload["segments"][0]["points"] == own_points

    flags = [payload["replaced"] for _index, _status, payload, _body in results]
    assert flags.count(False) == 1, flags

    with TestClient(create_app(db_path)) as reader:
        assert reader.get("/runs/older-run").status_code == 200
        assert reader.get("/runs/shared-heat").status_code == 200


# ---------------------------------------------------------------------------
# Responsiveness while a long curve is being compressed
# ---------------------------------------------------------------------------


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_reads_and_short_archives_are_served_during_long_compression(
    tmp_path, monkeypatch
):
    import uvicorn

    import app.service as service_module

    db_path = str(tmp_path / "responsive.db")
    app = create_app(db_path)
    original = service_module.compress_segment
    in_compression = threading.Event()
    release = threading.Event()

    def slow_for_long_curves(times, values, limit, tolerance):
        if len(times) > 1000:
            in_compression.set()
            # Hold the worker thread until the short operations have proven
            # they do not wait for it (bounded so the test cannot hang).
            release.wait(timeout=10)
        return original(times, values, limit, tolerance)

    monkeypatch.setattr(service_module, "compress_segment", slow_for_long_curves)

    # A real single-process server: one event loop drives every request.
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
        )
    )
    server_thread = threading.Thread(target=server.run)
    server_thread.start()
    base = f"http://127.0.0.1:{port}"

    import httpx

    try:
        deadline = time.time() + 15
        while not server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.02)

        with httpx.Client(base_url=base, timeout=20.0) as client:
            assert client.post("/runs", json=_body("catalog")).status_code == 200
            long_body = {
                "run_id": "long",
                "limit": 1000,
                "tolerance": 0,
                "max_points": 10**6,
                "segments": [[[t, t % 7] for t in range(0, 5000)]],
            }

            long_response: dict[str, object] = {}

            def submit_long():
                with httpx.Client(base_url=base, timeout=20.0) as worker:
                    long_response["response"] = worker.post(
                        "/runs", json=long_body
                    )

            thread = threading.Thread(target=submit_long)
            thread.start()
            try:
                assert in_compression.wait(timeout=10), "compression never started"
                started = time.perf_counter()
                looked_up = client.get("/runs/catalog")
                lookup_seconds = time.perf_counter() - started
                assert looked_up.status_code == 200

                started = time.perf_counter()
                short = client.post("/runs", json=_body("short-during-long"))
                short_seconds = time.perf_counter() - started
                assert short.status_code == 200, short.text
            finally:
                release.set()
                thread.join(timeout=30)

        assert not thread.is_alive()
        assert long_response["response"].status_code == 200
        assert lookup_seconds < 1.0, lookup_seconds
        assert short_seconds < 1.0, short_seconds
    finally:
        release.set()
        server.should_exit = True
        server_thread.join(timeout=15)


# ---------------------------------------------------------------------------
# Rejected / failed writes never overwrite
# ---------------------------------------------------------------------------


def test_storage_failure_keeps_previous_archive(tmp_path, monkeypatch):
    db_path = str(tmp_path / "failure.db")
    app = create_app(db_path)
    with TestClient(app) as client:
        assert client.post("/runs", json=_body("durable")).status_code == 200
        before = client.get("/runs/durable").json()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(app.state.store, "put", boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/runs", json=_body("durable", offset=3))
        assert response.status_code == 500
        after = client.get("/runs/durable").json()
    assert after == before


def test_invalid_request_keeps_previous_archive(tmp_path):
    db_path = str(tmp_path / "invalid.db")
    with TestClient(create_app(db_path)) as client:
        assert client.post("/runs", json=_body("kept")).status_code == 200
        before = client.get("/runs/kept").json()

        broken = _body("kept", offset=5)
        broken["segments"] = [[[0, 10], [0, 20]]]  # non-increasing times
        assert client.post("/runs", json=broken).status_code == 422
        assert client.get("/runs/kept").json() == before
