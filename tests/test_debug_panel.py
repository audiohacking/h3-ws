"""Debug panel API: report zip, self-test guard, console sequence cursor."""

import io
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import h3_console
import h3_selftest
import web_ui
from h3_selftest import parse_bench_output, parse_dit_profile


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(web_ui, "_ensure_web_deps", lambda: None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    engine = SimpleNamespace(h3_bin=bin_dir / "h3", model_dir=tmp_path / "models")
    state = web_ui.AppState(tmp_path / "out", tmp_path / "up", engine)
    app = web_ui.create_app(state, mount_static=False)
    return TestClient(app), state, bin_dir


def _clip(state, **fields):
    clip = web_ui.ClipRecord(
        id="c1", prompt="a secret prompt about Picture 1", label="CURRENT",
        video_url="", filename="x.mp4", chain_id="k", clip_index=0,
        mode="ref2va", status="done", created_at="2026-09-26T10:00:00",
        width=768, height=1024, num_frames=107, elapsed_s=1494.8,
        generation={"refs": [{"kind": "image", "path": "/Users/me/private.jpg"}]},
        timings={"outcome": "done", "total_s": 1494.8,
                 "phases": [{"phase": "denoise", "start_s": 132.1,
                             "seconds": 1275.3, "steps": 3, "total": 3}],
                 "profile": []},
        **fields,
    )
    state.clips[clip.id] = clip


def test_report_zip_has_facts_and_timings_but_no_prompt_or_paths(client):
    http, state, _ = client
    _clip(state)
    response = http.get("/api/debug/report?kind=snapshot")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert {"report.json", "report.txt", "console.txt"} <= set(archive.namelist())
    report = json.loads(archive.read("report.json"))
    generation = report["recent_generations"][-1]
    assert generation["timings"]["phases"][0]["phase"] == "denoise"
    assert generation["refs"] == ["image"]
    assert generation["prompt_chars"] == len("a secret prompt about Picture 1")
    everything = b"".join(archive.read(name) for name in archive.namelist())
    assert b"secret prompt" not in everything
    assert b"private.jpg" not in everything
    assert "denoise" in archive.read("report.txt").decode()


def test_selftest_refused_while_generating(client):
    http, state, _ = client
    state._active_run_id = "r1"
    assert http.post("/api/debug/selftest", json={"mode": "quick"}).status_code == 409
    state._active_run_id = None


def test_stale_running_records_do_not_block_selftest(client):
    # A crash can leave runs marked "running" in index.json forever.
    http, state, _ = client
    state.runs["old"] = web_ui.RunRecord(id="old", status="running",
                                         prompts=["x"], chain_id="k")
    assert http.post("/api/debug/selftest", json={"mode": "quick"}).status_code == 200
    deadline = time.time() + 10
    while state.selftest.running and time.time() < deadline:
        time.sleep(0.05)


def test_selftest_reports_missing_binaries(client):
    http, state, _ = client
    assert http.post("/api/debug/selftest", json={"mode": "bogus"}).status_code == 400
    started = http.post("/api/debug/selftest", json={"mode": "quick"})
    assert started.status_code == 200
    deadline = time.time() + 10
    while state.selftest.running and time.time() < deadline:
        time.sleep(0.05)
    status = http.get("/api/debug/selftest").json()
    assert status["state"] == "failed"
    assert {step["status"] for step in status["steps"]} == {"missing"}


def test_selftest_runs_bundled_binaries(client):
    http, state, bin_dir = client
    for name, _, _ in h3_selftest.QUICK_STEPS:
        script = bin_dir / name
        body = ('echo \'{"op":"gemm_bf16","shape":"x","seconds":0.1,"tflops":25.3}\''
                if name == "h3_bench" else f"echo ok: {name}")
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
    http.post("/api/debug/selftest", json={"mode": "quick"})
    deadline = time.time() + 20
    while state.selftest.running and time.time() < deadline:
        time.sleep(0.05)
    status = http.get("/api/debug/selftest").json()
    assert status["state"] == "passed"
    assert status["bench"] == [{"op": "gemm_bf16", "shape": "x",
                                "seconds": 0.1, "tflops": 25.3}]
    assert status["steps"][0]["summary"] == "ok: h3_tests"


def test_console_cursor_returns_only_new_lines():
    h3_console.clear_console()
    start = h3_console.get_console_since(0)["seq"]
    h3_console.append_console("first")
    h3_console.append_console("second")
    chunk = h3_console.get_console_since(start)
    assert [line[11:] for line in chunk["lines"]] == ["first", "second"]
    assert h3_console.get_console_since(chunk["seq"])["lines"] == []
    assert chunk["dropped"] == 0


def test_parsers_read_h3_output():
    assert parse_bench_output('noise\n{"op":"a","tflops":1.5}\n{bad') == [
        {"op": "a", "tflops": 1.5}]
    profile = parse_dit_profile(
        "h3: DiT sequence 61395 rows (text 7263, video 24576)\n"
        "denoise 0/3   h3: DiT op profile over 3 blocks at 61395 rows "
        "(7.751 s/block)\n"
        "h3:   DiT full attention                                 "
        "5600.7 ms/block  72.3%\n")
    assert profile["sequence_rows"] == 61395
    assert profile["seconds_per_block"] == 7.751
    assert profile["ops"] == [{"op": "DiT full attention",
                               "ms_per_block": 5600.7, "percent": 72.3}]


def test_report_saved_to_downloads_and_only_saved_paths_reveal(client, tmp_path,
                                                               monkeypatch):
    http, _, _ = client
    home = tmp_path / "home"
    (home / "Downloads").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    opened = []
    real_run = web_ui.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "open":
            opened.append(cmd)
            return None
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(web_ui.subprocess, "run", fake_run)
    saved = http.post("/api/debug/report/save", json={"kind": "snapshot"}).json()
    path = Path(saved["path"])
    assert path.parent == home / "Downloads" and path.suffix == ".zip"
    assert zipfile.ZipFile(path).testzip() is None
    assert http.post("/api/debug/reveal", json={"path": saved["path"]}).status_code == 200
    assert opened == [["open", "-R", saved["path"]]]
    assert http.post("/api/debug/reveal",
                     json={"path": "/etc/passwd"}).status_code == 404


def test_logs_in_report_mask_prompt_file_names_and_home(client, tmp_path,
                                                        monkeypatch):
    http, _, _ = client
    logs = tmp_path / "logs"
    logs.mkdir()
    home = str(Path.home())
    (logs / "server.log").write_text(
        '14:40:43  1.2.3.4 - "GET /api/videos/web_subject_definitions_subject_1_'
        'is_the_artist_as_s_20260925_134234_0.mp4 HTTP/1.1" 200\n'
        f"h3 session argv: {home}/Documents/git/h3-ws/h3 -d models\n")
    monkeypatch.setenv("H3_WS_LOG_DIR", str(logs))
    archive = zipfile.ZipFile(io.BytesIO(http.get("/api/debug/report").content))
    text = archive.read("logs/server.log").decode()
    assert "web_…_20260925_134234_0.mp4" in text
    assert "subject_definitions" not in text
    assert home not in text and "~/Documents/git/h3-ws/h3" in text


def test_runs_left_running_by_a_crash_are_marked_failed_on_load(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    clip = {"id": "c1", "prompt": "p", "label": "", "video_url": "",
            "filename": "x.mp4", "chain_id": "k", "clip_index": 0,
            "mode": "ref2va", "status": "running", "created_at": "t"}
    done = dict(clip, id="c2", status="done")
    runs = [{"id": "r1", "status": "running", "prompts": ["p"], "chain_id": "k"},
            {"id": "r2", "status": "queued", "prompts": ["p"], "chain_id": "k"},
            {"id": "r3", "status": "done", "prompts": ["p"], "chain_id": "k"}]
    (out / "index.json").write_text(json.dumps({"clips": [clip, done], "runs": runs}))
    engine = SimpleNamespace(h3_bin=tmp_path / "h3", model_dir=tmp_path / "m")
    state = web_ui.AppState(out, tmp_path / "up", engine)
    state.load_index()
    assert state.runs["r1"].status == "failed" and "interrupted" in state.runs["r1"].error
    assert state.runs["r2"].status == "failed"
    assert state.runs["r3"].status == "done" and state.runs["r3"].error is None
    assert state.clips["c1"].status == "failed" and state.clips["c2"].status == "done"
    saved = json.loads((out / "index.json").read_text())
    assert {r["id"]: r["status"] for r in saved["runs"]} == {
        "r1": "failed", "r2": "failed", "r3": "done"}
