"""Queued generation cancel/edit plumbing — library cards need immediate status."""

from types import SimpleNamespace

import pytest

import web_ui


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(web_ui, "_ensure_web_deps", lambda: None)
    engine = SimpleNamespace(
        h3_bin=tmp_path / "h3",
        model_dir=tmp_path / "models",
        request_cancel=lambda: None,
    )
    state = web_ui.AppState(tmp_path / "out", tmp_path / "up", engine)
    app = web_ui.create_app(state, mount_static=False)
    return TestClient(app), state


def test_cancel_queued_run_marks_clips_cancelled(client):
    http, state = client
    run_id = "run-q1"
    clip_id = "clip-q1"
    state.clips[clip_id] = web_ui.ClipRecord(
        id=clip_id,
        prompt="queued shot",
        label="CURRENT",
        video_url="",
        filename="web_queued.mp4",
        chain_id="chain-q",
        clip_index=0,
        mode="t2va",
        status=web_ui.RunStatus.QUEUED.value,
        created_at="2026-10-04T12:00:00",
        run_id=run_id,
    )
    state.runs[run_id] = web_ui.RunRecord(
        id=run_id,
        status=web_ui.RunStatus.QUEUED.value,
        prompts=["queued shot"],
        chain_id="chain-q",
        clip_ids=[clip_id],
        created_at="2026-10-04T12:00:00",
    )
    state._active_run_id = "run-other"

    r = http.post(f"/api/runs/{run_id}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    assert state.runs[run_id].status == web_ui.RunStatus.CANCELLED.value
    assert state.clips[clip_id].status == web_ui.RunStatus.CANCELLED.value


def test_clip_api_exposes_run_id(client):
    http, state = client
    state.clips["c1"] = web_ui.ClipRecord(
        id="c1",
        prompt="hi",
        label="CURRENT",
        video_url="",
        filename="x.mp4",
        chain_id="k",
        clip_index=0,
        mode="t2va",
        status="queued",
        created_at="2026-10-04T12:00:00",
        run_id="run-1",
        project_id=state.active_project_id,
    )
    data = http.get("/api/clips").json()
    hit = next(c for c in data["clips"] if c["id"] == "c1")
    assert hit["run_id"] == "run-1"
