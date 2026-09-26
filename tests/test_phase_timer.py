"""PhaseTimer turns the h3.c progress stream into per-phase generation timings."""

from h3_backend import PhaseTimer, timings_console_line


def _run(events):
    timer = PhaseTimer()
    timer.start(now=100.0)
    for clock, mp in events:
        timer.observe(mp, now=clock)
    return timer


def test_phases_end_where_the_next_begins():
    timer = _run([
        (100.5, {"stage": "loading", "label": "Loading Ref2VA weights"}),
        (102.0, {"stage": "video VAE encoder", "step": 0, "total": 28}),
        (160.0, {"stage": "video VAE encoder", "step": 28, "total": 28}),
        (262.0, {"stage": "text encoder", "step": 0, "total": 50}),
        # Status lines between counters must not open phantom phases.
        (270.0, {"stage": "loading", "label": "h3: conditioning cache miss"}),
        (300.0, {"stage": "text encoder", "step": 50, "total": 50}),
        (310.0, {"stage": "denoise", "step": 1, "total": 3}),
        (1210.0, {"stage": "denoise", "step": 3, "total": 3}),
    ])
    summary = timer.summary(outcome="done", now=1300.0)
    phases = summary["phases"]
    assert [p["phase"] for p in phases] == [
        "startup", "video VAE encoder", "text encoder", "denoise"]
    assert phases[0] == {"phase": "startup", "start_s": 0.5, "seconds": 1.5}
    assert phases[1]["seconds"] == 160.0
    assert phases[1]["steps"] == 28 and phases[1]["total"] == 28
    assert phases[2]["start_s"] == 162.0 and phases[2]["seconds"] == 48.0
    assert phases[3]["seconds"] == 990.0
    assert phases[3]["s_per_step"] == 330.0
    assert summary["total_s"] == 1200.0
    assert summary["outcome"] == "done"


def test_profile_lines_are_kept_not_treated_as_phases():
    line = ("h3 profile: video VAE encoder        total          wall=  "
            "164.329s encode=  0.229s wait= 162.595s root-gpu=  6.427s "
            "peak= 50.298GiB alloc= 54.431GiB submissions=504")
    timer = _run([
        (101.0, {"stage": "video VAE encoder", "step": 1, "total": 28}),
        (265.0, {"stage": "loading", "label": line}),
    ])
    summary = timer.summary(outcome="failed", now=270.0)
    assert [p["phase"] for p in summary["phases"]] == ["video VAE encoder"]
    assert summary["profile"] == [{
        "name": "video VAE encoder", "mark": "total",
        "wall_s": 164.329, "peak_gib": 50.298}]


def test_phase_followed_by_another_counts_as_complete():
    timer = _run([
        (100.0, {"stage": "denoise", "step": 1, "total": 3}),
        (500.0, {"stage": "denoise", "step": 2, "total": 3}),
        # h3 goes straight to the decoder without printing "denoise 3/3".
        (1000.0, {"stage": "audio VAE", "step": 0, "total": 7}),
    ])
    phases = timer.summary(outcome="done", now=1001.0)["phases"]
    assert phases[0]["steps"] == 3 and phases[0]["s_per_step"] == 300.0
    # The last phase is only as complete as its last counter.
    assert phases[1]["steps"] == 0 and "s_per_step" not in phases[1]


def test_console_line_lists_every_phase():
    timer = _run([
        (100.0, {"stage": "text encoder", "step": 0, "total": 50}),
        (138.0, {"stage": "denoise", "step": 1, "total": 3}),
    ])
    line = timings_console_line(timer.summary(outcome="cancelled", now=200.0))
    assert line == ("h3 timings (cancelled, total 01:40): "
                    "text encoder 00:38 · denoise 01:02")
