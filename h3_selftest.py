"""Built-in self-test and debug report for the Web UI Debug panel.

Quick mode runs the bundled h3.c oracle tests and the GPU micro-bench
(``h3_bench``) from the engine directory: about a minute, no model weights.
Deep mode additionally runs a short real-weights generation with
``H3_DIT_OP_PROFILE`` and stops it once the per-op table is printed.

``build_report_zip`` packages device facts, self-test results, recent
generation timings (no prompts, no media paths), the console buffer and log
tails into one zip that users can send back.
"""

from __future__ import annotations

import io
import json
import logging
import platform
import queue
import re
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

QUICK_STEPS: list[tuple[str, list[str], float]] = [
    # (binary, args, timeout seconds)
    ("h3_tests", [], 120.0),
    ("h3_gqa_tests", [], 180.0),
    ("h3_conv3d_tests", [], 120.0),
    ("h3_sdpa_split_tests", [], 180.0),
    ("h3_cache_invalidate_tests", [], 60.0),
    ("h3_bench", ["1"], 300.0),
]

DEEP_PROMPT = (
    "A slow dolly shot through a sunlit greenhouse, leaves moving in a light "
    "breeze, soft ambient birdsong."
)
DEEP_TIMEOUT_S = 1800.0

_SEQUENCE_RE = re.compile(r"DiT sequence (\d+) rows \((.*)\)")
_PROFILE_HEADER_RE = re.compile(
    r"DiT op profile over (\d+) blocks at (\d+) rows \(([\d.]+) s/block\)"
)
_PROFILE_ROW_RE = re.compile(
    r"h3:\s+(?P<op>.+?)\s{2,}(?P<ms>[\d.]+) ms/block\s+(?P<pct>[\d.]+)%"
)


def _tail(text: str, lines: int = 15) -> list[str]:
    rows = [row.rstrip() for row in text.replace("\r", "\n").splitlines()]
    return [row for row in rows if row.strip()][-lines:]


def parse_bench_output(text: str) -> list[dict[str, Any]]:
    results = []
    for row in text.splitlines():
        row = row.strip()
        if row.startswith("{"):
            try:
                results.append(json.loads(row))
            except json.JSONDecodeError:
                continue
    return results


def parse_dit_profile(text: str) -> dict[str, Any]:
    """Sequence composition and the per-op table from h3.c stderr."""
    out: dict[str, Any] = {}
    for row in text.replace("\r", "\n").splitlines():
        match = _SEQUENCE_RE.search(row)
        if match:
            out["sequence_rows"] = int(match.group(1))
            out["sequence_parts"] = match.group(2)
            continue
        match = _PROFILE_HEADER_RE.search(row)
        if match:
            out["blocks"] = int(match.group(1))
            out["rows"] = int(match.group(2))
            out["seconds_per_block"] = float(match.group(3))
            out["ops"] = []
            continue
        match = _PROFILE_ROW_RE.search(row)
        if match and "ops" in out:
            out["ops"].append({
                "op": match.group("op").strip(),
                "ms_per_block": float(match.group("ms")),
                "percent": float(match.group("pct")),
            })
    return out


class SelfTestRunner:
    """One self-test at a time, in a background thread; poll ``status()``."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._proc: subprocess.Popen[str] | None = None
        self._status: dict[str, Any] = {"state": "idle"}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status))

    def start(self, mode: str) -> dict[str, Any]:
        if mode not in {"quick", "deep"}:
            raise ValueError("mode must be quick or deep")
        if self.running:
            raise RuntimeError("a self-test is already running")
        steps = [{"name": name, "status": "pending"} for name, _, _ in QUICK_STEPS]
        if mode == "deep":
            steps.append({"name": "dit_op_profile", "status": "pending"})
        with self._lock:
            self._status = {
                "state": "running",
                "mode": mode,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "steps": steps,
                "bench": [],
                "dit_profile": None,
            }
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run, args=(mode,), name="h3-selftest", daemon=True
        )
        self._thread.start()
        return self.status()

    def cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def _update_step(self, index: int, **fields: Any) -> None:
        with self._lock:
            self._status["steps"][index].update(fields)

    def _console(self, message: str) -> None:
        from h3_console import append_console

        append_console("self-test: %s", message)

    def _run(self, mode: str) -> None:
        started = time.time()
        failed = False
        try:
            for index, (name, args, timeout) in enumerate(QUICK_STEPS):
                if self._cancel.is_set():
                    break
                ok = self._run_binary(index, name, args, timeout)
                failed = failed or not ok
            if mode == "deep" and not self._cancel.is_set():
                ok = self._run_deep(len(QUICK_STEPS))
                failed = failed or not ok
        except Exception as exc:  # report, never crash the server
            log.exception("self-test crashed")
            failed = True
            with self._lock:
                self._status["error"] = str(exc)
                for step in self._status["steps"]:
                    if step["status"] == "running":
                        step["status"] = "failed"
                        step["summary"] = str(exc)
        state = "cancelled" if self._cancel.is_set() else (
            "failed" if failed else "passed")
        with self._lock:
            self._status["state"] = state
            self._status["seconds"] = round(time.time() - started, 1)
            self._status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._console(f"{mode} run {state} in {time.time() - started:.0f}s")

    def _bin_dir(self) -> Path:
        return Path(self.engine.h3_bin).resolve().parent

    def _run_binary(self, index: int, name: str, args: list[str],
                    timeout: float) -> bool:
        path = self._bin_dir() / name
        if not path.is_file():
            self._update_step(index, status="missing",
                              summary=f"{name} is not bundled with this build")
            self._console(f"{name}: missing")
            return False
        self._update_step(index, status="running")
        self._console(f"{name}: running")
        started = time.time()
        try:
            self._proc = subprocess.Popen(
                [str(path), *args], cwd=str(self._bin_dir()), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            output, _ = self._proc.communicate(timeout=timeout)
            code = self._proc.returncode
        except subprocess.TimeoutExpired:
            self._proc.kill()
            output, _ = self._proc.communicate()
            code = None
        finally:
            self._proc = None
        seconds = round(time.time() - started, 2)
        tail = _tail(output or "")
        ok = code == 0
        status = "passed" if ok else ("timeout" if code is None else "failed")
        if self._cancel.is_set():
            status = "cancelled"
        fields: dict[str, Any] = {
            "status": status, "seconds": seconds, "exit_code": code,
            "summary": tail[-1] if tail else "", "output": tail,
        }
        if name == "h3_bench":
            bench = parse_bench_output(output or "")
            with self._lock:
                self._status["bench"] = bench
            fields["summary"] = ", ".join(
                f"{row['op']} {row.get('tflops', '?')} TFLOPS" for row in bench
                if "tflops" in row)
        self._update_step(index, **fields)
        self._console(f"{name}: {status} ({seconds}s) {fields['summary']}")
        return ok

    def _deep_command(self, scratch: Path) -> tuple[list[str], str]:
        from h3_backend import model_layout_ok

        model_dir = Path(self.engine.model_dir)
        cmd = [str(Path(self.engine.h3_bin).resolve()), "-d",
               str(model_dir.resolve()),
               "--width", "768", "--height", "1024", "--frames", "107",
               "--steps", "3", "--layers", "50", "--reuse", "1",
               "-p", DEEP_PROMPT, "-o", str(scratch / "selftest.mp4")]
        ref_ok, _ = model_layout_ok(model_dir, need_ref2va=True)
        if not ref_ok:
            return cmd, "t2va"
        from PIL import Image

        image = Image.linear_gradient("L").resize((768, 1344)).convert("RGB")
        ref = scratch / "selftest_ref.png"
        image.save(ref)
        return cmd + ["--ref-image", str(ref)], "ref2va"

    def _run_deep(self, index: int) -> bool:
        from h3_backend import model_layout_ok
        from h3_paths import h3_media_env, h3_process_cwd, mk_scratch_dir

        ok, note = model_layout_ok(Path(self.engine.model_dir))
        if not ok:
            self._update_step(index, status="skipped", summary=note)
            return True
        self._update_step(index, status="running")
        # The warm session holds the DiT (tens of GB); free it first.
        stop = getattr(self.engine, "_stop_session", None)
        if stop:
            stop()
        scratch = Path(mk_scratch_dir("h3_selftest_"))
        cmd, mode = self._deep_command(scratch)
        env = h3_media_env()
        env.update({"H3_DIT_OP_PROFILE": "3", "H3_PROFILE": "1"})
        self._console(f"dit_op_profile: loading weights ({mode}, 768x1024x107)")
        started = time.time()
        lines: queue.Queue[str | None] = queue.Queue()
        captured: list[str] = []
        self._proc = subprocess.Popen(
            cmd, cwd=str(h3_process_cwd(self.engine.h3_bin)), env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        def reader(stream: Any) -> None:
            for row in iter(stream.readline, ""):
                lines.put(row)
            lines.put(None)

        threading.Thread(target=reader, args=(self._proc.stdout,),
                         daemon=True).start()
        header_at: float | None = None
        while True:
            if self._cancel.is_set() or time.time() - started > DEEP_TIMEOUT_S:
                break
            if header_at is not None and time.time() - header_at > 2.0:
                break
            try:
                row = lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if row is None:
                break
            for part in row.replace("\r", "\n").splitlines():
                if part.strip():
                    captured.append(part.rstrip())
                    if "h3 profile:" in part or "DiT " in part:
                        self._console(part.strip())
            if header_at is None and "DiT op profile" in row:
                header_at = time.time()
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._proc = None
        profile = parse_dit_profile("\n".join(captured))
        profile["mode"] = mode
        with self._lock:
            self._status["dit_profile"] = profile
        passed = bool(profile.get("ops"))
        seconds = round(time.time() - started, 1)
        summary = (
            f"{profile['rows']} rows, {profile['seconds_per_block']} s/block"
            if passed else "no op profile captured")
        status = "cancelled" if self._cancel.is_set() else (
            "passed" if passed else "failed")
        self._update_step(index, status=status, seconds=seconds,
                          summary=summary, output=captured[-25:])
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)
        return passed


def _run_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def system_facts(engine: Any = None) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "chip": _run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "cpu_cores": _run_text(["sysctl", "-n", "hw.ncpu"]),
    }
    memsize = _run_text(["sysctl", "-n", "hw.memsize"])
    if memsize.isdigit():
        facts["memory_gib"] = round(int(memsize) / 1024**3)
    try:
        gpus = json.loads(_run_text(
            ["system_profiler", "SPDisplaysDataType", "-json"]) or "{}")
        gpu = (gpus.get("SPDisplaysDataType") or [{}])[0]
        facts["gpu"] = gpu.get("sppci_model")
        facts["gpu_cores"] = gpu.get("sppci_cores")
        facts["metal"] = gpu.get("spdisplays_mtlgpufamilysupport")
    except (json.JSONDecodeError, IndexError, AttributeError):
        pass
    try:
        from h3_update import installed_version

        facts["app_version"] = installed_version()
    except Exception:
        facts["app_version"] = None
    if engine is not None:
        h3_bin = Path(engine.h3_bin)
        facts["h3_binary"] = str(h3_bin)
        if h3_bin.is_file():
            facts["h3_binary_mtime"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(h3_bin.stat().st_mtime))
        from h3_backend import model_layout_ok

        model_dir = Path(engine.model_dir)
        facts["fl2va_installed"] = model_layout_ok(model_dir)[0]
        facts["ref2va_installed"] = model_layout_ok(
            model_dir, need_ref2va=True)[0]
    return facts


def recent_generations(clips: list[Any], limit: int = 12) -> list[dict[str, Any]]:
    """Generation metadata without prompts or file paths."""
    rows = []
    for clip in list(clips)[-limit:]:
        data = clip if isinstance(clip, dict) else getattr(clip, "__dict__", {})
        generation = data.get("generation") or {}
        refs = generation.get("refs") or []
        rows.append({
            "created_at": data.get("created_at"),
            "status": data.get("status"),
            "mode": data.get("mode"),
            "width": data.get("width"),
            "height": data.get("height"),
            "frames": data.get("num_frames"),
            "steps": data.get("num_steps"),
            "layers": data.get("layers"),
            "reuse": data.get("reuse"),
            "quality": data.get("quality"),
            "loras": [item.get("name") or Path(str(item.get("path", ""))).name
                      for item in (data.get("loras") or []) if isinstance(item, dict)],
            "refs": [ref.get("kind") for ref in refs if isinstance(ref, dict)],
            "prompt_chars": len(data.get("prompt") or ""),
            "elapsed_s": data.get("elapsed_s"),
            "error": data.get("error"),
            "timings": data.get("timings"),
        })
    return rows


# Output files are named after the prompt ("web_<prompt slug>_<date>_<n>").
_PROMPT_SLUG_RE = re.compile(r"(web|h3)_[a-z0-9_]+?_(\d{8}_\d{6})")


def redact(text: str) -> str:
    """Mask prompt-derived file names and the home directory in log text."""
    text = _PROMPT_SLUG_RE.sub(r"\1_…_\2", text)
    return text.replace(str(Path.home()), "~")


def _file_tail(path: Path, lines: int) -> str:
    try:
        rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(rows[-lines:]) + "\n"


def _text_summary(report: dict[str, Any]) -> str:
    facts = report.get("system") or {}
    out = [
        f"H3-WS debug report · {report['created_at']} · {report['kind']}",
        f"app {facts.get('app_version')} · {facts.get('chip')} · "
        f"GPU {facts.get('gpu_cores')} cores · {facts.get('memory_gib')} GiB · "
        f"macOS {facts.get('macos')}",
        "",
    ]
    selftest = report.get("selftest") or {}
    if selftest.get("state") and selftest["state"] != "idle":
        out.append(f"Self-test ({selftest.get('mode')}): {selftest['state']} "
                   f"in {selftest.get('seconds')} s")
        for step in selftest.get("steps", []):
            out.append(f"  {step.get('status', '?'):9s} {step['name']:28s} "
                       f"{step.get('seconds', '')}s  {step.get('summary', '')}")
        profile = selftest.get("dit_profile") or {}
        for op in profile.get("ops", []):
            out.append(f"    {op['op']:48s} {op['ms_per_block']:9.1f} ms/block "
                       f"{op['percent']:5.1f}%")
        out.append("")
    gens = report.get("recent_generations") or []
    if gens:
        out.append("Recent generations:")
        for gen in gens:
            timings = gen.get("timings") or {}
            phases = " · ".join(
                f"{p['phase']} {p['seconds']:.0f}s"
                for p in timings.get("phases", []) if p.get("seconds", 0) >= 1)
            elapsed = gen.get("elapsed_s")
            took = f"{elapsed:.0f}s" if isinstance(elapsed, (int, float)) else "-"
            out.append(f"  {str(gen.get('created_at'))[:19]} {gen.get('mode')} "
                       f"{gen.get('width')}x{gen.get('height')}x{gen.get('frames')} "
                       f"{gen.get('status')} {took}  {phases}")
    return "\n".join(out) + "\n"


def build_report_zip(
    *,
    engine: Any = None,
    clips: list[Any] | None = None,
    selftest: dict[str, Any] | None = None,
    kind: str = "snapshot",
) -> bytes:
    """Zip for bug reports: report.json/.txt, console.txt, logs/*."""
    from h3_console import get_console_lines
    from h3_paths import default_logs_dir

    report = {
        "kind": kind,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": system_facts(engine),
        "selftest": selftest,
        "recent_generations": recent_generations(clips or []),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "report.json", redact(json.dumps(report, indent=2, default=str)))
        archive.writestr("report.txt", _text_summary(report))
        archive.writestr(
            "console.txt", redact("\n".join(get_console_lines(5000)) + "\n"))
        log_dir = default_logs_dir()
        for name in ("server.log", "desktop.log", "error.log"):
            path = log_dir / name
            if path.is_file():
                archive.writestr(f"logs/{name}", redact(_file_tail(path, 4000)))
    return buffer.getvalue()
