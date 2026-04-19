# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM server lifecycle management."""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config, ServerConfig
from .resolved import ResolvedRun

# Track all spawned server process groups for cleanup on exit.
_active_pgids: set[int] = set()
_cleanup_lock = threading.Lock()


def _cleanup_servers() -> None:
    """Kill all tracked server process groups (SIGTERM then SIGKILL)."""
    with _cleanup_lock:
        for pgid in list(_active_pgids):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if _active_pgids:
            try:
                time.sleep(3)
            except (KeyboardInterrupt, SystemExit):
                pass
        for pgid in list(_active_pgids):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        _active_pgids.clear()


# Deferred handler installation to avoid module-level side effects.
_atexit_installed = False
_sigterm_installed = False
_handler_lock = threading.Lock()


def _install_handlers() -> None:
    """Register atexit and SIGTERM handlers (once, on first Server use).

    Signal handlers can only be installed from the main thread, so the
    SIGTERM handler is deferred until a main-thread call occurs.  The
    atexit handler is always safe to register from any thread.
    """
    global _atexit_installed, _sigterm_installed
    with _handler_lock:
        if not _atexit_installed:
            atexit.register(_cleanup_servers)
            _atexit_installed = True

        if not _sigterm_installed and \
                threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGHUP):
                orig = signal.getsignal(sig)

                def _make_handler(orig_handler):
                    def _handler(signum, frame):
                        _cleanup_servers()
                        if callable(orig_handler):
                            orig_handler(signum, frame)
                        else:
                            raise SystemExit(1)
                    return _handler

                signal.signal(sig, _make_handler(orig))
            _sigterm_installed = True


class _LogTail:
    """Follow a log file in a background thread, collecting prefixed lines."""

    def __init__(self, log_path: Path, prefix: str):
        self._path = log_path
        self._prefix = prefix
        self._stop_ev = threading.Event()
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._pos = 0

    def start(self) -> None:
        self._pos = self._path.stat().st_size if self._path.exists() else 0
        self._thread.start()

    def stop(self) -> None:
        self._stop_ev.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_ev.is_set():
            try:
                if self._path.exists():
                    size = self._path.stat().st_size
                    if size > self._pos:
                        with open(self._path, errors="replace") as f:
                            f.seek(self._pos)
                            new_data = f.read()
                            self._pos = f.tell()
                        if new_data:
                            with self._lock:
                                for line in new_data.splitlines():
                                    self._lines.append(
                                        f"{self._prefix}[server] {line}")
            except OSError:
                pass
            self._stop_ev.wait(0.5)

    def drain(self) -> list[str]:
        """Return and clear collected lines."""
        with self._lock:
            lines = self._lines
            self._lines = []
            return lines


class Server:
    """Context manager for vllm server lifecycle.

    Starts the server in its own process group, waits for health,
    runs a sanity check, and kills on exit.

    When prefix is set (parallel mode), output is buffered and flushed
    at milestones (server ready, sanity check, stop) to avoid interleaving.
    """

    def __init__(self, resolved: ResolvedRun, config: Config,
                 log_dir: Path, prefix: str = "",
                 flush_jitter: float = 0.0):
        _install_handlers()
        self.resolved = resolved
        self.repo_dir = resolved.repo_dir
        self.config = config
        self.run = resolved.run
        self.server: ServerConfig = resolved.server
        self.log_path = log_dir / f"{resolved.label}_server.log"
        self.prefix = prefix
        self._buffered = flush_jitter > 0
        self._flush_jitter = flush_jitter
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self._buf: list[str] = []
        self._log_tail: _LogTail | None = None

    def _log(self, msg: str) -> None:
        for line in msg.splitlines():
            text = f"{self.prefix}{line}"
            if self._buffered:
                self._buf.append(text)
            else:
                print(text, flush=True)

    def _flush(self) -> None:
        """Print all buffered lines atomically."""
        if not self._buf and not self._log_tail:
            return
        if self._flush_jitter:
            time.sleep(self._flush_jitter)
        # Drain server log lines into buffer
        if self._log_tail:
            self._buf.extend(self._log_tail.drain())
        if self._buf:
            print("\n".join(self._buf), flush=True)
            self._buf.clear()

    def __enter__(self) -> Server:
        self._start()
        try:
            self._wait_health()
            self._sanity_check()
        except BaseException:
            self._stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._stop()

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.server.port}"

    def _build_serve_cmd(self) -> list[str]:
        return self.server.build_serve_cmd(
            model=self.config.project.model,
            vllm_bin=str(self.resolved.vllm_bin),
        )

    def _start(self) -> None:
        srv = self.server
        self._log(f"{'─' * 46}")
        self._log(f"  Run: {self.run.label}")
        self._log(f"  Branch: {self.run.branch}"
                  f"{f' @ {self.run.commit}' if self.run.commit else ''}")
        self._log(f"  Model: {self.config.project.model}")
        if srv.compilation_config:
            self._log(f"  Compilation config: "
                      f"{json.dumps(srv.compilation_config)}")
        self._log(f"{'─' * 46}")

        serve_cmd = self._build_serve_cmd()
        self._log(f"$ {' '.join(serve_cmd)}")

        if self.server.clear_caches:
            from .runner import _clear_jit_caches
            self._log("Clearing JIT caches...")
            _clear_jit_caches()

        env = None
        if self.server.env:
            env = {**os.environ, **self.server.env}
        if self.server.log_level:
            env = {**(env or os.environ)}
            env["VLLM_LOGGING_LEVEL"] = self.server.log_level.upper()
        if self.config.project.isolate_flashinfer_cache:
            # Per-venv flashinfer JIT cache to prevent cross-venv symbol
            # conflicts (e.g. cu12 vs cu13 compiled .so files).
            env = {**(env or os.environ),
                   "FLASHINFER_WORKSPACE_BASE":
                       str(self.repo_dir / ".flashinfer")}

        self._log_fh = open(self.log_path, "w")
        try:
            self._proc = subprocess.Popen(
                serve_cmd,
                cwd=self.repo_dir,
                env=env,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            self._log_fh.close()
            self._log_fh = None
            raise
        # Track for cleanup on crash/exit
        with _cleanup_lock:
            _active_pgids.add(os.getpgid(self._proc.pid))
        self._log(f"Server started (PID={self._proc.pid})")
        self._flush()

    def _wait_health(self) -> None:
        self._log(f"Waiting for server on port {self.server.port}...")
        self._flush()

        # Start tailing server log
        self._log_tail = _LogTail(self.log_path, self.prefix)
        self._log_tail.start()

        start = time.monotonic()
        last_status = 0.0
        try:
            while True:
                try:
                    req = urllib.request.Request(f"{self.base_url}/health")
                    urllib.request.urlopen(req, timeout=2)
                    self._log_tail.stop()
                    elapsed = time.monotonic() - start
                    self._log(f"Server ready after {elapsed:.0f}s")
                    self._flush()
                    self._log_tail = None
                    return
                except (urllib.error.URLError, OSError, TimeoutError):
                    pass

                if self._proc and self._proc.poll() is not None:
                    self._log_tail.stop()
                    elapsed = time.monotonic() - start
                    self._log(
                        f"ERROR: Server process exited prematurely "
                        f"after {elapsed:.0f}s "
                        f"(rc={self._proc.returncode})")
                    self._dump_log_tail()
                    self._flush()
                    self._log_tail = None
                    raise RuntimeError("Server exited prematurely")

                time.sleep(5)
                elapsed = time.monotonic() - start

                # Periodically flush buffered server log
                self._flush()

                # Log elapsed time every 60s for visibility
                if elapsed - last_status >= 60:
                    self._log(f"Still waiting for server... "
                              f"({elapsed:.0f}s elapsed)")
                    self._flush()
                    last_status = elapsed
        except (KeyboardInterrupt, RuntimeError):
            if self._log_tail:
                self._log_tail.stop()
                self._log_tail = None
            raise

    def _sanity_check(self) -> None:
        """Send a single completion request to verify the server works."""
        self._log("Sanity check: generating one response...")
        payload = json.dumps({
            "model": self.config.project.model,
            "prompt": "Hello",
            "max_tokens": 32,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["text"][:80].replace("\n", "\\n")
            self._log(f"Response: {text}")
            self._log("Sanity check passed.")
        except Exception as e:
            self._log(f"WARNING: Sanity check failed: {e}")
        self._flush()

    def warmup(self, num_prompts: int) -> None:
        """Send throwaway requests to stabilize performance before benchmark."""
        if num_prompts <= 0:
            return
        self._log(f"Warming up with {num_prompts} prompt(s)...")
        for i in range(num_prompts):
            payload = json.dumps({
                "model": self.config.project.model,
                "prompt": f"Warmup request {i}",
                "max_tokens": 32,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/v1/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp.read()
            except Exception as e:
                self._log(f"WARNING: Warmup request {i} failed: {e}")
        self._log("Warmup complete.")
        self._flush()

    def _stop(self) -> None:
        if self._proc is None:
            return
        pid = self._proc.pid
        self._log(f"Stopping server (PID={pid})...")
        # Block SIGINT during cleanup so Ctrl+C can't leave orphaned servers
        pgid = None
        prev_handler = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except ValueError:
            # Not in main thread; can't change signal handler
            prev_handler = None
        try:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._log("Server didn't stop after SIGTERM, "
                              "sending SIGKILL...")
                    os.killpg(pgid, signal.SIGKILL)
                    self._proc.wait(timeout=5)
            except (ProcessLookupError, PermissionError):
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            # Untrack from cleanup
            if pgid is not None:
                with _cleanup_lock:
                    _active_pgids.discard(pgid)
            self._proc = None
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None
            self._log("Server stopped.")
            self._flush()
            time.sleep(3)
        finally:
            if prev_handler is not None:
                try:
                    signal.signal(signal.SIGINT, prev_handler)
                except ValueError:
                    pass

    def _dump_log_tail(self, lines: int = 30) -> None:
        """Print last N lines of server log."""
        if self.log_path.exists():
            self._log("Server log tail:")
            all_lines = self.log_path.read_text().splitlines()
            for line in all_lines[-lines:]:
                self._log(f"  {line}")
