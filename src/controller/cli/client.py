"""HTTP + SSE client for the Veda CLI.

Donor: veda/cli/client.py, read in full. Changes:
  - Liveness check moved from `/api/teams/status` (Teams, out of scope)
    to `/api/health` (controller/routes/health.py, Batch 9).
  - `upload_image`/`upload_images` and every `image_paths` parameter
    dropped — vision/image attachments are out of scope; `ChatRequest`/
    `StreamRequest` (schemas/chat.py) no longer have the field at all.
  - `status_snapshot()` drops the `vision`/`teams` probes (routes don't
    exist); keeps `gov`/`yolo`/`proac`.
  - Auto-spawn now launches `uvicorn bootstrap.app:app` directly (this repo's
    composition root, src/bootstrap/) instead of `python -m veda` (the donor's
    package shim).
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx

from core.logging_config import logger

_DEFAULT_URL = os.environ.get("VEDA_SERVER_URL", "http://127.0.0.1:8000")


def _pid_file(port: int) -> Path:
    """Per port, so a server on another port is never mistaken for this one."""
    return Path(__file__).resolve().parents[3] / "data" / f"veda-server-{port}.pid"


def register_server(port: int) -> None:
    """Called by a server this process is about to start in the foreground (`veda server`), before it loads any model.

    Loading takes about a minute and the port is not open until it is done, so without this a `veda` typed meanwhile finds
    nothing answering and starts a second server (a second copy of every model in memory). With it, `veda` sees a server
    that is still starting and waits for it. The marker is removed when this process exits.
    """
    path = _pid_file(port)
    pid = os.getpid()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(pid))
    except OSError:
        return

    def _clear() -> None:
        try:
            if path.read_text().strip() == str(pid):
                path.unlink()
        except OSError:
            pass

    atexit.register(_clear)


class CliError(RuntimeError):
    """Raised for any CLI-level failure (connection, server error, etc)."""


class VedaClient:
    """Thin synchronous client. Sufficient for argparse-driven CLIs; the
    server itself stays async — we just block on HTTP from the caller."""

    def __init__(self, server_url: str | None = None, timeout: float = 30.0) -> None:
        self.server_url = (server_url or _DEFAULT_URL).rstrip("/")
        # Only needed when this CLI talks to a Veda server on another machine (loopback needs no token).
        token = os.environ.get("VEDA_TOKEN", "").strip()
        self._http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {token}"} if token else None)
        self.how = ""   # what ensure_up did: reused | waited | started | restarted

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    # ---------- liveness / spawn ----------

    def is_up(self) -> bool:
        try:
            r = self._http.get(f"{self.server_url}/api/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    # ---------- stop / restart ----------

    @property
    def _port(self) -> int:
        return urlparse(self.server_url).port or 8000

    def _listener(self):
        """The psutil.Process listening on this client's port, or None."""
        import psutil

        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == self._port and conn.pid:
                    return psutil.Process(conn.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None
        return None

    @staticmethod
    def _is_veda_server(proc) -> bool:
        """Only ever stop a process that is recognisably a Veda server."""
        try:
            cmd = " ".join(proc.cmdline()).lower()
        except Exception:
            return False
        return "bootstrap.app:app" in cmd or "server:app" in cmd or ("veda" in cmd and " server" in cmd) or ("controller.cli" in cmd and "server" in cmd)

    def stop_server(self, *, wait_secs: float = 10.0) -> int | None:
        """Stop the Veda server on this client's port. Returns the stopped pid, or None if none was running.

        Tries the graceful ``POST /api/admin/shutdown`` first (lets the voice
        session and microphone shut down cleanly), then terminates, then kills.
        Raises :class:`CliError` if the port is held by something that is not a
        Veda server — that process is never touched.
        """
        import psutil

        proc = self._listener()
        if proc is None:
            return None
        if not self._is_veda_server(proc):
            raise CliError(
                f"Port {self._port} is in use by '{proc.name()}' (pid {proc.pid}), which is not a Veda server; "
                f"leaving it alone. Use --server-url to pick another port."
            )
        pid = proc.pid
        try:
            self._http.post(f"{self.server_url}/api/admin/shutdown", timeout=3.0)
        except Exception:
            pass  # not responding / old build: fall through to terminate
        try:
            proc.wait(timeout=min(6.0, wait_secs))
            return pid
        except psutil.TimeoutExpired:
            pass
        except psutil.NoSuchProcess:
            return pid
        try:
            for child in proc.children(recursive=True):
                child.terminate()
            proc.terminate()
            proc.wait(timeout=wait_secs)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=wait_secs)
        except psutil.NoSuchProcess:
            pass
        logger.info(f"Stopped existing Veda server (pid {pid}) on port {self._port}")
        return pid

    def _starting_pid(self) -> int | None:
        """The pid of a Veda server that an earlier `veda` spawned and that is still loading (the port is not open
        until startup, which includes loading and warming the model, has finished). None when there is none."""
        import psutil

        try:
            pid = int(_pid_file(self._port).read_text().strip())
            proc = psutil.Process(pid)
            return pid if proc.is_running() and self._is_veda_server(proc) else None
        except (OSError, ValueError, psutil.Error):
            return None

    @staticmethod
    def _log_tail(log_path: Path, lines: int = 8) -> str:
        try:
            rows = [r for r in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if r.strip()]
        except OSError:
            return ""
        return "\n".join(r[:240] for r in rows[-lines:])

    def ensure_up(self, *, allow_spawn: bool = True, wait_secs: float | None = None, restart: bool = False) -> int | None:
        """Make sure a Veda server is answering, then return.

        * Already running: use it (nothing is stopped or started).
        * Starting up (spawned by an earlier `veda` and still loading the model): wait for that one; never start a second.
        * Not running: spawn one and wait for it.
        ``restart=True`` first stops any existing Veda server so a fresh process (current code and config) is served.
        ``self.how`` says which of reused | waited | started | restarted happened. Returns the pid a restart stopped.

        The wait defaults to 240 s (VEDA_START_TIMEOUT) because the first answer comes only after the model is loaded and
        warmed (about a minute on a laptop, longer on a Pi). Raises :class:`CliError` when it does not come up in time,
        when ``allow_spawn=False`` and none is running, or at once when the server process dies (with its last log lines).
        """
        if wait_secs is None:
            try:
                wait_secs = float(os.environ.get("VEDA_START_TIMEOUT") or 240)
            except ValueError:
                wait_secs = 240.0
        stopped = self.stop_server() if (restart and allow_spawn) else None
        if self.is_up():
            self.how = "restarted" if stopped else "reused"
            return stopped
        if not allow_spawn:
            raise CliError(
                f"Veda server not reachable at {self.server_url}. "
                f"Start it with `veda server` or drop --no-spawn."
            )
        repo_root = Path(__file__).resolve().parents[3]
        log_path = repo_root / "data" / "veda-server.log"
        proc = None
        waiting_on = self._starting_pid()
        if waiting_on is not None:
            self.how = "waited"
            logger.info(f"A Veda server (pid {waiting_on}) is still starting; waiting for it instead of starting another")
        else:
            proc = self._spawn(repo_root, log_path)
            self.how = "restarted" if stopped else "started"
            waiting_on = getattr(proc, "pid", None)
            logger.info("Starting the Veda server: loading the model takes a minute or more the first time")

        began = time.monotonic()
        next_note = 15.0
        while time.monotonic() - began < wait_secs:
            if self.is_up():
                return stopped
            if self._died(proc, waiting_on):
                raise CliError(
                    "The Veda server stopped while starting. Last lines of "
                    f"{log_path}:\n{self._log_tail(log_path) or '(empty)'}"
                )
            if time.monotonic() - began >= next_note:
                logger.info(f"Still starting ({int(time.monotonic() - began)}s)...")
                next_note += 15.0
            time.sleep(0.5)
        raise CliError(
            f"Veda server did not answer within {int(wait_secs)}s (it may still be loading the model; run `veda` again to keep "
            f"waiting for it, or set VEDA_START_TIMEOUT higher). Log: {log_path}"
        )

    @staticmethod
    def _died(proc, pid: int | None) -> bool:
        if proc is not None and hasattr(proc, "poll"):
            return proc.poll() is not None
        if pid is None:
            return False
        import psutil

        try:
            return not psutil.Process(pid).is_running()
        except psutil.NoSuchProcess:
            return True
        except psutil.Error:
            return False

    def _bind_host(self) -> str:
        """Where a server started by this client listens: this machine only, unless devices.lan_enabled is on (the Pi's
        hardware profile turns it on) so a paired phone can reach the server that `veda` / `veda --restart` starts."""
        from core.config import load_full_config

        if load_full_config().devices.lan_enabled:
            return "0.0.0.0"
        return urlparse(self.server_url).hostname or "127.0.0.1"

    def _spawn(self, repo_root: Path, log_path: Path):
        src_dir = repo_root / "src"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"   # log lines reach data/veda-server.log as they happen, not in blocks
        host = self._bind_host()
        env["VEDA_BIND_HOST"] = host   # read by /api/devices/pairing-code so `veda pair` can warn when nothing can reach us
        port = str(self._port)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "wb")
        kwargs: dict[str, Any] = {
            "stdout": log_file, "stderr": log_file, "stdin": subprocess.DEVNULL,
            # The app is addressed by its full module path (bootstrap.app:app), so it resolves the same
            # whatever the working directory; src/ is on PYTHONPATH above.
            "env": env, "cwd": str(src_dir),
        }
        if sys.platform == "win32":
            # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP - hide the console
            # and isolate the child from our REPL's Ctrl-C.
            kwargs["creationflags"] = 0x08000000 | 0x00000200
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn", "bootstrap.app:app",
                    "--host", host, "--port", port, "--no-access-log",
                ],
                **kwargs,
            )
        finally:
            log_file.close()
        pid = getattr(proc, "pid", None)
        if pid:   # lets the next `veda` see that this server is still loading instead of starting a second one
            try:
                _pid_file(self._port).write_text(str(pid))
            except OSError:
                pass
        return proc

    # ---------- chat / stream ----------

    def stream_chat(self, message: str, *, from_voice: bool = False, system_context: str = "") -> Iterator[str]:
        """Yield text tokens as they arrive from /api/stream.

        Stops at the ``[DONE]`` sentinel. Caller can break early to cancel
        (the underlying disconnect signals the server to abort the run).
        """
        body: dict[str, Any] = {"message": message, "from_voice": from_voice, "system_context": system_context}

        with self._http.stream(
            "POST", f"{self.server_url}/api/stream", json=body,
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
        ) as resp:
            resp.raise_for_status()
            data_lines: list[str] = []
            for raw in resp.iter_lines():
                line = raw or ""
                if line == "":
                    if not data_lines:
                        continue
                    text = "\n".join(data_lines)
                    data_lines = []
                    if text == "[DONE]":
                        return
                    yield text
                    continue
                if line.startswith(":"):
                    continue  # SSE comment
                if line.startswith("data:"):
                    value = line[5:]
                    if value.startswith(" "):
                        value = value[1:]
                    data_lines.append(value)

    # ---------- persona ----------

    def get_persona(self) -> dict[str, Any]:
        r = self._http.get(f"{self.server_url}/api/persona")
        r.raise_for_status()
        return r.json()

    def set_persona(self, persona: str | None = None, completed: bool | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if persona is not None:
            body["persona"] = persona
        if completed is not None:
            body["completed"] = completed
        r = self._http.post(f"{self.server_url}/api/persona", json=body)
        r.raise_for_status()
        return r.json()

    # ---------- phone devices (loopback-only routes) ----------

    def create_pairing_code(self, *, allow_approve: bool = False, scopes: list[str] | None = None) -> dict[str, Any]:
        r = self._http.post(f"{self.server_url}/api/devices/pairing-code", json={"allow_approve": allow_approve, "scopes": scopes})
        r.raise_for_status()
        return r.json()

    def list_devices(self) -> list[dict[str, Any]]:
        r = self._http.get(f"{self.server_url}/api/devices")
        r.raise_for_status()
        return r.json()

    def revoke_device(self, device_id: str) -> bool:
        r = self._http.delete(f"{self.server_url}/api/devices/{device_id}")
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True

    # ---------- approvals ----------

    def pending_approvals(self) -> list[dict[str, Any]]:
        r = self._http.get(f"{self.server_url}/api/approval/pending")
        r.raise_for_status()
        return r.json().get("pending", []) or []

    def approve(self, request_id: str) -> None:
        r = self._http.post(f"{self.server_url}/api/approval/{request_id}/approve")
        r.raise_for_status()

    def deny(self, request_id: str) -> None:
        r = self._http.post(f"{self.server_url}/api/approval/{request_id}/deny")
        r.raise_for_status()

    # ---------- config ----------

    def yolo(self, on: bool) -> None:
        suffix = "on" if on else "off"
        r = self._http.post(f"{self.server_url}/api/approval/yolo/{suffix}")
        r.raise_for_status()

    def set_proactivity(self, level: str) -> None:
        r = self._http.post(f"{self.server_url}/api/config/proactivity", json={"level": level})
        r.raise_for_status()

    # ---------- privacy / mic ----------

    def privacy_status(self) -> dict[str, Any]:
        r = self._http.get(f"{self.server_url}/api/privacy/status")
        r.raise_for_status()
        return r.json()

    def voice_status(self) -> dict[str, Any]:
        r = self._http.get(f"{self.server_url}/api/voice/status")
        r.raise_for_status()
        return r.json()

    def set_voice_language(self, language: str) -> dict[str, Any]:
        r = self._http.post(f"{self.server_url}/api/voice/language", json={"language": language})
        r.raise_for_status()
        return r.json()

    def set_mute(self, muted: bool) -> dict[str, Any]:
        r = self._http.post(f"{self.server_url}/api/privacy/mute", json={"muted": muted})
        r.raise_for_status()
        return r.json()

    def toggle_mute(self) -> dict[str, Any]:
        r = self._http.post(f"{self.server_url}/api/privacy/mute/toggle")
        r.raise_for_status()
        return r.json()

    # ---------- tool-turn traces ----------

    def recent_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        r = self._http.get(f"{self.server_url}/api/trace", params={"limit": limit})
        r.raise_for_status()
        return r.json().get("traces", []) or []

    # ---------- online tool health ----------

    def lookup_health(self) -> dict[str, Any]:
        r = self._http.get(f"{self.server_url}/api/lookup/health", timeout=30.0)
        r.raise_for_status()
        return r.json()

    # ---------- tasks ----------

    def list_tasks(self, include_done: bool = False) -> list[dict[str, Any]]:
        r = self._http.get(f"{self.server_url}/api/tasks", params={"include_done": include_done})
        r.raise_for_status()
        return r.json().get("tasks", []) or []

    def add_task(self, title: str, notes: str = "") -> dict[str, Any]:
        r = self._http.post(f"{self.server_url}/api/tasks", json={"title": title, "notes": notes})
        r.raise_for_status()
        return r.json()

    def complete_task(self, task_id: int) -> dict[str, Any]:
        r = self._http.post(f"{self.server_url}/api/tasks/{task_id}/complete")
        r.raise_for_status()
        return r.json()

    # ---------- google sign-in ----------

    def google_status(self) -> dict[str, Any]:
        r = self._http.get(f"{self.server_url}/api/google/status", timeout=20.0)
        r.raise_for_status()
        return r.json()

    def google_reload(self) -> dict[str, Any]:
        r = self._http.post(f"{self.server_url}/api/google/reload", timeout=20.0)
        r.raise_for_status()
        return r.json()

    # ---------- reminders ----------

    def reminders_recent(self, after: int | None) -> dict[str, Any]:
        params = {} if after is None else {"after": after}
        r = self._http.get(f"{self.server_url}/api/reminders/recent", params=params, timeout=4.0)
        r.raise_for_status()
        return r.json()

    def reminders_status(self) -> dict[str, Any]:
        r = self._http.get(f"{self.server_url}/api/reminders/status", timeout=4.0)
        r.raise_for_status()
        return r.json()

    def set_reminders(self, **changes: Any) -> dict[str, Any]:
        r = self._http.put(f"{self.server_url}/api/reminders/config", json=changes, timeout=4.0)
        r.raise_for_status()
        return r.json()

    # ---------- status snapshot ----------

    def status_snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {"server_url": self.server_url}
        for key, path in [
            ("gov", "/api/governance/audit/stats"),
            ("yolo", "/api/approval/yolo"),
            ("proac", "/api/config/proactivity"),
        ]:
            try:
                r = self._http.get(f"{self.server_url}{path}", timeout=4.0)
                out[key] = r.json() if r.status_code == 200 else None
            except Exception:
                out[key] = None
        return out


@contextmanager
def open_client(server_url: str | None = None) -> Iterator[VedaClient]:
    c = VedaClient(server_url=server_url)
    try:
        yield c
    finally:
        c.close()