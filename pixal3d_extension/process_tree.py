"""Cross-platform subprocess-group custody for extension GPU workers."""

from __future__ import annotations

import os
import signal
import subprocess
import time


def popen_process_group_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_tree(process: subprocess.Popen, *, timeout: float = 1.0, force: bool = False) -> None:
    """Terminate the owned process group and reap its direct child."""
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int):
        process.kill() if force else process.terminate()
        process.wait(timeout=timeout)
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return

    try:
        os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass
    if not force:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and _group_exists(pid):
            process.poll()
            time.sleep(0.02)
        if _group_exists(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.kill()
        process.wait()
