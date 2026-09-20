from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def supervise(config_path: str | Path) -> int:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    status_path = Path(config["status_file"])
    started = time.time()
    process = subprocess.Popen(
        config["command"],
        cwd=config["working_directory"],
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
    )
    _write_json_atomic(
        status_path,
        {
            "task_id": config["task_id"],
            "status": "RUNNING",
            "supervisor_pid": os.getpid(),
            "worker_pid": process.pid,
            "started_at_unix": started,
            "max_wall_time_seconds": config["max_wall_time_seconds"],
        },
    )
    try:
        exit_code = process.wait(timeout=float(config["max_wall_time_seconds"]))
        status = "COMPLETED" if exit_code == 0 else "FAILED"
        error = None
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        exit_code, status, error = None, "TIMED_OUT", "maximum wall time exceeded"
    _write_json_atomic(
        status_path,
        {
            "task_id": config["task_id"],
            "status": status,
            "supervisor_pid": os.getpid(),
            "worker_pid": process.pid,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "exit_code": exit_code,
            "error": error,
        },
    )
    if config["task_id"].startswith("gpu3c-"):
        from minillm_forge.experiments_gpu3c.reporting import refresh_background_reports

        refresh_background_reports(config["working_directory"])
    return 0 if status == "COMPLETED" else 1


def launch(
    repo: str | Path,
    *,
    task_id: str,
    command: list[str],
    max_wall_time_seconds: int,
    estimated_duration_seconds: float,
    expected_artifacts: list[str],
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    job_dir = repo / "runs/gpu3c/background" / task_id
    job_dir.mkdir(parents=True, exist_ok=True)
    lock_path = job_dir / "launch.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"task already launched: {task_id}") from exc
    os.close(descriptor)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    status_path = job_dir / "status.json"
    config_path = job_dir / "job.json"
    config = {
        "task_id": task_id,
        "command": command,
        "working_directory": str(repo),
        "max_wall_time_seconds": max_wall_time_seconds,
        "estimated_duration_seconds": estimated_duration_seconds,
        "expected_artifacts": expected_artifacts,
        "status_file": str(status_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "created_at_unix": time.time(),
    }
    _write_json_atomic(config_path, config)
    _write_json_atomic(status_path, {"task_id": task_id, "status": "LAUNCHING"})
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        creationflags = 0
    with (
        stdout_path.open("a", encoding="utf-8") as stdout,
        stderr_path.open("a", encoding="utf-8") as stderr,
    ):
        supervisor = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "minillm_forge.experiments_gpu3c.worker",
                "supervise",
                "--config",
                str(config_path),
            ],
            cwd=repo,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
    launched = {**config, "supervisor_pid": supervisor.pid}
    _write_json_atomic(job_dir / "launch.json", launched)
    return launched
