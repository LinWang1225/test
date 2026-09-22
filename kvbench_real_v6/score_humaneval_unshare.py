#!/usr/bin/env python3
"""
Score saved HumanEval generations without Docker, using Linux namespaces + chroot.

This script:
- does NOT rerun model inference;
- executes generated HumanEval code inside an unprivileged user namespace;
- creates separate network/mount/PID/IPC/UTS namespaces;
- bind-mounts only Python runtime directories read-only into an ephemeral chroot;
- provides a private tmpfs /tmp;
- drops Linux capabilities and enables no_new_privs before generated code runs;
- applies CPU / address-space / file / fd / process limits;
- preserves the scores.jsonl / scores.summary.json / grading_protocol.json layout
  used by kvbench_real_v6/score_saved.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import atomic_json


SANDBOX_WRAPPER = r'''
import ctypes
import json
import os
import resource
import sys
import traceback

ROOT = os.path.realpath(sys.argv[1])
TIMEOUT = max(1, int(sys.argv[2]))

MS_RDONLY  = 1
MS_NOSUID  = 2
MS_NODEV   = 4
MS_NOEXEC  = 8
MS_REMOUNT = 32
MS_BIND    = 4096
MS_REC     = 16384
MS_PRIVATE = 1 << 18
PR_SET_NO_NEW_PRIVS = 38

libc = ctypes.CDLL(None, use_errno=True)
libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_ulong, ctypes.c_char_p]
libc.mount.restype = ctypes.c_int
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                       ctypes.c_ulong, ctypes.c_ulong]
libc.prctl.restype = ctypes.c_int

class CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

class CapData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32),
                ("permitted", ctypes.c_uint32),
                ("inheritable", ctypes.c_uint32)]

def b(x):
    return None if x is None else os.fsencode(x)

def mount(src, dst, fstype=None, flags=0, data=None):
    rc = libc.mount(b(src), b(dst), b(fstype), flags, b(data))
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), dst)

def bind_ro(src, dst):
    if not os.path.exists(src):
        return
    mount(src, dst, None, MS_BIND | MS_REC, None)
    mount(None, dst, None,
          MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV, None)

def drop_caps():
    # Linux capability ABI v3: clear effective/permitted/inheritable sets.
    hdr = CapHeader(0x20080522, 0)
    data = (CapData * 2)()
    capset = libc.capset
    if capset(ctypes.byref(hdr), ctypes.byref(data)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, "capset failed: " + os.strerror(err))
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, "PR_SET_NO_NEW_PRIVS failed: " + os.strerror(err))

def setup_root():
    # Nothing mounted below may propagate back to the host namespace.
    mount(None, "/", None, MS_REC | MS_PRIVATE, None)

    # Runtime is visible read-only. /home, /etc and project files are not exposed.
    for src in ("/usr", "/lib", "/lib64", "/bin"):
        if os.path.exists(src):
            dst = os.path.join(ROOT, src.lstrip("/"))
            os.makedirs(dst, exist_ok=True)
            bind_ro(src, dst)

    # Private writable /tmp only.
    tmp = os.path.join(ROOT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    mount("tmpfs", tmp, "tmpfs",
          MS_NOSUID | MS_NODEV | MS_NOEXEC,
          "size=64m,mode=1777")

    os.chroot(ROOT)
    os.chdir("/tmp")

def apply_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT, TIMEOUT + 1))
    mem = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    fsz = 2 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsz, fsz))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ValueError, OSError):
        pass
    os.umask(0o077)

def main():
    req = json.load(sys.stdin)
    setup_root()
    apply_limits()
    # Mount/chroot setup is complete; generated code receives no namespace caps.
    drop_caps()

    program = req["program"]
    try:
        glb = {"__name__": "__main__", "__file__": "<humaneval>"}
        exec(compile(program, "<humaneval>", "exec"), glb, glb)
        print(json.dumps({"passed": True}), flush=True)
    except BaseException as exc:
        print(json.dumps({
            "passed": False,
            "exception": type(exc).__name__,
            "error": str(exc)[-4096:],
            "traceback": traceback.format_exc()[-8192:],
        }), flush=True)

try:
    main()
except BaseException as exc:
    print("SANDBOX_SETUP_ERROR:" + repr(exc), file=sys.stderr, flush=True)
    raise SystemExit(125)
'''


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="kvbench-human-root-"))
    for d in ("usr", "lib", "lib64", "bin", "tmp"):
        (root / d).mkdir(exist_ok=True)
    # Generated code must not be able to create files at chroot '/'.
    root.chmod(0o555)
    return root


def cleanup_root(root: Path) -> None:
    try:
        root.chmod(0o755)
    except OSError:
        pass
    shutil.rmtree(root, ignore_errors=True)


def sandbox_cmd(python_bin: Path, root: Path, timeout: int) -> list[str]:
    return [
        "unshare",
        "--user", "--map-root-user",
        "--net", "--mount", "--ipc",
        "--pid", "--fork", "--uts",
        str(python_bin), "-I", "-B", "-c", SANDBOX_WRAPPER,
        str(root), str(timeout),
    ]


def sandbox_run(program: str, python_bin: Path, timeout: int) -> dict:
    root = make_root()
    cmd = sandbox_cmd(python_bin, root, timeout)
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
    }
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(
            json.dumps({"program": program}), timeout=timeout + 8
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        cleanup_root(root)
        return {
            "passed": False,
            "timeout": True,
            "returncode": proc.returncode,
            "stderr": "outer wall-clock timeout",
        }
    finally:
        cleanup_root(root)

    if proc.returncode == 125 or "SANDBOX_SETUP_ERROR:" in stderr:
        raise RuntimeError("sandbox infrastructure error:\n" + stderr[-8000:])

    lines = [line for line in stdout.splitlines() if line.strip()]
    if lines:
        try:
            result = json.loads(lines[-1])
            result["returncode"] = proc.returncode
            if stderr:
                result["stderr"] = stderr[-8192:]
            return result
        except json.JSONDecodeError:
            pass

    # os._exit(), abort(), signal termination, etc. are task failures.
    return {
        "passed": False,
        "runtime_abnormal": True,
        "returncode": proc.returncode,
        "stdout": stdout[-8192:],
        "stderr": stderr[-8192:],
    }


def full_namespace_check() -> None:
    cmd = [
        "unshare",
        "--user", "--map-root-user",
        "--net", "--mount", "--ipc",
        "--pid", "--fork", "--uts",
        "true",
    ]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError("required namespaces unavailable:\n" + p.stderr)


def sandbox_self_test(python_bin: Path) -> None:
    program = r'''
import os
import socket

assert 2 + 2 == 4
assert not os.path.exists("/home"), "host /home is visible"

try:
    with open("/should_not_write", "w") as f:
        f.write("x")
    raise AssertionError("chroot root is writable")
except OSError:
    pass

s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("1.1.1.1", 80))
    raise AssertionError("external network reachable")
except OSError:
    pass
finally:
    s.close()
'''
    r = sandbox_run(program, python_bin, 3)
    if not r.get("passed"):
        raise RuntimeError(
            "sandbox self-test failed:\n" +
            json.dumps(r, indent=2, ensure_ascii=False)
        )
    print("sandbox self-test: PASS")


def human_score(row: dict, python_bin: Path, timeout: int) -> dict:
    from evalscope.benchmarks.humaneval.humaneval_adapter import HumanevalAdapter

    text = row.get("final_answer_for_scoring", "")
    problem = row.get("original_record")
    if not isinstance(problem, dict) or not {"prompt", "test", "entry_point"} <= problem.keys():
        return {"score": None, "grading_status": "missing_original_problem"}
    if not text.strip():
        return {
            "score": False,
            "grading_status": "graded",
            "reason": "no_final_answer",
            "extracted_prediction": "",
        }

    completion = HumanevalAdapter._postprocess(text)
    program = (
        problem["prompt"] + completion + "\n" +
        problem["test"] + "\n" +
        f'check({problem["entry_point"]})'
    )
    try:
        result = sandbox_run(program, python_bin, timeout)
    except Exception as exc:
        return {
            "score": None,
            "grading_status": "sandbox_error",
            "error": str(exc),
            "extracted_prediction": completion,
        }
    return {
        "score": bool(result.get("passed")),
        "grading_status": "graded",
        "execution_result": result,
        "extracted_prediction": completion,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--python", type=Path, default=Path("/usr/bin/python3"),
                   help="Host system Python used to enter the namespace")
    p.add_argument("--human-timeout", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--self-test-only", action="store_true")
    a = p.parse_args()

    if a.human_timeout <= 0:
        raise ValueError("--human-timeout must be positive")
    if shutil.which("unshare") is None:
        raise RuntimeError("unshare not found")
    if not a.python.is_file():
        raise RuntimeError(f"Python not found: {a.python}")

    full_namespace_check()
    sandbox_self_test(a.python.resolve())
    if a.self_test_only:
        return

    suite = a.suite.expanduser().resolve()
    protocol_base = {
        "evalscope_version": importlib.metadata.version("evalscope"),
        "score_script": str(Path(__file__).resolve()),
        "score_script_sha256": sha256_file(Path(__file__).resolve()),
        "human_rule": "EvalScope HumanevalAdapter._postprocess + original HumanEval prompt/test",
        "human_sandbox": "linux_unshare_user_net_mount_pid_ipc_uts_plus_chroot",
        "human_python": str(a.python.resolve()),
        "human_test_timeout_s": a.human_timeout,
        "memory_limit": "RLIMIT_AS 512 MiB",
        "tmp_limit": "tmpfs 64 MiB",
        "network": "new network namespace; self-test verifies no external connect",
        "rootfs": "ephemeral chroot; /usr,/lib,/lib64,/bin bind-mounted read-only",
        "capabilities": "cleared before generated code; no_new_privs=1",
        "n_per_case": 1,
        "inference_rerun": False,
        "security_note": (
            "Linux namespace/chroot sandbox; stronger than venv/host exec, "
            "but not equivalent to a hardened VM/gVisor/Firecracker boundary "
            "against kernel exploits"
        ),
    }

    for point in sorted((suite / "points").glob("*")):
        states = sorted(point.glob("attempt*/state.json"))
        if not states:
            continue
        d = states[-1].parent
        state = json.loads(states[-1].read_text())
        if state["point"]["dataset"] != "humaneval":
            continue
        outputs = d / "requests.jsonl"
        if not outputs.exists():
            continue

        if (d / "scores.summary.json").exists():
            if not a.force:
                print("skip existing score:", d)
                continue
            stamp = str(time.time_ns())
            for name in ("scores.jsonl", "scores.summary.json", "grading_protocol.json"):
                f = d / name
                if f.exists():
                    f.rename(d / (name + ".backup_" + stamp))

        rows = [
            json.loads(line)
            for line in outputs.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        atomic_json(d / "grading_protocol.json", {
            **protocol_base,
            "outputs_sha256": sha256_file(outputs),
        })

        results = []
        with (d / "scores.jsonl").open("w", encoding="utf-8") as f:
            for i, row in enumerate(rows, 1):
                if not row.get("http_success"):
                    result = {"score": False, "grading_status": "generation_failed"}
                else:
                    result = human_score(row, a.python.resolve(), a.human_timeout)
                record = {
                    "case_id": row["case_id"],
                    "dataset": "humaneval",
                    "finish_reason": row.get("finish_reason"),
                    "length_capped": row["is_length_capped"],
                    **result,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                results.append(record)
                if i % 20 == 0 or i == len(rows):
                    print(f"{state['point']['id']}: {i}/{len(rows)}")

        n = state["point"]["number"]
        correct = sum(r.get("score") is True for r in results)
        unscored = sum(r.get("score") is None for r in results)
        complete = len(rows) == n and unscored == 0 and state["status"] == "ok"
        summary = {
            "quality_scoring_complete": complete,
            "quality_total_tasks": n,
            "quality_scored_tasks": len(results) - unscored,
            "quality_unscored_tasks": unscored,
            "quality_correct_tasks": correct,
            "accuracy_all_tasks": correct / n if complete else None,
            "humaneval_pass_at_1": correct / n if complete else None,
            "known_correct_fraction_lower_bound": correct / n if n else None,
            "quality_metric_note": (
                "n=1 per task; generated Python executed in Linux "
                "namespace/chroot sandbox; grader infrastructure errors remain unknown"
            ),
        }
        perf = json.loads((d / "stats.json").read_text()) if (d / "stats.json").exists() else {}
        wall = perf.get("request_window_s")
        summary["correct_tasks_per_second"] = correct / wall if complete and wall else None
        atomic_json(d / "scores.summary.json", summary)
        print(state["point"]["id"], summary)

    from run_real import collect
    collect(suite)
    print("updated:", suite / "summary.csv")


if __name__ == "__main__":
    main()
