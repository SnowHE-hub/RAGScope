"""Read only the identified GPU PID, its direct process tree and relevant paths."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pid", type=int, required=True)
args = parser.parse_args()
proc = Path("/proc") / str(args.pid)
argv = proc.joinpath("cmdline").read_bytes().decode("utf-8", errors="replace").strip("\0").split("\0")
secret_names = ("api_key", "api-key", "password", "secret", "access-token", "auth-token")
safe, redact = [], False
for arg in argv:
    if redact:
        safe.append("<redacted>")
        redact = False
    elif any(name in arg.lower() for name in secret_names):
        safe.append(arg.split("=", 1)[0] + "=<redacted>" if "=" in arg else arg)
        redact = "=" not in arg
    else:
        safe.append(arg)
environment = {}
for field in proc.joinpath("environ").read_bytes().split(b"\0"):
    if b"=" not in field:
        continue
    key, value = field.split(b"=", 1)
    name = key.decode(errors="replace")
    if name in {"VIRTUAL_ENV", "CONDA_PREFIX", "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}:
        environment[name] = value.decode(errors="replace")
fds = {}
for name in ("0", "1", "2"):
    try:
        fds[name] = os.readlink(proc / "fd" / name)
    except OSError as exc:
        fds[name] = type(exc).__name__
status = proc.joinpath("status").read_text()
selected_status = [line for line in status.splitlines() if line.startswith(("Name:", "State:", "PPid:", "Uid:", "VmRSS:", "Threads:"))]
result = {"pid": args.pid, "start_ticks": proc.joinpath("stat").read_text().split()[21],
          "argv": safe, "cwd": os.readlink(proc / "cwd"), "executable": os.readlink(proc / "exe"),
          "status": selected_status, "environment_whitelist": environment, "stdio": fds,
          "children": proc.joinpath(f"task/{args.pid}/children").read_text().strip().split()}
print(json.dumps(result, ensure_ascii=False, indent=2))
