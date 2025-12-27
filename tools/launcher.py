import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def _resolve_root() -> Path:
    env_root = os.getenv("IMAGE_SERVICE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir / ".venv-ui" / "Scripts" / "python.exe").exists() or (
            exe_dir / ".venv-lama" / "Scripts" / "python.exe"
        ).exists():
            return exe_dir
        if (exe_dir.parent / ".venv-ui" / "Scripts" / "python.exe").exists() or (
            exe_dir.parent / ".venv-lama" / "Scripts" / "python.exe"
        ).exists():
            return exe_dir.parent
        return exe_dir
    return Path(__file__).resolve().parents[1]


ROOT = _resolve_root()
LAMA_PY = ROOT / ".venv-lama" / "Scripts" / "python.exe"
LAMA_EXE = ROOT / ".venv-lama" / "Scripts" / "lama-cleaner.exe"
UI_PY = ROOT / ".venv-ui" / "Scripts" / "python.exe"
APP_PY = ROOT / "app.py"


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if _port_open(host, port, timeout=0.2):
            return True
        time.sleep(0.2)
    return False


def _resolve_host(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return host


def _pick_device(py_exe: Path, prefer: str) -> str:
    if prefer != "cuda":
        return prefer
    try:
        out = subprocess.check_output(
            [str(py_exe), "-c", "import torch; print('1' if torch.cuda.is_available() else '0')"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out == "1":
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Image Service launcher")
    parser.add_argument("--lama-host", default="0.0.0.0")
    parser.add_argument("--lama-port", type=int, default=8090)
    parser.add_argument("--lama-model", default="lama")
    parser.add_argument("--lama-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--ui-port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-lama", action="store_true")
    parser.add_argument("--skip-ui", action="store_true")
    args = parser.parse_args()

    if not args.skip_lama and not (LAMA_EXE.exists() or LAMA_PY.exists()):
        print(f"Missing {LAMA_EXE} (or {LAMA_PY})")
        return 1
    if not UI_PY.exists() and not args.skip_ui:
        print(f"Missing {UI_PY}")
        return 1
    if not APP_PY.exists() and not args.skip_ui:
        print(f"Missing {APP_PY}")
        return 1

    procs = []
    lama_host = args.lama_host
    lama_host_local = _resolve_host(lama_host)

    if not args.skip_lama:
        if _port_open(lama_host_local, args.lama_port):
            print(f"Port {args.lama_port} already in use, skip lama-cleaner")
        else:
            device = _pick_device(LAMA_PY, args.lama_device)
            if LAMA_EXE.exists():
                cmd = [
                    str(LAMA_EXE),
                    "--host",
                    lama_host,
                    "--port",
                    str(args.lama_port),
                    "--device",
                    device,
                    "--model",
                    args.lama_model,
                ]
            else:
                cmd = [
                    str(LAMA_PY),
                    "-m",
                    "lama_cleaner.cli",
                    "--host",
                    lama_host,
                    "--port",
                    str(args.lama_port),
                    "--device",
                    device,
                    "--model",
                    args.lama_model,
                ]
            print(f"Starting lama-cleaner on {lama_host}:{args.lama_port} (device={device})")
            procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))

    if not args.skip_ui:
        if _port_open("127.0.0.1", args.ui_port):
            print(f"Port {args.ui_port} already in use, skip UI")
        else:
            env = os.environ.copy()
            env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
            env["LAMA_SERVER"] = f"http://{lama_host_local}:{args.lama_port}"
            cmd = [str(UI_PY), str(APP_PY)]
            print(f"Starting UI on 127.0.0.1:{args.ui_port}")
            procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env))

    if not args.no_browser and not args.skip_ui:
        if _wait_for_port("127.0.0.1", args.ui_port, timeout=20):
            webbrowser.open(f"http://127.0.0.1:{args.ui_port}")

    if not procs:
        return 0

    try:
        while True:
            time.sleep(0.5)
            if all(p.poll() is not None for p in procs):
                return 0
    except KeyboardInterrupt:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
