"""Dashboard sebagai jendela desktop: server Streamlit di background, tampil di jendela pywebview.

Server dihentikan saat jendela ditutup. Pemakaian: python scripts/desktop_app.py [--port 8502]
(live capture butuh hak Administrator: jalankan terminal/bat sebagai Administrator).
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parents[1]
WINDOW_TITLE = "VoIP QoS Monitor"
WINDOW_SIZE = (1500, 950)
SERVER_TIMEOUT_SEC = 90
STOP_TIMEOUT_SEC = 10


def free_port(preferred: int) -> int:
    """`preferred` jika kosong, selain itu port bebas yang dipilih OS."""
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return s.getsockname()[1]
    raise SystemExit("Tidak ada port bebas")


def wait_for_server(port: int, server: subprocess.Popen) -> None:
    """Tunggu endpoint health Streamlit siap; berhenti jika proses server mati."""
    deadline = time.time() + SERVER_TIMEOUT_SEC
    while time.time() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"Server Streamlit berhenti (kode {server.returncode})")
        try:
            if urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).read() == b"ok":
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("Server Streamlit tidak siap")


def stop_server(server: subprocess.Popen) -> None:
    """Hentikan server Streamlit (dan paksa jika tidak berhenti)."""
    server.terminate()
    try:
        server.wait(timeout=STOP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        server.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    port = free_port(args.port)
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "Home.py"), "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(port, server)
        webview.create_window(WINDOW_TITLE, f"http://localhost:{port}", width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])
        webview.start()  # blok sampai jendela ditutup
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
