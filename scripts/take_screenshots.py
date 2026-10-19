"""Tangkapan layar tiap halaman dashboard (Playwright headless) untuk Lampiran 7 naskah.

Menjalankan server Streamlit lokal, memutar replay PCAP demo di Dashboard, lalu menyimpan PNG ke
reports/screenshots/. Butuh: pip install -r requirements-dev.txt && python -m playwright install chromium

Pemakaian: python scripts/take_screenshots.py [--pcap data/sample_pcap/demo_voip.pcap] [--speed 3] [--port 8601]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from playwright.sync_api import Page, sync_playwright  # noqa: E402

from qos.capture.demo import write_demo_pcap  # noqa: E402
from qos.config import load_config, resolve_path  # noqa: E402

VIEWPORT = {"width": 1500, "height": 3200}  # tinggi besar: seluruh halaman muat tanpa scroll internal
MIN_HEIGHT = 760
RENDER_WAIT_MS = 2500  # beri waktu Plotly selesai menggambar
SERVER_TIMEOUT_SEC = 60
REPLAY_FINISH_TIMEOUT_MS = 90_000


def wait_for_server(port: int) -> None:
    """Tunggu endpoint health Streamlit siap."""
    deadline = time.time() + SERVER_TIMEOUT_SEC
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).read() == b"ok":
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("Server Streamlit tidak siap")


def fit_viewport(page: Page) -> None:
    """Sesuaikan tinggi viewport dengan tinggi konten halaman (min MIN_HEIGHT) agar tangkapan tanpa ruang kosong."""
    height = page.evaluate(
        """() => {
            const el = document.querySelector('[data-testid="stMainBlockContainer"]') || document.querySelector('section.main');
            return el ? Math.ceil(el.getBoundingClientRect().bottom + (document.querySelector('[data-testid="stMain"]')?.scrollTop || 0)) : 0;
        }"""
    )
    page.set_viewport_size({"width": VIEWPORT["width"], "height": max(MIN_HEIGHT, min(int(height) + 20, VIEWPORT["height"]))})
    page.wait_for_timeout(800)


def shot(page: Page, out: Path, name: str) -> Path:
    """Tunggu render, sesuaikan tinggi, lalu simpan PNG."""
    page.wait_for_timeout(RENDER_WAIT_MS)
    page.evaluate("() => document.querySelector('[data-testid=\"stMain\"]')?.scrollTo(0, 0)")  # kembali ke atas halaman
    fit_viewport(page)
    path = out / name
    page.screenshot(path=str(path))
    print(f"  {path}")
    return path


def shot_section(page: Page, out: Path, name: str, heading: str, height: int = 1250) -> Path:
    """Tangkapan satu bagian halaman: gulir ke `heading` lalu simpan satu layar penuh (untuk halaman yang panjang)."""
    page.set_viewport_size({"width": VIEWPORT["width"], "height": height})
    page.wait_for_timeout(500)
    page.get_by_role("heading", name=heading).first.evaluate("e => e.scrollIntoView()")
    page.wait_for_timeout(RENDER_WAIT_MS)
    path = out / name
    page.screenshot(path=str(path))
    print(f"  {path}")
    return path


def go(page: Page, link: str, heading: str) -> None:
    """Pindah halaman lewat navigasi sidebar (menjaga st.session_state sesi) dan tunggu judulnya."""
    page.get_by_role("link", name=link).first.click()
    page.get_by_role("heading", name=heading).first.wait_for()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", default="data/sample_pcap/demo_voip.pcap")
    parser.add_argument("--speed", type=float, default=3.0, help="kecepatan replay (× waktu asli)")
    parser.add_argument("--port", type=int, default=8601)
    parser.add_argument("--live-wait", type=float, default=7.0, help="detik menunggu sebelum tangkapan Dashboard")
    args = parser.parse_args()

    cfg = load_config()
    pcap = resolve_path(args.pcap)
    if not pcap.is_file():
        write_demo_pcap(pcap)
        print(f"PCAP demo (SINTETIS) dibuat: {pcap}")
    out = resolve_path(cfg["paths"]["reports_dir"]) / "screenshots"
    out.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "Home.py"), "--server.headless", "true",
         "--server.port", str(args.port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(args.port)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_context(viewport=VIEWPORT).new_page()
            page.goto(f"http://localhost:{args.port}")
            page.get_by_role("heading", name="Pemetaan Karakteristik VoIP").wait_for()
            print("Menyimpan tangkapan layar:")
            shot(page, out, "01_home.png")

            go(page, "Dashboard Overview", "Dashboard Overview")
            page.get_by_label("PCAP contoh (data/sample_pcap)").click()
            page.get_by_role("option", name=pcap.name).click()
            speed = page.get_by_label("Kecepatan replay (×, 0 = secepat mungkin)")
            speed.fill(str(args.speed))
            speed.press("Enter")
            page.wait_for_timeout(500)
            page.get_by_role("button", name="Start", exact=True).click()
            page.wait_for_timeout(int(args.live_wait * 1000))
            shot(page, out, "02_dashboard_replay.png")

            page.get_by_text("Replay selesai").wait_for(timeout=REPLAY_FINISH_TIMEOUT_MS)
            page.get_by_role("button", name="Stop", exact=True).click()
            page.wait_for_timeout(1000)

            go(page, "Analisis Evaluasi Klaster", "Analisis & Evaluasi Klaster")
            shot(page, out, "03_analisis_klaster.png")
            page.get_by_text("Skala log").click()
            shot(page, out, "03b_analisis_klaster_skala_log.png")

            go(page, "Pengaturan Parameter", "Pengaturan Parameter & Ambang Batas")
            shot(page, out, "04_pengaturan.png")

            go(page, "Rekomendasi Ekspor", "Rekomendasi Mitigasi & Ekspor Laporan")
            page.get_by_role("button", name="Buat PDF ringkasan").click()
            page.get_by_role("button", name="Unduh PDF ringkasan").wait_for()
            shot(page, out, "05_rekomendasi_ekspor.png")
            shot_section(page, out, "05b_rekomendasi_riwayat_laporan_sesi.png", "Riwayat sesi")
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
