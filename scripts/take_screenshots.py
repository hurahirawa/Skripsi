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
TRAIN_TIMEOUT_MS = 600_000


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


def shot_section(page: Page, out: Path, name: str, heading: str, height: int = 1250, by_text: bool = False) -> Path:
    """Tangkapan satu bagian halaman: gulir ke `heading` lalu simpan satu layar penuh (untuk halaman yang panjang)."""
    page.set_viewport_size({"width": VIEWPORT["width"], "height": height})
    page.wait_for_timeout(500)
    target = page.get_by_text(heading).first if by_text else page.get_by_role("heading", name=heading).first
    target.evaluate("e => e.scrollIntoView()")
    page.wait_for_timeout(RENDER_WAIT_MS)
    path = out / name
    page.screenshot(path=str(path))
    print(f"  {path}")
    return path


def go(page: Page, link: str, heading: str) -> None:
    """Pindah halaman lewat navigasi sidebar (menjaga st.session_state sesi) dan tunggu judulnya."""
    page.get_by_role("link", name=link).first.click()
    page.get_by_role("heading", name=heading).first.wait_for()


def sample_upload(cfg: dict) -> Path:
    """CSV contoh unggahan: gabungan panggilan_*.csv format terbaru di reports/riwayat (hasil monitoring nyata/replay)."""
    import pandas as pd

    folder = resolve_path(cfg["paths"]["reports_dir"])
    files = [f for f in (folder / "riwayat").glob("panggilan_*.csv") if "status_akhir" in f.read_text("utf-8").splitlines()[0]]
    if not files:
        raise SystemExit("Tidak ada panggilan_*.csv format terbaru untuk contoh unggahan; jalankan monitoring dulu.")
    out = folder / "contoh_unggah" / "panggilan_contoh.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(out, index=False)
    return out


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
            page.get_by_role("heading", name="Pemetaan Karakteristik").wait_for()
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
            # panggilan ditutup saat sumber habis → toast muncul sebentar; ambil sebelum hilang (tanpa shot() yang menunggu)
            page.locator('[data-testid="stToast"]').first.wait_for(timeout=15_000)
            page.screenshot(path=str(out / "02b_toast_panggilan_selesai.png"))
            print(f"  {out / '02b_toast_panggilan_selesai.png'}")
            page.get_by_role("heading", name="Panggilan Selesai").wait_for()
            shot(page, out, "02c_dashboard_panggilan_selesai.png")
            page.get_by_role("button", name="Stop", exact=True).click()
            page.wait_for_timeout(1000)

            go(page, "Analisis Evaluasi Klaster", "Analisis & Evaluasi Klaster")
            shot(page, out, "03_analisis_klaster.png")
            shot_section(page, out, "03c_perbandingan_hdbscan_kmeans.png", "Perbandingan HDBSCAN vs K-Means")
            shot_section(page, out, "03d_grid_hdbscan.png", "Grid HDBSCAN (min_cluster_size × min_samples)")
            shot_section(page, out, "03e_centroid_batas_jarak.png", "Centroid dan batas jarak")
            page.get_by_text("Skala log").click()
            shot(page, out, "03b_analisis_klaster_skala_log.png")

            go(page, "Pengaturan Parameter", "Pengaturan Parameter")
            shot(page, out, "04_pengaturan.png")
            shot_section(page, out, "04b_pengaturan_perbarui_deskripsi_latih_ulang.png", "Perbarui Deskripsi Klaster")

            # Latih ulang: unggah contoh panggilan_*.csv (hasil monitoring nyata), pratinjau, latih, lihat perbandingan
            page.locator('input[type="file"][accept*=".csv"]').set_input_files(str(sample_upload(cfg)))
            page.get_by_text("Pratinjau").wait_for(timeout=30_000)
            page.wait_for_timeout(1500)
            shot_section(page, out, "04c_pengaturan_unggah_pratinjau.png", "Latih Ulang Model", height=1900)
            page.get_by_text("Saya mengerti ini membuat versi model baru").click()
            page.get_by_role("button", name="Latih Ulang Model").click()
            page.get_by_text("Model aktif vs model baru").wait_for(timeout=TRAIN_TIMEOUT_MS)
            shot_section(page, out, "04d_pengaturan_hasil_latih_ulang.png", "Model aktif vs model baru", height=1500, by_text=True)

            go(page, "Rekomendasi Ekspor", "Ekspor Laporan dan Saran Klaster")
            page.get_by_role("button", name="Buat PDF ringkasan").click()
            page.get_by_role("button", name="Unduh PDF ringkasan").wait_for()
            shot(page, out, "05_rekomendasi_ekspor.png")
            shot_section(page, out, "05b_rekomendasi_riwayat_laporan_sesi.png", "Riwayat sesi")
            shot_section(page, out, "05c_rekomendasi_panggilan_per_sesi.png", "Panggilan per sesi")
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
