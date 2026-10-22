# Pemetaan Karakteristik VoIP Traffic Network untuk Identifikasi QoS menggunakan HDBSCAN

Dashboard web lokal (Streamlit) untuk administrator jaringan yang:

1. melatih model **HDBSCAN** secara offline dari statistik aliran (unsupervised);
2. memberi label tiap klaster **QoS Baik / Cukup / Buruk** berdasarkan ITU-T G.114 (delay), ITU-T Y.1541 Kelas 0/1 (jitter), serta ITU-T G.1010 (conversational voice, PLR < 3%) dan panduan Cisco VoIP (PLR ≤ 1%) untuk packet loss, dengan noise (-1) sebagai **Anomali**;
3. memetakan trafik per jendela waktu ke klaster yang sudah terbentuk (`hdbscan.approximate_predict`) lalu mewarisi label klaster;
4. menampilkan label, visualisasi, dan rekomendasi mitigasi, serta menyimpan riwayat sesi ke CSV dan PDF.

Metodologi CRISP-DM. Tanpa database relasional (semua berbasis berkas). Spesifikasi lengkap ada di [`PRD.md`](PRD.md) (sumber kebenaran proyek ini); aturan kerja ada di [`CLAUDE.md`](CLAUDE.md).

> **Catatan penting tentang data dan hasil.** Data latih adalah CIC-Darknet2020 (`Darknet.CSV`, flow berlabel `VOIP`), bukan trafik yang dirancang khusus untuk VoIP. Setelah filter (tanpa Tor, fwd dan bwd ≥ 5 paket) tersisa **247 flow**. Model aktif saat ini tidak memiliki klaster "QoS Baik". Lihat bagian [Keterbatasan](#keterbatasan-dan-catatan-metodologi) sebelum mengutip angka.

## Daftar isi

- [Prasyarat](#prasyarat) · [Instalasi (Windows)](#instalasi-windows) · [Data](#data)
- [Pelatihan](#pelatihan) · [Replay PCAP](#replay-pcap) · [Live capture](#live-capture) · [Dashboard](#dashboard)
- [Konfigurasi](#konfigurasi) · [Artefak dan versi model](#artefak-dan-versi-model) · [Riwayat dan ekspor](#riwayat-dan-ekspor)
- [Tangkapan layar](#tangkapan-layar) · [Pengujian](#pengujian) · [Struktur folder](#struktur-folder) · [Keterbatasan](#keterbatasan-dan-catatan-metodologi)

## Prasyarat

| Komponen | Keterangan |
|---|---|
| OS | Windows 11 (utama) atau Ubuntu 22.04 |
| Python | 3.10 atau lebih baru (PRD menargetkan 3.10/3.11; pengembangan dan uji berjalan di **3.12**) |
| Npcap | **Wajib untuk live capture** di Windows (replay PCAP tidak membutuhkannya) |
| Wireshark + TShark | Hanya untuk backend `pyshark` (opsional); backend default `scapy` tidak membutuhkannya |
| Hak akses | Live capture bisa membutuhkan Administrator (Windows) atau root / `CAP_NET_RAW` (Linux), lihat [Live capture](#live-capture) |
| Internet | Hanya untuk instalasi paket; aplikasi berjalan tanpa internet |

## Instalasi (Windows)

Jalankan di PowerShell dari folder proyek.

```powershell
# 1. Lingkungan virtual dan dependensi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                      # agar `python scripts/...` menemukan paket `qos`
pip install -r requirements-dev.txt   # opsional: pytest dan Playwright (tangkapan layar)
python -m playwright install chromium # opsional: hanya untuk scripts/take_screenshots.py
```

Jika PowerShell menolak mengaktifkan venv: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Npcap (untuk live capture)

1. Unduh dan pasang Npcap dari <https://npcap.com> (opsi bawaan sudah cukup).
2. Jika saat instalasi **"Restrict Npcap driver's access to Administrators only"** dicentang, terminal harus dijalankan sebagai Administrator. Jika tidak dicentang, capture jalan tanpa Administrator.
3. Periksa: `python scripts/live_capture.py --check --backend scapy`.

### Wireshark dan TShark (hanya untuk backend `pyshark`)

1. Pasang Wireshark dari <https://www.wireshark.org> dan pastikan komponen **TShark** ikut terpasang (Npcap dipasang oleh installer Wireshark bila belum ada).
2. Jika `tshark.exe` tidak ada di `PATH` atau di `C:\Program Files\Wireshark`, isi `monitoring.tshark_path` di `config/default_config.yaml` (atau `user_config.yaml`).
3. Pilih backend: `monitoring.capture_backend: pyshark`, atau di sidebar dashboard (mode Live → *Backend capture*).
4. Periksa: `python scripts/live_capture.py --check --backend pyshark`.

> Status pengujian: jalur `scapy` sudah diuji pada Wi-Fi nyata. Jalur `pyshark` baru diuji dengan objek tiruan; uji dengan TShark sungguhan belum dijalankan.

### Ubuntu 22.04 (ringkas)

```bash
sudo apt install python3-venv libpcap0.8 tshark   # tshark hanya untuk pyshark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
sudo setcap cap_net_raw,cap_net_admin=eip "$(readlink -f .venv/bin/python3)"   # capture tanpa root
```

## Data

Letakkan CSV di `data/raw/` (folder ini **tidak** di-commit).

| Dataset | Status | Keterangan |
|---|---|---|
| CIC-Darknet2020 (`Darknet.CSV`) | dipakai | Format `darknet` terdeteksi otomatis; filter VoIP memakai kolom kategori `Label.1 == VOIP` |
| CIC-IDS2017 | didukung loader | Format `cic`; filter VoIP mode `port` (UDP + port SIP/RTP) |
| BCCC-CIC-IDS2017 | didukung loader | Format `bccc`; satuan detik; nama kolom `packet_IAT_mean` belum diuji pada data nyata |

Laporan eksplorasi data:

```powershell
python scripts/eda.py --config config/default_config.yaml          # reports/eda/eda_m1.md
python scripts/sensitivity.py --config config/default_config.yaml  # sensitivitas filter sesi
python scripts/report_final.py --config config/default_config.yaml # konfigurasi final + distribusi level ITU per flow
```

## Pelatihan

```powershell
python scripts/train.py --config config/default_config.yaml            # grid search + simpan versi baru
python scripts/train.py --config config/default_config.yaml --no-grid  # pakai min_cluster_size/min_samples dari config
python scripts/train.py --config config/default_config.yaml --activate # jadikan versi ini aktif
python scripts/train.py --source log                                   # latih dari log jendela monitoring
```

Pipeline: baca CSV → filter VoIP → buang `exclude_traffic_types` (default `Tor`) dan flow dengan fwd/bwd < `min_session` (default 5/5) → cleaning (±inf→NaN, duplikat, negatif, imputasi median) → hitung `fai_pct` → `log1p` pada `iat_mean_us` dan `iat_std_us` → `StandardScaler` → grid search HDBSCAN → evaluasi (Silhouette tanpa noise, CHI, Noise Ratio) → profil klaster dan label ITU → simpan artefak.

- Fitur klaster default (Varian C): `[iat_mean_us, iat_std_us, fai_pct]`, dengan `log1p` pada dua kolom pertama dan **tanpa** log pada `fai_pct`. Proksi delay = *Flow IAT Mean*; jitter = *Flow IAT Std*; loss = FAI.
- Pemilihan grid: Silhouette tertinggi, lalu CHI, lalu `min_cluster_size` terkecil, lalu `min_samples` terkecil. Aturan dan kandidat seri dicatat di `train_meta.json`.
- Hasil reprodusibel (`random_state: 42`): latih ulang dari CSV dengan config default menghasilkan parameter, skor, klaster, dan label yang identik dengan versi `20261019_175647`.
- Versi baru **tidak** menjadi aktif kecuali dengan `--activate` (atau bila belum ada versi aktif sama sekali).

Hasil model aktif `20261019_175647` (247 flow): 8 klaster + noise, Silhouette 0,8866 (Strong Structure), CHI 7.554,68, Noise Ratio 13,77%, `min_cluster_size` 5, `min_samples` 5.

## Replay PCAP

Replay membaca `.pcap`/`.pcapng` dengan timing asli dibagi `replay_speed`, tanpa Npcap dan tanpa hak khusus.

```powershell
python scripts/replay_pcap.py --make-demo                       # buat data/sample_pcap/demo_voip.pcap (SINTETIS, bukan data nyata)
python scripts/replay_pcap.py --pcap data/sample_pcap/<berkas>.pcap --speed 0   # 0 = secepat mungkin
python scripts/replay_pcap.py --pcap <berkas> --speed 0 --compare               # bandingkan feature_mode window vs cumulative
```

Keluarannya: CSV jendela di `reports/riwayat/jendela_<id_sesi>.csv` dan ringkasan sesi `riwayat_pemantauan_<waktu>.csv` (ditulis saat Stop).

`monitoring.feature_mode`: `cumulative` (default; statistik sejak awal flow) atau `window` (hanya paket dalam jendela). Mode `window` memberi FAI ≈ 0,13% pada RTP tanpa loss karena efek batas jendela, sehingga `cumulative` dijadikan default.

## Live capture

```powershell
python scripts/live_capture.py --check                               # periksa prasyarat (tshark, Npcap, hak akses)
python scripts/live_capture.py --list-interfaces
python scripts/live_capture.py --backend scapy --interface "Wi-Fi" --seconds 30
```

- **Backend default `scapy`** (butuh Npcap). `pyshark` adalah opsi (butuh TShark). Tidak ada fallback otomatis antar-backend; jika prasyarat tidak ada, aplikasi berhenti dengan pesan yang menyebut langkah perbaikannya.
- **Filter BPF default**: `udp and not port 443 and not port 53 and not port 123 and not port 5353 and not port 1900`. Port tersebut (QUIC, DNS, NTP, mDNS, SSDP) di luar distribusi latih dan hanya menghasilkan label Anomali yang menyesatkan.
- Jendela ditutup berdasarkan timestamp paket, ditambah detak `live_poll_sec` saat tidak ada paket.
- Flow/jendela dengan fwd atau bwd < `monitoring.min_session_live` (default 5/5) tidak diprediksi: statusnya **"Data Belum Cukup"**, tetap dicatat di CSV, dan tidak dihitung sebagai Anomali maupun dalam `kritis_ratio`.
- Hanya statistik agregat dan alamat IP flow yang disimpan di `reports/riwayat/`; payload tidak pernah dibaca. Folder `reports/` tidak di-commit.

## Dashboard

```powershell
streamlit run app/Home.py
```

| Halaman | Isi |
|---|---|
| Home | Ringkasan model aktif dan alur kerja |
| Dashboard Overview | 5 kartu (Total, Baik, Cukup, Buruk, Anomali), PCA scatter live, bar chart vs ambang ITU, tabel jendela terbaru, rekomendasi sesi berjalan; auto-refresh |
| Analisis & Evaluasi Klaster | Silhouette (dengan interpretasi), CHI, Noise Ratio, tabel profil, box plot 3 metrik (toggle skala log), sampel data latih |
| Pengaturan Parameter | Slider HDBSCAN, ambang ITU, Save/Reset, **Terapkan Ulang Pelabelan**, **Latih Ulang Model**, daftar dan aktivasi versi model |
| Rekomendasi & Ekspor | Alert klaster bermasalah dan sesi Kritis dengan saran mitigasi, riwayat sesi, unduh CSV dan PDF |

Sidebar (semua halaman): versi model aktif, mode Replay PCAP / Live, sumber, Start/Stop. Latih ulang dan pergantian model aktif dinonaktifkan selama monitoring berjalan.

## Konfigurasi

| Berkas | Fungsi |
|---|---|
| `config/default_config.yaml` | Seluruh parameter dan ambang (tidak ada angka ambang di kode) |
| `config/user_config.yaml` | Dibuat oleh *Save Configuration*; menimpa default; tidak di-commit; dihapus oleh *Reset to Default* |
| `config/recommendations.yaml` | Aturan saran mitigasi per metrik (Delay, Jitter, Packet Loss, Anomali) |

Ambang ITU default: delay 150/400 ms, jitter 50 ms, loss 1/3% (≤1% baik, 1–3% cukup, >3% buruk; sumber ITU-T G.1010 + panduan Cisco VoIP). **Hasil skripsi memakai ambang default**; mengubah ambang di UI hanya untuk eksplorasi.

Satuan internal: delay dan jitter dalam ms, loss dalam %; data CIC dalam µs dibagi 1000.

## Artefak dan versi model

Setiap pelatihan membuat `artifacts/<YYYYMMDD_HHMMSS>/` dan `artifacts/current.txt` menunjuk versi aktif. Isi satu versi:

`hdbscan_model.joblib`, `scaler.joblib`, `pca.joblib`, `cluster_profile.json`, `train_meta.json`, `grid_results.csv`, `training_sample.csv`.

- **Latih ulang** (halaman Pengaturan atau CLI) tidak pernah menimpa versi lama dan tidak mengubah `current.txt`; versi baru aktif hanya setelah tombol *Aktifkan model ini* (atau `--activate`).
- **Sumber latih ulang**: CSV (Darknet/CIC) atau log jendela monitoring. Untuk log: baris *Data Belum Cukup* wajib dikecualikan; baris Anomali, baris `diimputasi = true`, dan baris dengan `feature_mode` selain yang diatur (default `cumulative`) dikecualikan sesuai `retrain.log_source`; CSV lama tanpa kolom `feature_mode`/`diimputasi` dianggap tidak diketahui dan dibuang; jumlah minimum baris (`min_rows`, default 200) dicek sebelum pelatihan.
- **Terapkan Ulang Pelabelan** menghitung ulang label dari statistik tersimpan dengan ambang baru, **tanpa melatih ulang HDBSCAN**, dan menghasilkan versi baru (model, scaler, PCA disalin tanpa berubah).
- `cluster_profile.json` memuat `metrik_penyebab` per klaster (mis. `"delay;loss"`); rekomendasi membaca kolom ini.

## Riwayat dan ekspor

- `reports/riwayat/jendela_<id_sesi>.csv` — ditambah tiap jendela: `waktu_jendela, id_sesi, flow_id, delay_ms, jitter_ms, fwd, bwd, fai_pct, cluster_id, membership_prob, label, kategori_tindakan, metrik_penyebab, pc1, pc2, feature_mode, diimputasi`. `id_sesi` berakhiran 6 karakter acak.
- `reports/riwayat/riwayat_pemantauan_<waktu>.csv` — ringkasan sesi saat Stop (status "Kritis" bila proporsi QoS Buruk + Anomali > `kritis_ratio`, tidak menghitung *Data Belum Cukup*).
- `reports/exports/laporan_<id_sesi>.csv|pdf` — dibuat dari halaman *Rekomendasi & Ekspor* (PDF: info sesi dan model, distribusi label, rekomendasi, tabel aliran, profil klaster).

## Tangkapan layar

Untuk Lampiran 7. Skrip menjalankan server lokal, memutar replay PCAP demo di Dashboard, lalu menyimpan PNG ke `reports/screenshots/`:

```powershell
pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/take_screenshots.py [--pcap data/sample_pcap/<berkas>.pcap] [--speed 3]
```

Berkas: `01_home`, `02_dashboard_replay`, `03_analisis_klaster`, `03b_analisis_klaster_skala_log`, `04_pengaturan`, `05_rekomendasi_ekspor`, `05b_rekomendasi_riwayat_laporan_sesi`. Tabel riwayat pada tangkapan memuat semua sesi yang ada di `reports/riwayat/`; kosongkan atau pindahkan folder itu lebih dulu bila ingin riwayat yang bersih.

## Pengujian

```powershell
pytest                       # seluruh uji (±240 uji, ±20 detik)
pytest tests/test_profile.py # uji Lampiran 5 (PRD §13)
```

- Test Lampiran 5 memakai `tests/fixtures/lampiran_l21.csv` (data naskah lama).
- `tests/fixtures/sample_real_10.csv` berisi 10 flow nyata dari 247 flow final (dibuat `scripts/make_sample_fixture.py`); nilai antara untuk Lampiran 2–5 ada di `reports/lampiran/lampiran_data.md`.
- Uji UI memakai `streamlit.testing.v1.AppTest` dengan config dan model sintetis terisolasi (variabel `QOS_CONFIG`).

## Struktur folder

```
.
├── PRD.md  CLAUDE.md  README.md  pyproject.toml  requirements.txt  requirements-dev.txt
├── config/            default_config.yaml  recommendations.yaml  (user_config.yaml, tidak di-commit)
├── data/
│   ├── raw/           CSV dataset (tidak di-commit)
│   └── sample_pcap/   PCAP untuk replay (tidak di-commit)
├── artifacts/<versi>/ model, scaler, PCA, profil, meta  +  artifacts/current.txt   (tidak di-commit)
├── reports/           eda/  experiments/  lampiran/  riwayat/  exports/  screenshots/  (tidak di-commit)
├── logs/app.log
├── src/qos/
│   ├── config.py
│   ├── data/          loader.py (muat CSV, filter VoIP)  eda.py
│   ├── features/      extract.py (delay/jitter/FAI)  preprocess.py (cleaning, imputasi, scaler, fitur)
│   ├── model/         train.py  evaluate.py  retrain.py  versions.py
│   ├── profiling/     profile.py (statistik, box plot, PLR)  labeling.py (aturan ITU)
│   ├── capture/       flow_table.py  sniffer.py (replay, scapy, pyshark)  demo.py (PCAP sintetis)
│   ├── monitor/       engine.py (inferensi, sesi, riwayat)
│   ├── recommend/     rules.py
│   ├── report/        export.py (CSV, PDF)
│   └── viz/           plots.py
├── app/               Home.py  ui.py  pages/1_… sampai 4_…
├── scripts/           train.py  replay_pcap.py  live_capture.py  eda.py  sensitivity.py  report_final.py
│                      compare_variants.py  make_sample_fixture.py  take_screenshots.py
└── tests/             test_*.py  conftest.py  fixtures/
```

## Keterbatasan dan catatan metodologi

- **Data bukan khusus VoIP.** Filter `VOIP` pada Darknet menghasilkan 3.566 flow; setelah pengecualian Tor dan filter sesi tersisa 247 (193 UDP, 54 TCP). Jumlah ini kecil untuk grid `min_cluster_size` hingga 25.
- **Semantik fitur bergeser.** Data latih adalah statistik seluruh flow (hingga 120 detik), sedangkan live memakai statistik kumulatif sejak awal flow atau per jendela 5 detik.
- **Kelas "QoS Cukup" kosong** pada model aktif (`20261022_120925`, hasil Terapkan Ulang Pelabelan dari `20261019_204042` dengan ambang loss 1/3%): Baik 49,8% (klaster 5, 123 flow), Buruk 36,4%, Anomali 13,8%. Satu-satunya klaster dengan loss level 2 (klaster 7, PLR 2,37%) juga melanggar jitter, sehingga tetap Buruk. Dengan ambang Y.1541 0,1% sebelumnya, klaster terbesar berlabel Buruk hanya karena PLR 0,1053%; FAI bersifat diskret sehingga ambang itu terlalu ketat (lihat `reports/experiments/labeling_scenarios.md`). P99,9 jitter pada klaster kecil ≈ nilai maksimum klaster.
- **Silhouette tinggi tidak otomatis berarti klaster bermakna QoS**; ekor ekstrem pada fitur dapat membentuk klaster tersendiri (lihat `scripts/compare_variants.py` untuk perbandingan varian fitur).
- **Demo PCAP bersifat sintetis** (`--make-demo`). Perbandingan `window` vs `cumulative` pada PCAP VoIP nyata belum dilakukan.
- **Backend pyshark** belum diuji dengan TShark sungguhan.
- **Kelas "Anomali"** hanya mendapat saran Anomali (investigasi kasus tunggal), sesuai Lampiran 5 naskah.

## Jangan di-commit

`data/raw/`, `*.pcap`, `*.pcapng`, `artifacts/`, `reports/`, `logs/`, `config/user_config.yaml` (sudah ada di `.gitignore`).
