# Pemetaan Karakteristik VoIP Traffic Network dengan Klasterisasi HDBSCAN

Dashboard web lokal (Streamlit) untuk administrator jaringan yang:

1. melatih model **HDBSCAN** secara offline dari statistik aliran (unsupervised), dengan **K-Means** sebagai pembanding;
2. membentuk klaster bernomor 1..k (urut ukuran menurun; noise HDBSCAN = *Noise*) dan memprofilkan tiap klaster dalam satuan asli; posisi klaster terhadap ambang ITU (ITU-T G.114 delay, Y.1541 jitter, G.1010 dan panduan Cisco VoIP untuk loss) hanya ditampilkan sebagai **deskripsi**, bukan label kualitas;
3. menugaskan trafik per jendela waktu ke **klaster terdekat berdasarkan jarak Euclidean ke centroid**; titik yang jaraknya ke setiap centroid melebihi batas klaster ditandai **Di luar klaster (terdekat: Klaster N)**;
4. menampilkan klaster terdekat, jarak, visualisasi, dan saran berdasarkan karakteristik klaster, serta menyimpan riwayat sesi ke CSV dan PDF.

Metodologi CRISP-DM. Tanpa database relasional (semua berbasis berkas). Spesifikasi lengkap ada di [`PRD.md`](PRD.md) (sumber kebenaran proyek ini; revisi klasterisasi murni bertanda T2); aturan kerja ada di [`CLAUDE.md`](CLAUDE.md).

> **Catatan penting tentang data dan hasil.** Data latih adalah CIC-Darknet2020 (`Darknet.CSV`, flow berlabel `VOIP`), bukan trafik yang dirancang khusus untuk VoIP. Setelah filter (tanpa Tor, fwd dan bwd ≥ 5 paket) tersisa **247 flow**. Keluaran aplikasi adalah klaster terdekat, bukan penilaian QoS. Lihat bagian [Keterbatasan](#keterbatasan-dan-catatan-metodologi) sebelum mengutip angka.

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

Pipeline: baca CSV → filter VoIP → buang `exclude_traffic_types` (default `Tor`) dan flow dengan fwd/bwd < `min_session` (default 5/5) → cleaning (±inf→NaN, duplikat, negatif, imputasi median) → hitung `fai_pct` → `log1p` pada `iat_mean_us` dan `iat_std_us` → `StandardScaler` → grid search HDBSCAN dengan ARI bootstrap tiap kandidat → seleksi final → centroid, batas jarak → K-Means pembanding → evaluasi, profil, dan deskripsi ITU → simpan artefak.

- Fitur klaster (Varian C): `[iat_mean_us, iat_std_us, fai_pct]`, `log1p` pada dua kolom pertama, **tanpa** log pada `fai_pct`. Proksi delay = *Flow IAT Mean*; jitter = *Flow IAT Std*; loss = FAI.
- **Aturan seleksi HDBSCAN** (`hdbscan.selection`): kandidat valid = noise ≤ 20%, klaster ≥ 2, ARI bootstrap (50×, subsampel 80%) mean ≥ 0,90; dipilih **DBCV tertinggi**, seri → Silhouette, lalu `min_cluster_size` dan `min_samples` terkecil. Tanpa kandidat valid dipakai parameter tetap config (dicatat `grid_fallback`).
- **Pembanding K-Means**: k = jumlah klaster HDBSCAN dan k optimal (rata-rata peringkat Silhouette dan DBI, k 2–10).
- **Penugasan**: centroid = rata-rata anggota di ruang ter-skala. Batas "di luar klaster" (`assignment.limit_rule`, default `max_cluster_global`) = per klaster max(p95 jarak klaster, p95 global); titik di luar klaster bila jaraknya ke setiap centroid melebihi batas klaster itu.
- Hasil reprodusibel (`random_state: 42`). Versi baru **tidak** menjadi aktif kecuali dengan `--activate` (atau bila belum ada versi aktif sama sekali).

Model aktif `20261027_005150` (247 flow): 5 klaster + noise, `min_cluster_size` 3, `min_samples` 8; HDBSCAN (tanpa noise) Silhouette 0,8767, DBI 0,2576, CHI 3.564,63, DBCV 0,7404, noise 13,36%, ARI 0,964 ± 0,024. Dengan noise di-assign ke centroid: Silhouette 0,7570, DBCV 0,3545. K-Means k=5: Silhouette 0,7729, DBCV 0,4258; K-Means k=3 (optimal): Silhouette 0,7787, DBCV −0,1737. Dasar keputusan: `reports/experiments/cluster_analysis.md` bagian F dan G (`scripts/cluster_analysis_final.py`, `scripts/cluster_analysis_selection.py`).

## Replay PCAP

Replay membaca `.pcap`/`.pcapng` dengan timing asli dibagi `replay_speed`, tanpa Npcap dan tanpa hak khusus.

```powershell
python scripts/replay_pcap.py --make-demo                       # buat data/sample_pcap/demo_voip.pcap (SINTETIS, bukan data nyata)
python scripts/replay_pcap.py --pcap data/sample_pcap/<berkas>.pcap --speed 0   # 0 = secepat mungkin
python scripts/replay_pcap.py --pcap <berkas> --speed 0 --compare               # bandingkan feature_mode window vs cumulative
```

Keluarannya: CSV jendela di `reports/riwayat/jendela_<id_sesi>.csv` dan ringkasan sesi `riwayat_pemantauan_<waktu>.csv` (ditulis saat Stop), dan ringkasan per panggilan `panggilan_<id_sesi>.csv`.

`monitoring.feature_mode`: `cumulative` (default; statistik sejak awal flow) atau `window` (hanya paket dalam jendela). Mode `window` memberi FAI ≈ 0,13% pada RTP tanpa loss karena efek batas jendela, sehingga `cumulative` dijadikan default.

## Live capture

```powershell
python scripts/live_capture.py --check                               # periksa prasyarat (tshark, Npcap, hak akses)
python scripts/live_capture.py --list-interfaces
python scripts/live_capture.py --backend scapy --interface "Wi-Fi" --seconds 30
```

- **Backend default `scapy`** (butuh Npcap). `pyshark` adalah opsi (butuh TShark). Tidak ada fallback otomatis antar-backend; jika prasyarat tidak ada, aplikasi berhenti dengan pesan yang menyebut langkah perbaikannya.
- **Filter BPF default**: `udp and not port 443 and not port 53 and not port 123 and not port 5353 and not port 1900`. Port tersebut (QUIC, DNS, NTP, mDNS, SSDP) di luar distribusi latih dan hanya menghasilkan titik di luar klaster yang menyesatkan.
- Jendela ditutup berdasarkan timestamp paket, ditambah detak `live_poll_sec` saat tidak ada paket.
- Flow/jendela dengan fwd atau bwd < `monitoring.min_session_live` (default 5/5) tidak diprediksi: statusnya **"Data Belum Cukup"**, dan tetap dicatat di CSV (tidak ditugaskan ke klaster).
- Hanya statistik agregat dan alamat IP flow yang disimpan di `reports/riwayat/`; payload tidak pernah dibaca. Folder `reports/` tidak di-commit.

## Dashboard

```powershell
streamlit run app/Home.py
```

Peluncur lokal (Windows, memakai `.venv`):

- `run_app.bat` — server Streamlit headless dan browser terbuka otomatis (port 8501; ubah dengan `set PORT=...` sebelum menjalankan).
- `run_desktop.bat` — jendela desktop "VoIP QoS Monitor" (pywebview, butuh WebView2 yang sudah ada di Windows 11); server berhenti saat jendela ditutup. Live capture butuh hak Administrator.

Bagian atas Dashboard menampilkan banner **Status Saat Ini: paling dekat ke Klaster N (jarak X)** (kelompok terbanyak di antara panggilan aktif; berwarna klaster, abu-abu untuk *Di luar klaster*), status capture (paket, flow aktif, paket terakhir; peringatan bila belum ada paket), kartu jumlah flow per klaster dan *Di luar klaster*, serta panel **Panggilan Aktif** (kartu per panggilan: klaster terdekat, jarak dan batas, bar jarak ke semua centroid, delay/jitter/FAI, sparkline klaster per jendela). Interface bawaan di sidebar adalah yang aktif dan terhubung; pilih manual bila perlu (mis. *Wi-Fi*, bukan adapter Bluetooth yang tidak aktif — sesi dengan interface mati menghasilkan 0 paket).

Sesi yang berakhir tanpa Stop tetap diringkas dengan status "Dihentikan tidak normal" (saat aplikasi ditutup atau capture error; bila proses dimatikan paksa, ringkasan ditulis saat aplikasi dibuka kembali setelah `monitoring.orphan_after_sec`). Halaman Saran & Ekspor memilih sesi terbaru otomatis.

Flow sinyal SIP (port di `monitoring.signaling_ports`, default 5060/5061) berstatus "Sinyal SIP": dicatat di CSV jendela tetapi tidak ditugaskan, tidak masuk kartu, ringkasan sesi, tabel panggilan, saran, maupun latih ulang. Tabel *Panggilan Selesai* menampilkan kolom ringkas (kolom lengkap di expander *Detail*); beberapa panggilan yang selesai dalam satu siklus refresh digabung menjadi satu toast.

| Halaman | Isi |
|---|---|
| Home | Ringkasan model aktif (Silhouette, DBCV, klaster, noise) dan alur kerja |
| Dashboard Overview | Banner status, kartu per klaster + Di luar klaster, panggilan aktif, PCA scatter dengan centroid, deskripsi klaster, saran sesi berjalan, panggilan selesai, jendela terbaru; auto-refresh |
| Analisis & Evaluasi Klaster | Metrik HDBSCAN (Silhouette, DBI, CHI, DBCV, noise, ARI), perbandingan HDBSCAN (dua versi) vs K-Means, kurva Elbow/Silhouette/DBI, heatmap grid dengan ARI, profil + deskripsi ITU, box plot dengan garis ambang ITU (toggle log), centroid dan batas jarak, sampel data latih |
| Pengaturan Parameter | Slider HDBSCAN, aturan seleksi, aturan batas jarak, ambang ITU (hanya deskripsi), Save/Reset, **Perbarui Deskripsi Klaster** (tanpa refit), **Latih Ulang Model** (setelah centang konfirmasi), daftar dan aktivasi versi model |
| Rekomendasi & Ekspor | Saran berdasarkan karakteristik klaster (bila `recommendations.enabled`), riwayat sesi, unduh CSV dan PDF |

Sidebar (semua halaman): versi model aktif, mode Replay PCAP / Live, sumber, Start/Stop. Latih ulang dan pergantian model aktif dinonaktifkan selama monitoring berjalan.

## Konfigurasi

| Berkas | Fungsi |
|---|---|
| `config/default_config.yaml` | Seluruh parameter dan ambang (tidak ada angka ambang di kode) |
| `config/user_config.yaml` | Dibuat oleh *Save Configuration*; menimpa default; tidak di-commit; dihapus oleh *Reset to Default* |
| `config/recommendations.yaml` | Aturan saran per metrik yang melewati ambang (Delay, Jitter, Packet Loss) dan untuk Di luar klaster |
| `config/analysis_config.yaml` | Analisis klasterisasi murni Tahap 1 (`scripts/cluster_analysis*.py`); tidak memengaruhi aplikasi |

Ambang ITU default: delay 150/400 ms, jitter 50 ms, loss 1/3% (sumber ITU-T G.114, Y.1541, G.1010, dan panduan Cisco VoIP). Ambang **tidak memengaruhi klasterisasi atau penugasan**; hanya untuk deskripsi klaster, garis referensi grafik, dan saran. Saran dapat dimatikan dengan `recommendations.enabled: false`.

Satuan internal: delay dan jitter dalam ms, loss dalam %; data CIC dalam µs dibagi 1000.

## Artefak dan versi model

Setiap pelatihan membuat `artifacts/<YYYYMMDD_HHMMSS>/` dan `artifacts/current.txt` menunjuk versi aktif. Isi satu versi:

`hdbscan_model.joblib`, `kmeans_n.joblib` (k = klaster HDBSCAN), `kmeans_opt.joblib` (k optimal), `scaler.joblib`, `pca.joblib`, `assignment.json` (centroid ter-skala, batas per klaster, batas global, aturan), `cluster_profile.json` (evaluasi, perbandingan, kurva K-Means, profil dan deskripsi tiap klaster), `train_meta.json`, `grid_results.csv` (dengan ARI), `training_sample.csv`.

- **Latih ulang** (halaman Pengaturan atau CLI) tidak pernah menimpa versi lama dan tidak mengubah `current.txt`; versi baru aktif hanya setelah *Aktifkan model ini* (atau `--activate`).
- **Versi lama** (pelabelan QoS, tanpa `schema: klaster_murni`) tetap ada tetapi **ditolak saat diaktifkan dan saat dimuat** dengan pesan jelas; daftar versi menandainya `kompatibel = False`.
- **Sumber latih ulang**: CSV (Darknet/CIC) atau log jendela monitoring. Untuk log: baris *Data Belum Cukup* dan *Sinyal SIP* dikecualikan; baris di luar klaster, baris `diimputasi = true`, dan baris dengan `feature_mode` selain yang diatur (default `cumulative`) dikecualikan sesuai `retrain.log_source`; jumlah minimum baris (`min_rows`, default 200) dicek sebelum pelatihan.
- Deskripsi klaster memuat median total paket, median durasi flow (detik), dan % UDP/TCP: `Klaster 2 · 32 paket · 90 dtk · Delay >400 · Jitter >50 · Loss ≤1`.
- **Perbarui Deskripsi Klaster** menghitung ulang deskripsi posisi ITU dari statistik tersimpan (profil lama melengkapi karakter dari `training_sample.csv`) dengan ambang baru, **tanpa melatih ulang**, dan menghasilkan versi baru (model, centroid, batas disalin tanpa berubah).

## Riwayat dan ekspor

- `reports/riwayat/jendela_<id_sesi>.csv` — ditambah tiap jendela: `waktu_jendela, id_sesi, flow_id, delay_ms, jitter_ms, fwd, bwd, fai_pct, status, klaster_terdekat, jarak_terdekat, di_luar_klaster, jarak_ke_klaster, klaster_kmeans5, pc1, pc2, feature_mode, diimputasi`. `status` = Terklaster | Di luar klaster | Data Belum Cukup | Sinyal SIP; `jarak_ke_klaster` berformat `1:0.123;2:1.402;…` (titik desimal); `klaster_kmeans5` = klaster terdekat menurut K-Means pembanding (k = jumlah klaster HDBSCAN; nama kolom tetap). `id_sesi` berakhiran 6 karakter acak.
- `reports/riwayat/panggilan_<id_sesi>.csv` — satu baris per panggilan (flow) yang selesai: idle melewati `monitoring.call_end_idle_sec` (default 10 detik), atau saat Stop / PCAP habis. Kolom: `flow_id, waktu_mulai, waktu_selesai, durasi_s, total_fwd, total_bwd, delay_ms, jitter_ms, fai_pct, klaster_akhir, jarak_akhir, di_luar_klaster_akhir, status_akhir, jumlah_jendela, persen_per_klaster, persen_di_luar_klaster, persen_belum_cukup, klaster_terbanyak, pernah_di_luar_klaster, catatan`. `*_akhir` = jendela kumulatif terakhir; panggilan yang berakhir *Data Belum Cukup* tetap dicatat. **Pada `feature_mode: window`** nilai hanya dari jendela terakhir (kolom `catatan` memberi peringatan); ringkasan panggilan yang sah hanya pada mode `cumulative`.
- `reports/riwayat/riwayat_pemantauan_<waktu>.csv` — ringkasan sesi saat Stop: `status_sesi` (Selesai | Dihentikan tidak normal | Tidak ada data), `jumlah_panggilan`, `panggilan_per_klaster` (`1:3;2:1`), `panggilan_di_luar_klaster`, `persen_di_luar_klaster`, `klaster_mayoritas`, `panggilan_belum_cukup`, serta skor model (`skor_silhouette, skor_dbcv, skor_chi, noise_ratio`). Tidak ada status "Kritis".
- `reports/exports/laporan_<id_sesi>.csv|pdf` — dibuat dari halaman *Rekomendasi & Ekspor* (PDF: info sesi dan model, distribusi klaster, saran berdasarkan karakteristik klaster, tabel panggilan selesai beserta saran per panggilan, tabel aliran, profil klaster dengan deskripsi ITU).

## Tangkapan layar

Untuk Lampiran 7. Skrip menjalankan server lokal, memutar replay PCAP demo di Dashboard, lalu menyimpan PNG ke `reports/screenshots/`:

```powershell
pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/take_screenshots.py [--pcap data/sample_pcap/<berkas>.pcap] [--speed 3]
```

Berkas: `01_home`, `02_dashboard_replay`, `02b_toast_panggilan_selesai`, `02c_dashboard_panggilan_selesai`, `03_analisis_klaster`, `03c_perbandingan_hdbscan_kmeans`, `03d_grid_hdbscan`, `03e_centroid_batas_jarak`, `03b_analisis_klaster_skala_log`, `04_pengaturan`, `04b_pengaturan_perbarui_deskripsi_latih_ulang`, `05_rekomendasi_ekspor`, `05b_rekomendasi_riwayat_laporan_sesi`, `05c_rekomendasi_panggilan_per_sesi`. Tabel riwayat pada tangkapan memuat semua sesi yang ada di `reports/riwayat/`; kosongkan atau pindahkan folder itu lebih dulu bila ingin riwayat yang bersih.

## Pengujian

```powershell
pytest                       # seluruh uji (±290 uji, ±1 menit)
pytest tests/test_profile.py # uji Lampiran 5 (statistik, box plot, FAI, noise ratio) dan deskripsi ITU
```

- Test Lampiran 5 memakai `tests/fixtures/lampiran_l21.csv` (data naskah lama); uji label QoS dihapus karena aplikasi tidak lagi melabeli. Penugasan centroid, aturan batas (a/b/c), di luar klaster, nomor klaster 1..k, kolom CSV, dan perbandingan K-Means diuji di `tests/test_assignment.py`, `test_train.py`, `test_engine.py`.
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
│   ├── model/         train.py  evaluate.py  assignment.py (centroid, batas, penugasan)  retrain.py  versions.py
│   ├── profiling/     profile.py (statistik, box plot, centroid)  describe.py (deskripsi posisi ITU)
│   ├── analysis/      analisis klasterisasi Tahap 1 (laporan cluster_analysis.md)
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
- **Tidak ada penilaian QoS.** Klaster hanya mengelompokkan karakteristik (IAT mean, IAT std, FAI). Deskripsi ITU memakai median klaster dan FAI sebagai proksi loss; klaster 2 dan 3 (delay median 2,9 dan 9,1 detik) menunjukkan aliran lambat/jarang pada data latih, bukan panggilan VoIP berkualitas buruk.
- **Stabilitas.** Banyak kandidat grid berklaster 5 yang metriknya sama persis; pemilihan DBCV + ARI ≥ 0,90 memilih `min_cluster_size` 3 dan `min_samples` 8 (ARI 0,964). Konfigurasi dengan 8 klaster (DBCV 0,758) tidak stabil (ARI 0,843 ± 0,232).
- **Batas jarak** klaster padat sangat kecil (mis. p95 0,023 pada klaster 3 bila dipakai per klaster); aturan `max_cluster_global` menaikkannya ke batas global 0,443. Pada 247 flow aturan ini menandai 90,9% noise latih (30/33) dan 1,4% non-noise (3/214) sebagai di luar klaster.
- **Silhouette tinggi tidak otomatis berarti klaster bermakna QoS**; ekor ekstrem pada fitur dapat membentuk klaster tersendiri (lihat `scripts/compare_variants.py`).
- **Demo PCAP bersifat sintetis** (`--make-demo`). Perbandingan `window` vs `cumulative` pada PCAP VoIP nyata belum dilakukan.
- **Backend pyshark** belum diuji dengan TShark sungguhan.
- **Titik "Di luar klaster"** hanya mendapat saran investigasi kasus tunggal; saran per klaster hanya untuk metrik yang melewati ambang ITU.

## Jangan di-commit

`data/raw/`, `*.pcap`, `*.pcapng`, `artifacts/`, `reports/`, `logs/`, `config/user_config.yaml` (sudah ada di `.gitignore`).
