Versi 1.0 · 12 Oktober 2026 · Pemilik: Hizkia Rahuel Anglie (535230073), Teknik Informatika UNTAR Sumber kebenaran: skripsi "Pemetaan Karakteristik VoIP Traffic Network untuk Identifikasi QoS menggunakan HDBSCAN"

1. Ringkasan

Web Dashboard lokal (Streamlit) untuk administrator jaringan/NOC yang:

Melatih model HDBSCAN secara offline dari statistik aliran CIC-IDS2017 (unsupervised, tanpa label).
Memberi label setiap klaster: QoS Baik / QoS Cukup / QoS Buruk berdasarkan ITU-T G.114 (delay) dan ITU-T Y.1541 Kelas 0/1 (jitter, packet loss). Noise (-1) = Anomali.
Memetakan trafik VoIP live per jendela waktu ke klaster yang sudah terbentuk via approximate_predict(), lalu mewarisi label klaster (lookup, tanpa hitung ulang).
Menampilkan label, visualisasi, dan rekomendasi mitigasi secara real-time, serta menyimpan riwayat dalam CSV.

Metodologi: CRISP-DM. Tidak ada database relasional (file-based).

2. Pengguna & Use Case

Aktor tunggal: Administrator Jaringan (akses penuh, tanpa login).

UC-01 Memilih network interface dan mode capture
UC-02 Start / Stop Monitoring
UC-03 Melihat ringkasan label dan sebaran klaster real-time
UC-04 Meninjau evaluasi dan profil klaster (skor, tabel, box plot)
UC-05 Mengatur parameter HDBSCAN dan ambang ITU
UC-06 Melatih ulang model (manual atau terjadwal, di luar sesi aktif)
UC-07 Melihat rekomendasi mitigasi
UC-08 Meninjau riwayat sesi dan mengekspor laporan CSV/PDF
3. Ruang Lingkup

In scope: pelatihan offline, monitoring real-time satu network segment, mode replay PCAP (untuk dev/demo), pelabelan ITU, rule-based recommendation, ekspor laporan. Out of scope: multi-segmen simultan, MOS, DPI/dekripsi payload, database relasional, autentikasi, deployment cloud.

4. Tech Stack
Python 3.10/3.11, Windows 11 atau Ubuntu 22.04
pandas, numpy, scikit-learn (StandardScaler, PCA, silhouette_score, calinski_harabasz_score)
hdbscan (WAJIB paket ini, BUKAN sklearn.cluster.HDBSCAN, karena butuh approximate_predict dan prediction_data=True)
joblib, pyyaml, psutil
pyshark (default, butuh tshark) dan scapy (fallback)
streamlit (>=1.37, pakai st.fragment(run_every=...) untuk auto-refresh), plotly
matplotlib + fpdf2 (PDF laporan), pytest
5. Arsitektur
Fase Pemantauan (real-time)
Fase Pelatihan (offline)
dimuat
dimuat
dimuat
CSV CIC-IDS2017
Filter VoIP (UDP, SIP, RTP)
Cleaning: inf ke NaN,duplikat, imputasi median
4 fitur + konversi ms + FAI
StandardScaler fit
HDBSCAN fit + grid search
Evaluasi: Silhouette, CHI,Noise Ratio
Profil klaster: statistik + boxplot
Pelabelan ITU-T G.114 danY.1541
artifacts: model, scaler, pca,cluster_profile.json
NIC / port mirroring / replayPCAP
Flow table 5-tuple + slidingwindow
4 fitur per jendela +imputasi median latih
scaler.transform
approximate_predict
Lookup label
Rule engine rekomendasi
Dashboard Streamlitauto-refresh
CSV riwayat
File CSV
Inference Engine
Capture Worker (thread)
Streamlit UI
File CSV
Inference Engine
Capture Worker (thread)
Streamlit UI
loop
[setiap window_sec]
Administrator Jaringan
Pilih interface + Start Monitoring
start(interface, mode)
agregasi paket per flow 5-tuple
vektor fitur per flow
scaler.transform, approximate_predict, lookup label
append baris jendela
push ke queue
st.fragment(run_every) baca queue lalu render
Stop Monitoring
stop()
tulis ringkasan sesi
Administrator Jaringan

Capture worker berjalan di background thread/process terpisah dari siklus rerun Streamlit; komunikasi via queue.Queue thread-safe.

6. Struktur Folder
voip-qos-hdbscan/
├── PRD.md  CLAUDE.md  README.md  requirements.txt  .gitignore
├── config/default_config.yaml, recommendations.yaml
├── data/raw/            # CSV CIC-IDS2017 (gitignored)
├── data/sample_pcap/    # pcap SIP/RTP untuk mode replay (gitignored)
├── artifacts/<versi>/   # model per versi + artifacts/current.txt
├── reports/riwayat/  reports/exports/   logs/
├── src/qos/
│   ├── config.py
│   ├── data/loader.py            # load + rename kolom + filter VoIP
│   ├── features/extract.py       # konversi satuan, FAI
│   ├── features/preprocess.py    # cleaning, imputasi, scaler
│   ├── model/train.py            # HDBSCAN + grid search
│   ├── model/evaluate.py         # silhouette, CHI, noise ratio
│   ├── profiling/profile.py      # statistik, box plot, P99.9, PLR
│   ├── profiling/labeling.py     # aturan ITU, string label
│   ├── recommend/rules.py
│   ├── capture/sniffer.py        # pyshark / scapy / replay
│   ├── capture/flow_table.py     # flow 5-tuple dua arah + window
│   ├── monitor/engine.py         # inferensi + logging
│   ├── viz/plots.py              # plotly
│   └── report/export.py          # CSV + PDF
├── app/Home.py
├── app/pages/1_Dashboard_Overview.py
├── app/pages/2_Analisis_Evaluasi_Klaster.py
├── app/pages/3_Pengaturan_Parameter.py
├── app/pages/4_Rekomendasi_Ekspor.py
├── scripts/train.py  scripts/replay_pcap.py
└── tests/ (fixtures/lampiran_l21.csv, test_*.py)
7. Spesifikasi Data
7.1 Pemetaan kolom CIC-IDS2017 (strip spasi di nama kolom!)
Kolom sumber	Nama internal	Satuan	Peran
Flow Duration	duration_us	µs	proksi delay
Flow IAT Std	iat_std_us	µs	jitter
Total Fwd Packets	fwd_packets_count	paket	komponen loss
Total Backward Packets	bwd_packets_count	paket	komponen loss
Protocol, Source Port, Destination Port	protocol, src_port, dst_port	-	filter VoIP

Turunan (satuan asli untuk profil & pelabelan):

delay_ms = duration_us / 1000
jitter_ms = iat_std_us / 1000
fai_pct = (fwd − bwd) / fwd × 100 (Pers. 2.1, per aliran, tanpa clipping); baris dengan fwd = 0 dibuang
7.2 Filter VoIP (configurable)

protocol == 17 DAN (src/dst port ∈ sip_ports ATAU dalam rtp_port_ranges). Log jumlah flow sebelum/sesudah filter. Jika < min_flows_warning, tampilkan peringatan; mode fallback udp_all hanya aktif bila diset manual.

7.3 Cleaning

±inf → NaN, drop duplikat, drop nilai negatif pada 4 fitur, imputasi median (median disimpan di artefak untuk dipakai saat live), sampling max_train_rows dengan random_state 42.

7.4 Fitur klasterisasi

Default 4 dimensi: [duration_us, iat_std_us, fwd_packets_count, bwd_packets_count], Z-Score (StandardScaler). HDBSCAN dijalankan pada ruang fitur ini, BUKAN pada hasil PCA. PCA 2D hanya untuk visualisasi.

8. Kebutuhan Fungsional
FR-TR Pelatihan
TR-1 hdbscan.HDBSCAN(min_cluster_size, min_samples, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
TR-2 Grid search otomatis: kombinasi grid.min_cluster_size × grid.min_samples; kandidat valid jika n_klaster ≥ 2 dan noise ratio ≤ 20%; pilih Silhouette tertinggi, tie-break CHI. Simpan grid_results.csv.
TR-3 Evaluasi (sekali, akhir pelatihan): Silhouette & CHI dihitung tanpa titik noise; Silhouette pakai sample_size=10000. Interpretasi Silhouette: >0,70 Strong Structure; 0,51–0,70 Reasonable Structure; 0,26–0,50 Weak Structure; ≤0,25 No Substantial Structure. Noise Ratio (Pers. 2.6): 1–10% ideal, >20% terlalu ketat.
TR-4 PCA 2D fit pada data latih ter-skala → pca.joblib; simpan sampel ≤5000 titik latih (pc1, pc2, cluster) untuk latar scatter.
TR-5 Versioning: setiap pelatihan membuat artifacts/<YYYYMMDD_HHMMSS>/; current.txt menunjuk versi aktif. Pelatihan ulang ditolak selama monitoring aktif.
TR-6 CLI: python scripts/train.py --config config/default_config.yaml [--data path] [--no-grid] (bisa dijadwalkan via Task Scheduler/cron). Sumber data latih ulang: CSV CIC-IDS2017 ATAU log jendela hasil monitoring.
FR-PRF Profil Klaster (semua klaster termasuk -1, satuan asli)
Untuk delay_ms, jitter_ms, fai_pct: mean, median, std (ddof=1), min, max (Pers. 2.7–2.8)
Box plot: Q1, median, Q3 via np.percentile(method="linear") (= Pers. 2.9), IQR, pagar Q1−1,5·IQR dan Q3+1,5·IQR, whisker = data min/max di dalam pagar, daftar pencilan (simpan maks 500)
Statistik pelabelan: mean delay, P99,9 jitter (p = 0,999), PLR klaster (Pers. 2.12): Σ max(fwd−bwd, 0) / Σ fwd × 100
FR-LBL Pelabelan (Pers. 2.13–2.14, ambang dari config)
L_delay: 1 jika ≤150; 2 jika 150<d≤400; 3 jika >400
L_jitter: 1 jika P99,9 ≤50; 3 jika >50
L_loss: 1 jika PLR ≤0,1%; 3 jika >0,1%
T = max(L_delay, L_jitter, L_loss)
Format: T=1 → "QoS Baik"; selain itu "QoS {Cukup|Buruk} - {metrik penyebab}" dengan metrik penyebab = metrik yang levelnya = T; keterangan "Meningkat" untuk level 2, "Melebihi Batas" untuk level 3. Nama metrik: Delay, Jitter, Packet Loss; gabung dengan ", " dan " dan " sebelum item terakhir. Contoh: "QoS Cukup - Delay Meningkat", "QoS Buruk - Delay dan Packet Loss Melebihi Batas"
Noise (-1): hitung level dengan cara sama; metrik berlevel 3 → "Anomali - {metrik} Ekstrem"; jika tidak ada → "Anomali"
Kategori Tindakan: Baik "-", Cukup "Pemantauan", Buruk "Perbaikan Kapasitas/Konfigurasi", Anomali "Investigasi Kasus Tunggal"
Mengubah ambang ITU cukup memicu "Terapkan Ulang Pelabelan" (pakai statistik tersimpan, tanpa refit)
FR-MON Akuisisi & Inferensi Real-Time
MON-1 Backend: pyshark LiveCapture (default), scapy AsyncSniffer (fallback), replay (baca .pcap/.pcapng dengan timing asli × replay_speed)
MON-2 BPF filter dari config; daftar interface via tshark -D / psutil
MON-3 Flow table key 5-tuple dua arah (kanonik); arah forward = pengirim paket pertama (seperti CICFlowMeter)
MON-4 Per window_sec (default 5 detik): hitung duration_us, iat_std_us, fwd, bwd per flow. feature_mode: window (paket dalam jendela, default) atau cumulative (sejak awal flow). Flow dengan paket < min_packets_per_window dilewati.
MON-5 Nilai kosong → median dari artefak latih; scaler TIDAK di-fit ulang
MON-6 hdbscan.approximate_predict(model, X) → cluster_id + membership_prob; lookup label dari cluster_profile.json; -1 → label Anomali dari profil noise
MON-7 Tiap baris jendela di-append ke CSV secara real-time
MON-8 Error jelas di UI jika: tshark tidak ditemukan, izin capture ditolak (butuh Administrator/root), atau tidak ada model aktif
FR-REC Rekomendasi (rule-based, config/recommendations.yaml)

Hanya untuk QoS Buruk dan Anomali.

Pemicu	Kategori	Saran
Delay Melebihi Batas	Perbaikan Kapasitas/Konfigurasi	Perlebar bandwidth jalur VoIP, prioritas antrean DiffServ (EF), evaluasi beban router dan jalur routing
Jitter Melebihi Batas	Perbaikan Kapasitas/Konfigurasi	Priority queuing untuk RTP, periksa kongesti/buffer antrean, sesuaikan jitter buffer endpoint
Packet Loss Melebihi Batas	Perbaikan Kapasitas/Konfigurasi	Periksa utilisasi antrean dan buffer router, error interface, kualitas media transmisi
Anomali	Investigasi Kasus Tunggal	Periksa potensi flood/burst, gangguan fisik media transmisi, kegagalan gateway; cocokkan timestamp dengan log perangkat

Klaster dengan beberapa metrik penyebab menggabungkan saran semua metrik.

FR-VIZ Visualisasi
Warna label: Baik 
#2563EB (biru), Cukup 
#EAB308 (kuning), Buruk 
#F97316 (oranye), Anomali 
#DC2626 (merah)
PCA scatter 2D (hover, zoom, pan): latar titik latih pudar + titik jendela live
Bar chart statistik pelabelan per klaster (mean delay, P99,9 jitter, PLR) dengan garis ambang ITU
Box plot per metrik per klaster (termasuk noise) dari statistik tersimpan memakai go.Box(q1, median, q3, lowerfence, upperfence); garis ambang 150 & 400 ms (delay), 50 ms (jitter), 0,1% (loss); penanda mean delay, P99,9 jitter, PLR. Toggle skala log. Profil TIDAK dihitung ulang per jendela.
9. Halaman UI (sesuai Lampiran 7 skripsi)

P1 Dashboard Overview

Sidebar: navigasi, pemilih interface, mode (Live/Replay PCAP + uploader), Start/Stop Monitoring, versi model aktif
5 kartu: Total Aliran Trafik, QoS Baik, QoS Cukup, QoS Buruk, Anomali (hitung flow unik berdasarkan label terakhir)
PCA scatter live, bar chart vs ambang ITU, tabel jendela terbaru

P2 Analisis & Evaluasi Klaster

Kartu Silhouette (+ interpretasi), CHI, Noise Ratio
Tabel profil: ID Klaster, Jumlah Anggota, mean/median/std delay-jitter-FAI, statistik pelabelan, level per metrik, Label, Kategori Tindakan
Panel box plot 3 metrik (hover: Q1, median, Q3, whisker, jumlah pencilan)
Tabel sampel data: baris individual + cluster_id + membership_prob

P3 Pengaturan Parameter & Ambang Batas

Slider min_cluster_size 2–50 (default 5), min_samples 1–20 (default 2), toggle grid search otomatis
Input ambang: delay baik 150, delay maks 400, jitter maks 50, loss maks 0,1
Tombol Save Configuration, Reset to Default, Terapkan Ulang Pelabelan, Latih Ulang Model (disabled saat monitoring; progress bar; setelah selesai tampilkan skor + tombol "Aktifkan model ini")

P4 Rekomendasi Mitigasi & Ekspor Laporan

Alert box per klaster bermasalah dan per sesi
Tabel riwayat sesi (dari CSV)
Unduh laporan per sesi: CSV (jendela) dan PDF ringkasan
10. Artefak & Output
10.1 artifacts/<versi>/

hdbscan_model.joblib, scaler.joblib, pca.joblib, cluster_profile.json, train_meta.json (parameter, ambang, median imputasi, jumlah baris, filter VoIP, timestamp), grid_results.csv, training_sample.csv

10.2 cluster_profile.json (contoh dari Lampiran 5)
json
{
  "model_version": "20261012_230000",
  "thresholds": {"delay_good_ms": 150, "delay_max_ms": 400, "jitter_max_ms": 50, "loss_max_pct": 0.1},
  "evaluation": {"silhouette": 0.0, "silhouette_label": "", "chi": 0.0, "noise_ratio": 0.0, "n_clusters": 0},
  "clusters": [
    {
      "id_klaster": 0,
      "jumlah_anggota": 5,
      "statistik": {"delay_ms": {"mean": 109.23, "median": 105.60, "std": 18.71, "min": 89.40, "max": 135.20}},
      "boxplot": {"delay_ms": {"q1": 95.40, "median": 105.60, "q3": 120.54, "iqr": 25.14,
                   "lower_fence": 57.69, "upper_fence": 158.25, "whisker_low": 89.40,
                   "whisker_high": 135.20, "outliers": []}},
      "statistik_pelabelan": {"mean_delay_ms": 109.23, "p999_jitter_ms": 0.92, "plr_pct": 0.99},
      "tingkat": {"delay": 1, "jitter": 1, "loss": 3, "klaster": 3},
      "label": "QoS Buruk - Packet Loss Melebihi Batas",
      "kategori_tindakan": "Perbaikan Kapasitas/Konfigurasi"
    }
  ],
  "label_map": {"0": "QoS Buruk - Packet Loss Melebihi Batas", "-1": "Anomali - Delay dan Packet Loss Ekstrem"}
}

(statistik & boxplot berisi delay_ms, jitter_ms, fai_pct untuk setiap klaster)

10.3 Riwayat
reports/riwayat/jendela_<id_sesi>.csv (append tiap jendela): waktu_jendela, id_sesi, flow_id, delay_ms, jitter_ms, fwd, bwd, fai_pct, cluster_id, membership_prob, label, kategori_tindakan, pc1, pc2
reports/riwayat/riwayat_pemantauan_<YYYYMMDD_HHMMSS>.csv (ringkasan sesi saat Stop): id_sesi_pemantauan (interface + waktu mulai), waktu_mulai, waktu_selesai, total_aliran, jumlah_klaster, skor_silhouette, skor_chi, noise_ratio, status_keseluruhan, lokasi_laporan
jumlah_klaster dan skor_* diambil dari model aktif (hasil pelatihan)
status_keseluruhan: "Kritis" jika proporsi (QoS Buruk + Anomali) > kritis_ratio; selain itu label mayoritas
11. config/default_config.yaml
yaml
data:
  raw_dir: data/raw
  max_train_rows: 200000
  random_state: 42
  voip_filter:
    protocol: 17
    sip_ports: [5060, 5061]
    rtp_port_ranges: [[10000, 20000], [16384, 32767]]
    min_flows_warning: 1000
    fallback_udp_all: false
features:
  cluster_columns: [duration_us, iat_std_us, fwd_packets_count, bwd_packets_count]
hdbscan:
  min_cluster_size: 5
  min_samples: 2
  cluster_selection_method: eom
  grid_search:
    enabled: true
    min_cluster_size: [5, 10, 20, 30, 40, 50]
    min_samples: [1, 2, 5, 10, 20]
    max_noise_ratio: 0.20
    silhouette_sample_size: 10000
thresholds:
  delay_good_ms: 150.0
  delay_max_ms: 400.0
  jitter_max_ms: 50.0
  loss_max_pct: 0.1
monitoring:
  capture_backend: pyshark   # pyshark | scapy | replay
  bpf_filter: "udp"
  window_sec: 5
  feature_mode: window       # window | cumulative
  min_packets_per_window: 10
  refresh_sec: 2
  replay_speed: 1.0
  kritis_ratio: 0.5
paths:
  artifacts_dir: artifacts
  reports_dir: reports
  logs_dir: logs

Konfigurasi aktif dikelola di st.session_state; Save Configuration menulis config/user_config.yaml (bukan database).

12. Non-Fungsional
Jalan lokal: streamlit run app/Home.py; tanpa internet saat runtime
Label tampil ≤ window_sec + refresh_sec setelah paket masuk
Reproducible (random_state 42), logging ke logs/app.log
Type hints, docstring singkat Bahasa Indonesia, tidak ada angka ambang hardcode
13. Acceptance Criteria & Test

Fixture tests/fixtures/lampiran_l21.csv (Tabel L2.1 + penugasan klaster Lampiran 4):

sample,flow_duration_us,total_fwd_packets,total_backward_packets,flow_iat_std_us,cluster
s1,120540,150,148,812.45,0
s2,450120,220,175,3450.80,1
s3,89400,110,109,420.15,0
s4,980350,310,190,12540.30,-1
s5,135200,180,178,915.60,0
s6,1120000,400,210,18200.75,-1
s7,95400,125,124,510.20,0
s8,520400,260,195,4120.50,1
s9,105600,140,139,740.10,0
s10,1450000,500,240,24100.90,-1

Nilai yang WAJIB cocok (2 desimal):

Uji	Klaster 0	Klaster 1	Noise
Mean delay (ms)	109.23	485.26	1183.45
Std delay (ms)	18.71	49.70	241.17
P99,9 jitter (ms)	0.92	4.12	24.09
PLR (%)	0.99	22.92	47.11
Q1 / Q3 delay (ms)	95.40 / 120.54	467.69 / 502.83	1050.18 / 1285.00
Label	QoS Buruk - Packet Loss Melebihi Batas	QoS Buruk - Delay dan Packet Loss Melebihi Batas	Anomali - Delay dan Packet Loss Ekstrem

Tambahan: FAI s1 = 1.33, s2 = 20.45, s10 = 52.00; pagar delay K0 = 57.69 / 158.25; whisker K0 = 89.40 / 135.20; noise ratio = 30%; semua kombinasi Tabel 2.3 menghasilkan string label yang benar.

Integrasi:

Pelatihan pada data sintetis → semua artefak terbentuk dan bisa di-load; approximate_predict jalan
Replay pcap SIP/RTP → CSV jendela terisi, dashboard terupdate, Stop menulis ringkasan

Target model (dilaporkan, bukan test gagal): Silhouette > 0,50 dan Noise Ratio ≤ 20%.

14. Milestone (kerjakan berurutan, berhenti untuk review di tiap ⛔)
M0 Scaffold folder, requirements, config loader, pytest jalan
M1 Loader + filter VoIP + laporan EDA (jumlah flow, distribusi 4 fitur) ⛔ laporkan jumlah flow VoIP
M2 Features, cleaning, FAI, profil, pelabelan + test Lampiran 5 lulus ⛔
M3 Training + grid search + evaluasi + artefak + CLI ⛔ laporkan skor
M4 Inference engine + mode replay PCAP (tanpa UI)
M5 UI 4 halaman memakai mode replay ⛔
M6 Live capture pyshark/scapy + penanganan izin
M7 Rekomendasi, riwayat, ekspor CSV/PDF
M8 Latih ulang dari UI + versioning + Terapkan Ulang Pelabelan
M9 README, screenshot tiap halaman (untuk Bab/Lampiran)
15. Risiko Teknis
CIC-IDS2017 tidak dirancang khusus untuk VoIP; jumlah flow lolos filter bisa kecil → wajib dicek di M1
Semantik fitur: Flow Duration CIC = durasi seluruh flow, sedangkan live dihitung per jendela → distribusi bisa bergeser; bandingkan window vs cumulative saat M4
Ambang PLR 0,1% sangat ketat terhadap estimasi FAI (pada contoh Lampiran 5 bahkan klaster terbaik sudah 0,99%); kemungkinan tidak ada klaster "QoS Baik" → laporkan distribusi label di M3 dan jangan mengubah ambang tanpa persetujuan
min_cluster_size kecil pada data besar → klaster sangat banyak; andalkan grid search
Silhouette O(n²) → wajib sample_size
Live capture butuh hak Administrator/root dan tshark/Npcap