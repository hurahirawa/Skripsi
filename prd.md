Versi 1.0 · 12 Oktober 2026 · Pemilik: Hizkia Rahuel Anglie (535230073), Teknik Informatika UNTAR Catatan perubahan 19 Okt 2026: sumber data latih = CIC-Darknet2020 (Darknet.CSV, filter kategori VOIP), bukan CIC-IDS2017; proksi delay = Flow IAT Mean; filter sesi minimum fwd/bwd ≥ 5 dan exclude Tor; FAI di-clip ke ≥ 0; min_flows_warning 200; grid min_cluster_size [5, 10, 15, 20, 25]. Ambang ITU tidak berubah. Sumber kebenaran: skripsi "Pemetaan Karakteristik VoIP Traffic Network untuk Identifikasi QoS menggunakan HDBSCAN"

1. Ringkasan

Web Dashboard lokal (Streamlit) untuk administrator jaringan/NOC yang:

Melatih model HDBSCAN secara offline dari statistik aliran CIC-IDS2017 (unsupervised, tanpa label).
Memberi label setiap klaster: QoS Baik / QoS Cukup / QoS Buruk berdasarkan ITU-T G.114 (delay), ITU-T Y.1541 Kelas 0/1 (jitter), serta ITU-T G.1010 (conversational voice, PLR < 3%) dan panduan Cisco VoIP (PLR ≤ 1%) untuk packet loss. Noise (-1) = Anomali.
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
scapy (default sejak 19 Okt 2026; butuh Npcap/libpcap) dan pyshark (opsi; butuh tshark)
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
Flow Duration	duration_us	µs	Hanya EDA (bukan fitur)
Flow IAT Mean	iat_mean_us	µs	proksi delay
Flow IAT Std	iat_std_us	µs	jitter
Total Fwd Packets	fwd_packets_count	paket	komponen loss
Total Backward Packets	bwd_packets_count	paket	komponen loss
Protocol, Source Port, Destination Port	protocol, src_port, dst_port	-	filter VoIP

Turunan (satuan asli untuk profil & pelabelan):

delay_ms = iat_mean_us / 1000 (diubah 19 Okt 2026, sebelumnya duration_us / 1000)
jitter_ms = iat_std_us / 1000
fai_pct = (fwd − bwd) / fwd × 100 (Pers. 2.1, per aliran); baris dengan fwd = 0 dibuang. Bila features.fai_clip_negative = true (default, diubah 19 Okt 2026), FAI < 0 dijadikan 0
7.2 Filter VoIP (configurable)

protocol == 17 DAN (src/dst port ∈ sip_ports ATAU dalam rtp_port_ranges). Log jumlah flow sebelum/sesudah filter. Jika < min_flows_warning, tampilkan peringatan; mode fallback udp_all hanya aktif bila diset manual.

7.3 Cleaning

Filter sesi minimum (diubah 19 Okt 2026): buang traffic type di data.exclude_traffic_types (default [Tor]) dan flow dengan fwd < data.min_session.fwd atau bwd < data.min_session.bwd (default 5/5; null = nonaktif). Lalu: ±inf → NaN, drop duplikat, drop nilai negatif pada 4 fitur, imputasi median (median disimpan di artefak untuk dipakai saat live), sampling max_train_rows dengan random_state 42.

7.4 Fitur klasterisasi

Default 3 dimensi: [iat_mean_us, iat_std_us, fai_pct] (Varian C, diubah 19 Okt 2026). Transformasi log1p dipakai pada iat_mean_us dan iat_std_us; fai_pct TIDAK di-log (FAI sudah berskala persentase dan nol sering muncul). Setelah itu StandardScaler. duration_us tidak dipakai sebagai fitur; fwd/bwd mentah tidak dipakai karena mewakili panjang panggilan, bukan QoS. cluster_columns dan log_transform disimpan di train_meta.json agar inferensi live memakai transformasi yang sama, Z-Score (StandardScaler). HDBSCAN dijalankan pada ruang fitur ini, BUKAN pada hasil PCA. PCA 2D hanya untuk visualisasi.

8. Kebutuhan Fungsional
FR-TR Pelatihan
TR-1 hdbscan.HDBSCAN(min_cluster_size, min_samples, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
TR-2 Grid search otomatis: kombinasi grid.min_cluster_size × grid.min_samples; kandidat valid jika n_klaster ≥ 2 dan noise ratio ≤ 20%; pilih Silhouette tertinggi, tie-break CHI. Jika masih seri (kandidat identik, selisih < 1e-9), pilih min_cluster_size terkecil, lalu min_samples terkecil (pada model 20261019_175647 tidak ada seri penuh); aturan dan daftar kandidat seri penuh dicatat di train_meta.json (parameters.tie_break_rule, kandidat_seri_penuh). Simpan grid_results.csv.
TR-3 Evaluasi (sekali, akhir pelatihan): Silhouette & CHI dihitung tanpa titik noise; Silhouette pakai sample_size=10000. Interpretasi Silhouette: >0,70 Strong Structure; 0,51–0,70 Reasonable Structure; 0,26–0,50 Weak Structure; ≤0,25 No Substantial Structure. Noise Ratio (Pers. 2.6): 1–10% ideal, >20% terlalu ketat.
TR-4 PCA 2D fit pada data latih ter-skala → pca.joblib; simpan sampel ≤5000 titik latih (pc1, pc2, cluster) untuk latar scatter.
TR-5 Versioning: setiap pelatihan membuat artifacts/<YYYYMMDD_HHMMSS>/; current.txt menunjuk versi aktif. Pelatihan ulang ditolak selama monitoring aktif. Latih ulang TIDAK PERNAH menimpa artefak lama dan tidak mengubah current.txt; versi baru aktif hanya jika pengguna memilih "Aktifkan model ini" (atau --activate di CLI). Mengganti model aktif juga ditolak selama monitoring aktif (M8, 19 Okt 2026).
TR-6 CLI: python scripts/train.py --config config/default_config.yaml [--source csv|log] [--data path] [--no-grid] [--activate] (bisa dijadwalkan via Task Scheduler/cron). Sumber data latih ulang: CSV CIC-IDS2017 ATAU log jendela hasil monitoring.
FR-PRF Profil Klaster (semua klaster termasuk -1, satuan asli)
Untuk delay_ms, jitter_ms, fai_pct: mean, median, std (ddof=1), min, max (Pers. 2.7–2.8)
Box plot: Q1, median, Q3 via np.percentile(method="linear") (= Pers. 2.9), IQR, pagar Q1−1,5·IQR dan Q3+1,5·IQR, whisker = data min/max di dalam pagar, daftar pencilan (simpan maks 500)
Statistik pelabelan: mean delay, P99,9 jitter (p = 0,999), PLR klaster (Pers. 2.12): Σ max(fwd−bwd, 0) / Σ fwd × 100
FR-LBL Pelabelan (Pers. 2.13–2.14, ambang dari config)
L_delay: 1 jika ≤150; 2 jika 150<d≤400; 3 jika >400
L_jitter: 1 jika P99,9 ≤50; 3 jika >50
L_loss: 1 jika PLR ≤1% (loss_good_pct); 2 jika 1<PLR≤3% (loss_max_pct); 3 jika >3%. Sumber: ITU-T G.1010 (conversational voice, PLR < 3%) + panduan Cisco VoIP (≤ 1%). (Diubah 22 Okt 2026 dari satu ambang 0,1% Y.1541 atas persetujuan dosen; delay dan jitter tidak berubah.)
T = max(L_delay, L_jitter, L_loss)
Format: T=1 → "QoS Baik"; selain itu "QoS {Cukup|Buruk} - {metrik penyebab}" dengan metrik penyebab = metrik yang levelnya = T; keterangan "Meningkat" untuk level 2, "Melebihi Batas" untuk level 3. Nama metrik: Delay, Jitter, Packet Loss; gabung dengan ", " dan " dan " sebelum item terakhir. Contoh: "QoS Cukup - Delay Meningkat", "QoS Buruk - Delay dan Packet Loss Melebihi Batas"
Noise (-1): hitung level dengan cara sama; metrik berlevel 3 → "Anomali - {metrik} Ekstrem"; jika tidak ada → "Anomali"
Kategori Tindakan: Baik "-", Cukup "Pemantauan", Buruk "Perbaikan Kapasitas/Konfigurasi", Anomali "Investigasi Kasus Tunggal"
Mengubah ambang ITU cukup memicu "Terapkan Ulang Pelabelan" (pakai statistik tersimpan, tanpa refit). Hasilnya versi artefak BARU (model, scaler, PCA disalin tanpa berubah; label, tingkat, kategori, dan metrik_penyebab dihitung ulang; train_meta.json mencatat relabeled_from dan thresholds_sebelum); versi lama dan current.txt tidak disentuh. Hasil skripsi memakai ambang default; perubahan ambang hanya untuk eksplorasi.

Sumber latih ulang `log` (M8): log jendela monitoring reports/riwayat/jendela_*.csv, dengan aturan retrain.log_source: baris "Data Belum Cukup" WAJIB dikecualikan (exclude_insufficient harus true), baris berlabel Anomali dikecualikan bila exclude_anomaly = true (default), satu baris per flow (jendela terakhir) bila per_flow = last, lalu filter sesi minimum pelatihan (data.min_session). Baris dengan diimputasi = true dikecualikan (exclude_imputed, default true), dan hanya baris dengan feature_mode sama dengan retrain.log_source.feature_mode (default cumulative) yang dipakai; CSV lama tanpa kolom feature_mode/diimputasi dianggap tidak diketahui, dikecualikan, dan jumlahnya dilaporkan. Kolom `feature_mode` (mode jendela sesi) dan `diimputasi` (true bila delay/jitter diganti median artefak) ada di setiap baris CSV jendela. Jumlah minimum baris (min_rows, default 200) dicek SEBELUM pelatihan; di bawahnya latih ulang ditolak dengan pesan yang merinci jumlah baris tiap tahap. delay_ms dan jitter_ms pada log dikonversi kembali ke µs (× 1000); nilai pada log sudah melalui imputasi median.

Kolom `metrik_penyebab` ("delay;loss"; kosong untuk QoS Baik dan Data Belum Cukup) ada di setiap klaster pada cluster_profile.json dan di setiap baris CSV jendela. Rekomendasi membaca kolom ini; parser teks label hanya fallback untuk CSV/profil lama.
FR-MON Akuisisi & Inferensi Real-Time
MON-1 Backend: scapy AsyncSniffer (default sejak 19 Okt 2026), pyshark LiveCapture (opsi, butuh tshark), replay (baca .pcap/.pcapng dengan timing asli × replay_speed)
MON-2 BPF filter dari config. Default (19 Okt 2026) membuang port 443 (QUIC), 53 (DNS), 123 (NTP), 5353 (mDNS), dan 1900 (SSDP) karena trafik tersebut di luar distribusi latih (VoIP) dan hanya menghasilkan label Anomali yang menyesatkan pada uji live pertama; daftar interface via tshark -D / psutil
MON-3 Flow table key 5-tuple dua arah (kanonik); arah forward = pengirim paket pertama (seperti CICFlowMeter)
MON-4 Per window_sec (default 5 detik): hitung duration_us, iat_std_us, fwd, bwd per flow. feature_mode: cumulative (sejak awal flow, default sejak 19 Okt 2026) atau window (paket dalam jendela). Flow dengan paket < min_packets_per_window dilewati. Flow/jendela dengan fwd < monitoring.min_session_live.fwd atau bwd < monitoring.min_session_live.bwd (default 5/5, sama dengan filter sesi pelatihan) tidak masuk approximate_predict: statusnya "Data Belum Cukup", tetap dicatat di CSV jendela, dan TIDAK dihitung sebagai Anomali maupun dalam kritis_ratio.
MON-5 Nilai kosong → median dari artefak latih; scaler TIDAK di-fit ulang
MON-6 hdbscan.approximate_predict(model, X) → cluster_id + membership_prob; lookup label dari cluster_profile.json; -1 → label Anomali dari profil noise
MON-7 Tiap baris jendela di-append ke CSV secara real-time
MON-9 Ringkasan per panggilan (flow): flow dianggap selesai saat idle > min(monitoring.flow_idle_timeout_sec, monitoring.call_end_idle_sec) (call_end_idle_sec default 10 detik, agar panggilan yang ditutup cepat terdeteksi; flow_idle_timeout_sec default 120), atau saat Stop / sumber habis. Saat selesai, satu baris ditulis ke reports/riwayat/panggilan_<id_sesi>.csv (§10.3), ditampilkan di panel "Panggilan Selesai" (tabel terbaru di atas, baris berwarna sesuai label) dan memicu st.toast ("Panggilan A ↔ B selesai: <label_akhir>"). label_akhir = label jendela kumulatif terakhir; flow yang berakhir "Data Belum Cukup" tetap dicatat dengan label_akhir "Data Belum Cukup"; flow tanpa satu pun baris jendela tidak dicatat. Paket yang datang setelah flow selesai memulai flow/panggilan baru. Tidak mengubah model maupun ambang. Pada feature_mode window, nilai dan label_akhir berasal dari jendela terakhir (bukan kumulatif) dan kolom catatan memberi tahu bahwa ringkasan yang sah hanya pada mode cumulative.
MON-10 Flow sinyal SIP: flow yang salah satu portnya ada di monitoring.signaling_ports (default [5060, 5061]) berstatus "Sinyal SIP": tidak masuk approximate_predict (cluster_id kosong), tidak dihitung di kartu, kritis_ratio, total_aliran dan ringkasan sesi, tidak masuk panggilan_<id_sesi>.csv, tidak mendapat rekomendasi, dan tidak dipakai latih ulang. Tetap dicatat di CSV jendela dengan label "Sinyal SIP"; dashboard menampilkan jumlahnya sebagai catatan kecil. monitoring.min_session_live tidak berubah.
MON-11 Tampilan panggilan: tabel dashboard dan per sesi menampilkan kolom ringkas (waktu_selesai, pasangan alamat, durasi_s, label_akhir, label_terburuk) berlatar warna label; kolom lengkap di expander "Detail" dan CSV. Jika lebih dari satu panggilan selesai dalam satu siklus refresh, tampil satu toast gabungan ("3 panggilan selesai: 2 QoS Baik, 1 Anomali"); satu panggilan memakai format rinci.
MON-12 Launcher lokal: run_app.bat (aktifkan .venv, Streamlit headless, buka browser) dan run_desktop.bat → scripts/desktop_app.py (server Streamlit di background, jendela pywebview "VoIP QoS Monitor"; server berhenti saat jendela ditutup).
MON-13 Indikator live (Dashboard, paling atas, st.fragment run_every=refresh_sec): (a) banner "Status Koneksi Saat Ini" berwarna = label terparah (Baik<Cukup<Buruk<Anomali) di antara panggilan aktif pada jendela terakhir; "Data Belum Cukup" bila belum ada prediksi, "Menunggu data" bila tak ada panggilan aktif; (b) status capture: paket tertangkap, flow aktif, waktu paket terakhir, dan peringatan jika sesi berjalan > 2×window_sec tanpa paket (interface salah / BPF / izin); (c) panel "Panggilan Aktif": satu kartu per flow aktif (bukan Sinyal SIP) berisi pasangan alamat, durasi berjalan, badge label jendela terbaru, delay/jitter/FAI terkini, dan sparkline kategori per jendela (monitoring.sparkline_windows). Panggilan hilang dari panel saat selesai (MON-9). Interface bawaan di sidebar = interface pertama yang aktif dan punya IPv4 non-link-local; interface tidak aktif diberi keterangan.
MON-14 Sesi terputus: ringkasan sesi selalu tertulis. Tanpa Stop (aplikasi ditutup, error capture) ringkasan ditulis dengan status_keseluruhan "Dihentikan tidak normal" (atexit / error worker; skor model dari ringkasan normal tidak tersedia pada pemulihan). Jika proses dimatikan paksa (tak ada atexit), CSV jendela tanpa ringkasan yang tak berubah selama monitoring.orphan_after_sec diberi ringkasan abnormal saat aplikasi dibuka berikutnya. Halaman Rekomendasi & Ekspor memilih sesi terbaru otomatis dan menjelaskan sesi tanpa data jendela; halaman Analisis & Evaluasi Klaster memberi keterangan bahwa halamannya untuk evaluasi model dengan tautan ke hasil sesi.
MON-8 (M6: tidak ada fallback otomatis antar-backend; backend dipilih eksplisit lewat monitoring.capture_backend) Error jelas di UI jika: tshark tidak ditemukan, izin capture ditolak (butuh Administrator/root), atau tidak ada model aktif
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
Box plot per metrik per klaster (termasuk noise) dari statistik tersimpan memakai go.Box(q1, median, q3, lowerfence, upperfence); garis ambang 150 & 400 ms (delay), 50 ms (jitter), 1% dan 3% (loss); penanda mean delay, P99,9 jitter, PLR. Toggle skala log. Profil TIDAK dihitung ulang per jendela.
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
Input ambang: delay baik 150, delay maks 400, jitter maks 50, loss baik 1, loss maks 3
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
  "thresholds": {"delay_good_ms": 150, "delay_max_ms": 400, "jitter_max_ms": 50, "loss_good_pct": 1.0, "loss_max_pct": 3.0},
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
      "kategori_tindakan": "Perbaikan Kapasitas/Konfigurasi",
      "metrik_penyebab": "loss"
    }
  ],
  "label_map": {"0": "QoS Buruk - Packet Loss Melebihi Batas", "-1": "Anomali - Delay dan Packet Loss Ekstrem"}
}

(statistik & boxplot berisi delay_ms, jitter_ms, fai_pct untuk setiap klaster)

10.3 Riwayat
reports/riwayat/jendela_<id_sesi>.csv (append tiap jendela; label "Sinyal SIP" untuk flow sinyal, MON-10): waktu_jendela, id_sesi, flow_id, delay_ms, jitter_ms, fwd, bwd, fai_pct, cluster_id, membership_prob, label, kategori_tindakan, metrik_penyebab, pc1, pc2, feature_mode, diimputasi
reports/riwayat/panggilan_<id_sesi>.csv (satu baris per panggilan selesai, MON-9): flow_id, waktu_mulai, waktu_selesai, durasi_s, total_fwd, total_bwd, delay_ms, jitter_ms, fai_pct (nilai kumulatif akhir), cluster_id, label_akhir, kategori_tindakan, metrik_penyebab, jumlah_jendela, persen_baik, persen_cukup, persen_buruk, persen_anomali, persen_belum_cukup (persen jendela per kategori; jendela "Data Belum Cukup" dihitung agar jumlahnya 100%), label_terburuk (label kategori tertinggi Baik<Cukup<Buruk<Anomali; seri → paling sering; "Data Belum Cukup" jika tak ada jendela terprediksi), catatan (kosong pada mode cumulative; peringatan pada mode window)
reports/riwayat/riwayat_pemantauan_<YYYYMMDD_HHMMSS>.csv (ringkasan sesi saat Stop): id_sesi_pemantauan (interface + waktu mulai), waktu_mulai, waktu_selesai, total_aliran, jumlah_klaster, skor_silhouette, skor_chi, noise_ratio, status_keseluruhan, lokasi_laporan, jumlah_panggilan, panggilan_baik, panggilan_cukup, panggilan_buruk, panggilan_anomali, panggilan_belum_cukup (jumlah panggilan per kategori label_akhir)
jumlah_klaster dan skor_* diambil dari model aktif (hasil pelatihan)
status_keseluruhan: "Dihentikan tidak normal" untuk sesi tanpa Stop (MON-14); "Kritis" jika proporsi (QoS Buruk + Anomali) > kritis_ratio; selain itu label mayoritas
11. config/default_config.yaml
yaml
data:
  raw_dir: data/raw
  min_session:
    fwd: 5
    bwd: 5
  exclude_traffic_types: [Tor]
  max_train_rows: 200000
  random_state: 42
  voip_filter:
    protocol: 17
    sip_ports: [5060, 5061]
    rtp_port_ranges: [[10000, 20000], [16384, 32767]]
    min_flows_warning: 200
    fallback_udp_all: false
features:
  fai_clip_negative: true
  log_transform: []
  delay_column: iat_mean_us
  jitter_column: iat_std_us
  cluster_columns: [iat_mean_us, iat_std_us, fai_pct]
  log_transform: [iat_mean_us, iat_std_us]
hdbscan:
  min_cluster_size: 5
  min_samples: 2
  cluster_selection_method: eom
  grid_search:
    enabled: true
    min_cluster_size: [5, 10, 15, 20, 25]
    min_samples: [1, 2, 5, 10, 20]
    max_noise_ratio: 0.20
    silhouette_sample_size: 10000
evaluation:
  silhouette_bands: [0.70, 0.50, 0.25]
  noise_ideal: [0.01, 0.10]
  noise_too_strict: 0.20
  target_silhouette: 0.50
  pca_sample_size: 5000
profiling:
  jitter_percentile: 99.9
  max_outliers: 500
retrain:
  log_source:
    exclude_insufficient: true
    exclude_anomaly: true
    per_flow: last
    exclude_imputed: true
    feature_mode: cumulative
    min_rows: 200
thresholds:
  delay_good_ms: 150.0
  delay_max_ms: 400.0
  jitter_max_ms: 50.0
  loss_good_pct: 1.0
  loss_max_pct: 3.0
monitoring:
  capture_backend: scapy     # scapy (default) | pyshark | replay
  bpf_filter: "udp and not port 443 and not port 53 and not port 123 and not port 5353 and not port 1900"
  tshark_path: null
  live_poll_sec: 0.5
  window_sec: 5
  flow_idle_timeout_sec: 120
  call_end_idle_sec: 10
  signaling_ports: [5060, 5061]
  sparkline_windows: 20
  exit_join_sec: 5
  orphan_after_sec: 120
  feature_mode: cumulative   # cumulative (default) | window
  min_session_live:
    fwd: 5
    bwd: 5
  min_packets_per_window: 10
  refresh_sec: 2
  replay_speed: 1.0
  kritis_ratio: 0.5
paths:
  artifacts_dir: artifacts
  reports_dir: reports
  logs_dir: logs
  recommendations_file: config/recommendations.yaml

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
Label	QoS Baik (PLR 0,99% ≤ 1%; naskah lama dengan ambang 0,1%: QoS Buruk - Packet Loss Melebihi Batas)	QoS Buruk - Delay dan Packet Loss Melebihi Batas	Anomali - Delay dan Packet Loss Ekstrem

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
Hasil Varian C (19 Okt 2026, model 20261019_175647: 8 klaster + noise 13,77%, Silhouette 0,8866): tidak ada klaster "QoS Baik"; semua klaster berlabel "QoS Buruk" dan noise "Anomali". Klaster terbesar (123 flow, delay ≈ 10 ms) berlabel Buruk hanya karena PLR 0,1053% sedikit di atas ambang 0,1%. Mode window menghasilkan FAI ≈ 0,13% pada RTP tanpa loss karena efek batas jendela (selisih satu paket antara fwd dan bwd di tepi jendela), sehingga cumulative dijadikan default (19 Okt 2026). Resolusi FAI vs ambang 0,1%: FAI diskret (selisih satu paket pada flow berisi ~6000 paket = 0,0167%, dan pada flow < 1000 paket satu paket selisih sudah > 0,1%), sehingga ambang 0,1% nyaris berarti "tidak ada selisih paket sama sekali" dan sangat sensitif terhadap satu-dua paket. P99,9 jitter pada klaster kecil (n < 1000) berinterpolasi ke ujung atas data sehingga ≈ nilai maksimum klaster (rasio P99,9/maks 0,90–1,00 pada model ini), bukan persentil yang stabil. Ambang tidak diubah. Dataset bukan dirancang khusus untuk VoIP; jumlah flow lolos filter kecil → dicek di M1 (19 Okt 2026): 3.566 flow berlabel VOIP di Darknet.CSV, 247 flow setelah exclude Tor dan fwd/bwd ≥ 5 (193 UDP, 54 TCP; NonVPN 153, VPN 94), sehingga min_flows_warning diturunkan ke 200 dan grid min_cluster_size dibatasi ≤ 25
Semantik fitur: (diubah 19 Okt 2026: delay kini memakai Flow IAT Mean karena Flow Duration menempel ke batas 120 detik dan jauh di atas ambang ITU) Flow Duration CIC = durasi seluruh flow, sedangkan live dihitung per jendela → distribusi bisa bergeser; bandingkan window vs cumulative saat M4
Ambang loss 3 level (diubah 22 Okt 2026, disetujui dosen): ≤1% baik, 1–3% cukup, >3% buruk; sumber ITU-T G.1010 (conversational voice, PLR < 3%) + panduan Cisco VoIP (≤ 1%). Alasan: ambang Y.1541 0,1% terlalu ketat terhadap estimasi FAI yang diskret (lihat catatan 19 Okt 2026) sehingga tidak ada klaster "QoS Baik". Hasil relabel model 20261019_204042 → 20261022_120925 (tanpa refit): QoS Baik 49,8% (klaster 5, 123 flow), Cukup 0%, Buruk 36,4%, Anomali 13,8%; hanya klaster 5 (→ Baik) dan 7 (→ "Jitter Melebihi Batas") berubah label. "QoS Cukup" tetap kosong pada model ini karena klaster dengan loss level 2 (klaster 7) juga melanggar jitter. Rincian skenario S0–S3: reports/experiments/labeling_scenarios.md. Lampiran 5: label K0 kini "QoS Baik" (PLR 0,99% ≤ 1%).
min_cluster_size kecil pada data besar → klaster sangat banyak; andalkan grid search
Silhouette O(n²) → wajib sample_size
Live capture butuh hak Administrator/root dan tshark/Npcap