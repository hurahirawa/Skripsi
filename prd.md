Versi 1.0 · 12 Oktober 2026 · Pemilik: Hizkia Rahuel Anglie (535230073), Teknik Informatika UNTAR Catatan perubahan 19 Okt 2026: sumber data latih = CIC-Darknet2020 (Darknet.CSV, filter kategori VOIP), bukan CIC-IDS2017; proksi delay = Flow IAT Mean; filter sesi minimum fwd/bwd ≥ 5 dan exclude Tor; FAI di-clip ke ≥ 0; min_flows_warning 200; grid min_cluster_size [5, 10, 15, 20, 25]. Ambang ITU tidak berubah. Sumber kebenaran: skripsi "Pemetaan Karakteristik VoIP Traffic Network untuk Identifikasi QoS menggunakan HDBSCAN"

Catatan perubahan 27 Okt 2026 (revisi dosen, Tahap 2): aplikasi diubah menjadi KLASTERISASI MURNI. Keluaran akhir = "paling dekat ke Klaster N" berdasarkan jarak Euclidean ke centroid, BUKAN label QoS Baik/Cukup/Buruk/Anomali. Data latih = 247 flow (VOIP, tanpa Tor, fwd ≥ 5, bwd ≥ 5); algoritma utama HDBSCAN (min_cluster_size=3, min_samples=8, dipilih dengan aturan hdbscan.selection: noise ≤ 20%, klaster ≥ 2, ARI bootstrap ≥ 0,90, DBCV tertinggi); pembanding K-Means (k = jumlah klaster HDBSCAN dan k optimal). Ambang ITU tetap ada tetapi hanya untuk deskripsi klaster, garis referensi grafik, dan saran. Dasar keputusan: reports/experiments/cluster_analysis.md bagian F dan G. Bagian PRD di bawah yang menyebut label QoS, approximate_predict, Kritis, atau Terapkan Ulang Pelabelan digantikan oleh catatan ini dan bagian bertanda (T2).


1. Ringkasan

Web Dashboard lokal (Streamlit) untuk administrator jaringan/NOC yang:

Melatih model HDBSCAN secara offline dari statistik aliran CIC-Darknet2020 (unsupervised, tanpa label), dengan K-Means sebagai pembanding (T2).
Membentuk klaster bernomor 1..k (urut ukuran menurun); noise HDBSCAN = "Noise". Setiap klaster diprofilkan dalam satuan asli dan dideskripsikan posisinya terhadap ambang ITU-T G.114 (delay), ITU-T Y.1541 Kelas 0/1 (jitter), ITU-T G.1010 dan panduan Cisco VoIP (packet loss) sebagai TEKS DESKRIPTIF, bukan label kualitas (T2).
Menugaskan trafik VoIP live per jendela waktu ke klaster terdekat berdasarkan jarak Euclidean ke centroid (ruang ter-skala). Titik yang jaraknya ke SETIAP centroid melebihi batas klaster tersebut ditandai "Di luar klaster (terdekat: Klaster N)" (T2).
Menampilkan klaster terdekat, jarak, visualisasi, dan saran berdasarkan karakteristik klaster secara real-time, serta menyimpan riwayat dalam CSV.

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
hdbscan (WAJIB paket ini, BUKAN sklearn.cluster.HDBSCAN; juga hdbscan.validity untuk DBCV); K-Means dari scikit-learn sebagai pembanding
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
artifacts: model HDBSCAN, K-Means, scaler, pca, assignment.json, cluster_profile.json
NIC / port mirroring / replayPCAP
Flow table 5-tuple + slidingwindow
4 fitur per jendela +imputasi median latih
scaler.transform
Jarak Euclidean ke centroid
Klaster terdekat / di luar klaster
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
scaler.transform, jarak Euclidean ke centroid, klaster terdekat / di luar klaster
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
FR-TR Pelatihan (T2)
TR-1 hdbscan.HDBSCAN(min_cluster_size, min_samples, metric="euclidean", cluster_selection_method="eom", prediction_data=True). Paket hdbscan (bukan sklearn.cluster.HDBSCAN).
TR-2 Grid search otomatis (hdbscan.grid_search: min_cluster_size × min_samples) dengan ARI bootstrap (hdbscan.stability: 50×, subsampel 80%) untuk SEMUA kandidat. Aturan seleksi final (hdbscan.selection): kandidat valid = noise ≤ 20%, n_klaster ≥ 2, ARI mean ≥ 0,90; pilih DBCV tertinggi; seri → Silhouette tertinggi; seri → min_cluster_size lalu min_samples terkecil. Jika tidak ada kandidat valid, dipakai hdbscan.min_cluster_size/min_samples dan dicatat grid_fallback. Hasil pada 247 flow: min_cluster_size=3, min_samples=8, 5 klaster, noise 13,36%, Silhouette 0,8767, DBCV 0,7404, ARI 0,964 ± 0,024. grid_results.csv memuat semua kandidat dengan metrik, ari_mean, ari_sd, valid.
TR-3 Evaluasi tersimpan (cluster_profile.json: evaluation dan comparison): Silhouette (sample_size=10000), Davies-Bouldin, Calinski-Harabasz, DBCV (hdbscan.validity; di atas evaluation.dbcv_max_points dihitung pada subsampel), Noise Ratio, ARI bootstrap mean ± sd, waktu fit. Dihitung untuk HDBSCAN (tanpa noise dan noise di-assign ke centroid terdekat) serta K-Means k = jumlah klaster HDBSCAN dan K-Means k optimal (rata-rata peringkat Silhouette dan DBI pada k_min..k_max; kurva Elbow/Silhouette/DBI/CHI disimpan di kmeans_sweep). Interpretasi Silhouette: >0,70 Strong Structure; 0,51–0,70 Reasonable; 0,26–0,50 Weak; ≤0,25 No Substantial Structure. Noise Ratio: 1–10% ideal, >20% terlalu ketat.
TR-4 PCA 2D fit pada data latih ter-skala → pca.joblib; simpan sampel ≤5000 titik latih (pc1, pc2, cluster_id, jarak_terdekat, di_luar_klaster, klaster_kmeans_n, klaster_kmeans_opt) untuk latar scatter.
TR-5 Versioning: setiap pelatihan membuat artifacts/<YYYYMMDD_HHMMSS>/; current.txt menunjuk versi aktif. Pelatihan ulang ditolak selama monitoring aktif. Latih ulang TIDAK PERNAH menimpa artefak lama dan tidak mengubah current.txt; versi baru aktif hanya jika pengguna memilih "Aktifkan model ini" (atau --activate di CLI). Versi lama (pelabelan QoS: cluster_profile.json tanpa schema "klaster_murni") ditolak saat aktivasi dan saat dimuat dengan pesan jelas; daftar versi menandainya kompatibel = False.
TR-6 CLI: python scripts/train.py --config config/default_config.yaml [--source csv|log] [--data path] [--no-grid] [--activate]. Sumber data latih ulang: CSV Darknet/CIC ATAU log jendela hasil monitoring.
FR-PRF Profil Klaster (klaster 1..k dan Noise, satuan asli)
Untuk delay_ms, jitter_ms, fai_pct: mean, median, std (ddof=1), min, max; box plot (Q1, median, Q3 via np.percentile linear, IQR, pagar 1,5·IQR, whisker, pencilan maks 500). Statistik tambahan (statistik_tambahan): mean delay, P99,9 jitter, PLR klaster = Σ max(fwd−bwd,0)/Σ fwd × 100 (deskriptif; bukan dasar klasterisasi).
Karakter klaster (karakter): median total paket (fwd+bwd), median durasi flow (detik), % UDP/TCP/lain; dipakai di deskripsi ringkas ("Klaster 2 · 32 paket · 90 dtk · Delay >400 · Jitter >50 · Loss ≤1"), tabel profil, kartu dan banner dashboard. Profil lama tanpa karakter dilengkapi dari training_sample.csv saat Perbarui Deskripsi Klaster (tanpa refit).
Per klaster non-noise juga disimpan: centroid dalam satuan asli (centroid_asli), batas_jarak, jarak_sendiri (median/p95/maks jarak anggota ke centroid).
FR-DSK Deskripsi posisi terhadap ambang ITU (pengganti FR-LBL; deskriptif, ambang dari config)
Dari median klaster: delay ≤150 / 150–400 / >400 ms; jitter ≤50 / >50 ms; loss (FAI sebagai proksi) ≤1 / 1–3 / >3 %. Disimpan sebagai deskripsi (teks ringkas, posisi per metrik) dan metrik_melewati_ambang (metrik di atas batas maksimum). TIDAK ADA label Baik/Cukup/Buruk/Anomali, kategori tindakan, atau status Kritis.
Mengubah ambang ITU hanya memicu "Perbarui Deskripsi Klaster" (statistik tersimpan, tanpa refit): versi artefak BARU (model, scaler, PCA, centroid, batas disalin tanpa berubah; deskripsi dihitung ulang; train_meta.json mencatat deskripsi_dari dan thresholds_sebelum); versi lama dan current.txt tidak disentuh.
FR-ASG Penugasan Euclidean (T2)
Centroid = rata-rata anggota klaster di ruang ter-skala (assignment.json: hdbscan.centroid; kmeans_n dan kmeans_opt untuk pembanding). Klaster terdekat = jarak Euclidean minimum. Batas "di luar klaster" (assignment.limit_rule, default max_cluster_global = aturan (c) analisis G): per klaster max(p95 jarak klaster ke centroid sendiri, p95 global seluruh titik non-noise); aturan lain: per_cluster, global. Titik ditandai di luar klaster bila jaraknya ke SETIAP centroid melebihi batas klaster tersebut; klaster terdekat tetap dilaporkan. Pada 247 flow: batas global 0,443; batas per klaster K1 0,443, K2 0,443, K3 0,443, K4 0,788, K5 0,569.

Sumber latih ulang `log` (M8): log jendela monitoring reports/riwayat/jendela_*.csv, dengan aturan retrain.log_source: baris status "Data Belum Cukup" dan "Sinyal SIP" WAJIB dikecualikan (exclude_insufficient harus true), baris di luar klaster dikecualikan bila exclude_outside_cluster = true (default), satu baris per flow (jendela terakhir) bila per_flow = last, lalu filter sesi minimum pelatihan (data.min_session). Baris diimputasi dikecualikan (exclude_imputed) dan hanya feature_mode sesuai retrain.log_source.feature_mode (default cumulative) yang dipakai; CSV lama tanpa kolom feature_mode/diimputasi dianggap tidak diketahui. Log format lama (kolom label) tetap terbaca untuk status. min_rows (default 200) dicek SEBELUM pelatihan. delay_ms dan jitter_ms pada log dikonversi kembali ke µs (× 1000).
FR-MON Akuisisi & Inferensi Real-Time
MON-1 Backend: scapy AsyncSniffer (default sejak 19 Okt 2026), pyshark LiveCapture (opsi, butuh tshark), replay (baca .pcap/.pcapng dengan timing asli × replay_speed)
MON-2 BPF filter dari config. Default (19 Okt 2026) membuang port 443 (QUIC), 53 (DNS), 123 (NTP), 5353 (mDNS), dan 1900 (SSDP) karena trafik tersebut di luar distribusi latih (VoIP) dan hanya menghasilkan label Anomali yang menyesatkan pada uji live pertama; daftar interface via tshark -D / psutil
MON-3 Flow table key 5-tuple dua arah (kanonik); arah forward = pengirim paket pertama (seperti CICFlowMeter)
MON-4 Per window_sec (default 5 detik): hitung duration_us, iat_std_us, fwd, bwd per flow. feature_mode: cumulative (sejak awal flow, default) atau window (paket dalam jendela). Flow dengan paket < min_packets_per_window dilewati. Flow/jendela dengan fwd < monitoring.min_session_live.fwd atau bwd < monitoring.min_session_live.bwd (default 5/5) tidak ditugaskan ke klaster: statusnya "Data Belum Cukup" dan tetap dicatat di CSV jendela.
MON-5 Nilai kosong → median dari artefak latih; scaler TIDAK di-fit ulang
MON-6 (T2) Penugasan Euclidean: fitur → scaler.transform → jarak ke semua centroid HDBSCAN → klaster terdekat. status = "Terklaster" bila ada centroid yang jaraknya dalam batas klaster tersebut, selain itu "Di luar klaster" (UI: "Di luar klaster (terdekat: Klaster N)"). K-Means pembanding (k = jumlah klaster HDBSCAN) menugaskan dengan cara yang sama ke centroid-nya. approximate_predict tidak dipakai.
MON-7 Tiap baris jendela di-append ke CSV secara real-time
MON-9 Ringkasan per panggilan (flow): flow dianggap selesai saat idle > min(monitoring.flow_idle_timeout_sec, monitoring.call_end_idle_sec) (default 120 dan 10 detik), atau saat Stop / sumber habis. Saat selesai, satu baris ditulis ke reports/riwayat/panggilan_<id_sesi>.csv (§10.3), ditampilkan di panel "Panggilan Selesai" (baris berwarna sesuai klaster akhir) dan memicu st.toast ("Panggilan A ↔ B selesai: paling dekat ke Klaster N" / "Di luar klaster (terdekat: Klaster N)"). klaster_akhir/jarak_akhir/di_luar_klaster_akhir = jendela kumulatif terakhir; flow yang berakhir "Data Belum Cukup" tetap dicatat; flow tanpa baris jendela tidak dicatat. Pada feature_mode window nilai berasal dari jendela terakhir dan kolom catatan memberi peringatan.
MON-10 Flow sinyal SIP: flow yang salah satu portnya ada di monitoring.signaling_ports (default [5060, 5061]) berstatus "Sinyal SIP": tidak ditugaskan ke klaster, tidak dihitung di kartu, total_aliran, dan ringkasan sesi, tidak masuk panggilan_<id_sesi>.csv, tidak mendapat saran, dan tidak dipakai latih ulang. Tetap dicatat di CSV jendela; dashboard menampilkan jumlahnya sebagai catatan kecil.
MON-11 Tampilan panggilan: tabel dashboard dan per sesi menampilkan kolom ringkas (waktu_selesai, pasangan alamat, durasi_s, klaster akhir, jarak_akhir, % jendela di luar klaster) berlatar warna klaster; kolom lengkap di expander "Detail" dan CSV. Beberapa panggilan selesai dalam satu siklus refresh digabung dalam satu toast ("3 panggilan selesai: 2 di Klaster 1, 1 di luar klaster").
MON-12 Launcher lokal: run_app.bat (aktifkan .venv, Streamlit headless, buka browser) dan run_desktop.bat → scripts/desktop_app.py (server Streamlit di background, jendela pywebview "VoIP QoS Monitor"; server berhenti saat jendela ditutup).
MON-13 Indikator live (Dashboard, paling atas, st.fragment run_every=refresh_sec): (a) banner "Status Saat Ini: paling dekat ke Klaster N (jarak X)" + deskripsi singkat klaster, berwarna klaster (abu-abu untuk "Di luar klaster (terdekat: Klaster N)"); kelompok terbanyak di antara panggilan aktif menentukan banner (seri → klaster bernomor kecil); "Data Belum Cukup" / "Menunggu data" bila belum ada penugasan; (b) status capture: paket, flow aktif, paket terakhir, peringatan bila > 2×window_sec tanpa paket; (c) kartu jumlah flow per klaster + Di luar klaster; (d) panel "Panggilan Aktif": per flow aktif klaster terdekat, jarak (dan batas), bar kecil jarak ke semua centroid, delay/jitter/FAI terkini, sparkline klaster per jendela. Interface bawaan di sidebar = interface pertama yang aktif dan punya IPv4 non-link-local.
MON-14 Sesi terputus: ringkasan sesi selalu tertulis. Tanpa Stop (aplikasi ditutup, error capture) status_sesi "Dihentikan tidak normal" (atexit / error worker). Jika proses dimatikan paksa, CSV jendela tanpa ringkasan yang tak berubah selama monitoring.orphan_after_sec diberi ringkasan abnormal saat aplikasi dibuka berikutnya. Halaman Saran & Ekspor memilih sesi terbaru otomatis; halaman Analisis & Evaluasi Klaster memberi keterangan bahwa halamannya untuk evaluasi model.
MON-8 (M6: tidak ada fallback otomatis antar-backend; backend dipilih eksplisit lewat monitoring.capture_backend) Error jelas di UI jika: tshark tidak ditemukan, izin capture ditolak (butuh Administrator/root), atau tidak ada model aktif
FR-REC Saran berdasarkan karakteristik klaster (rule-based, config/recommendations.yaml; T2)

Dipicu oleh metrik median klaster yang melewati batas maksimum ITU (metrik_melewati_ambang) dan oleh titik "Di luar klaster". Dapat dimatikan: recommendations.enabled: false (default true) menghilangkan saran di UI dan PDF.

Pemicu	Jenis tindakan	Saran
Delay melewati ambang ITU	Perbaikan Kapasitas/Konfigurasi	Perlebar bandwidth jalur VoIP, prioritas antrean DiffServ (EF), evaluasi beban router dan jalur routing
Jitter melewati ambang ITU	Perbaikan Kapasitas/Konfigurasi	Priority queuing untuk RTP, periksa kongesti/buffer antrean, sesuaikan jitter buffer endpoint
Packet loss (FAI) melewati ambang ITU	Perbaikan Kapasitas/Konfigurasi	Periksa utilisasi antrean dan buffer router, error interface, kualitas media transmisi
Di luar klaster	Investigasi Kasus Tunggal	Periksa potensi flood/burst, gangguan fisik media transmisi, kegagalan gateway; cocokkan timestamp dengan log perangkat

Klaster dengan beberapa metrik di atas ambang menggabungkan saran semua metrik. Judul di UI: "Saran berdasarkan karakteristik klaster".

FR-VIZ Visualisasi (T2)
Warna: palet kategorikal per klaster (10 warna, berulang bila klaster lebih banyak) tanpa makna baik/buruk; abu-abu tua (#6B7280) untuk "Di luar klaster"/Noise; abu-abu muda untuk Data Belum Cukup dan Sinyal SIP.
PCA scatter 2D (hover, zoom, pan): latar titik latih pudar per klaster, titik jendela live (berwarna klaster terdekat; di luar klaster = X abu-abu), centroid sebagai bintang besar.
Analisis: tabel perbandingan HDBSCAN (dua versi) vs K-Means, grafik Elbow/Silhouette/DBI/CHI per k, heatmap grid HDBSCAN (termasuk ARI), tabel profil + deskripsi ITU, box plot per klaster dengan garis ambang ITU sebagai referensi (toggle skala log), tabel centroid dan batas jarak.

9. Halaman UI (sesuai Lampiran 7 skripsi)

P1 Dashboard Overview

Sidebar: navigasi, pemilih interface, mode (Live/Replay PCAP + uploader), Start/Stop Monitoring, versi model aktif
Banner Status Saat Ini, status capture, kartu: Total Aliran Trafik, Klaster 1..k, Di luar klaster (flow unik menurut status terakhir)
Panggilan Aktif, PCA scatter live dengan centroid, tabel deskripsi klaster, saran (bila enabled), Panggilan Selesai, tabel jendela terbaru

P2 Analisis & Evaluasi Klaster

Kartu Silhouette (+ interpretasi), DBI, CHI, DBCV, Noise Ratio, ARI bootstrap; tabel perbandingan HDBSCAN vs K-Means; kurva K-Means; heatmap grid; tabel profil + deskripsi ITU; box plot; tabel centroid dan batas jarak; sampel data latih

P3 Pengaturan Parameter

Slider min_cluster_size 2–50 (default 3), min_samples 1–20 (default 8), toggle grid search; aturan seleksi (noise maks, klaster minimum, ARI minimum); aturan batas jarak (aturan, persentil); ambang ITU (hanya deskripsi/referensi/saran)
Tombol Save Configuration, Reset to Default, Perbarui Deskripsi Klaster (tanpa refit), Latih Ulang Model (disabled saat monitoring dan sebelum checkbox "Saya mengerti ini membuat versi model baru" dicentang; progress bar; setelah selesai tampilkan skor + tombol "Aktifkan model ini"), daftar versi dengan kolom kompatibel

P4 Saran berdasarkan karakteristik klaster & Ekspor Laporan

Saran per klaster (model aktif) dan per sesi/panggilan (bila recommendations.enabled)
Tabel riwayat sesi (dari CSV)
Unduh laporan per sesi: CSV (jendela, panggilan) dan PDF ringkasan
10. Artefak & Output
10.1 artifacts/<versi>/ (T2)

hdbscan_model.joblib, kmeans_n.joblib (k = jumlah klaster HDBSCAN), kmeans_opt.joblib (k optimal), scaler.joblib, pca.joblib, assignment.json (centroid ter-skala HDBSCAN dan K-Means, batas per klaster, batas global, aturan, persentil), cluster_profile.json (schema "klaster_murni", evaluation, comparison, kmeans_sweep, clusters), train_meta.json (parameter, aturan seleksi, ambang, median imputasi, jumlah baris, filter VoIP, timestamp), grid_results.csv (dengan ARI), training_sample.csv

10.2 cluster_profile.json (ringkas)
json
{
  "schema": "klaster_murni", "schema_version": 2, "model_version": "20261027_005150",
  "thresholds": {"delay_good_ms": 150, "delay_max_ms": 400, "jitter_max_ms": 50, "loss_good_pct": 1.0, "loss_max_pct": 3.0},
  "evaluation": {"silhouette": 0.8767, "davies_bouldin": 0.2576, "chi": 3564.63, "dbcv": 0.7404, "noise_ratio": 0.1336, "n_clusters": 5, "ari_mean": 0.964, "ari_sd": 0.024, "silhouette_label": "Strong Structure"},
  "comparison": {"hdbscan": {}, "hdbscan_assigned": {}, "kmeans_n": {}, "kmeans_opt": {}},
  "kmeans_sweep": [{"k": 2, "inertia": 0.0, "silhouette": 0.0}],
  "clusters": [
    {
      "id_klaster": 1, "nama": "Klaster 1", "jumlah_anggota": 122,
      "statistik": {"delay_ms": {"mean": 9.96, "median": 10.0, "std": 0.5, "min": 9.0, "max": 12.0}},
      "boxplot": {"delay_ms": {"q1": 0, "median": 0, "q3": 0, "iqr": 0, "lower_fence": 0, "upper_fence": 0, "whisker_low": 0, "whisker_high": 0, "outliers": []}},
      "statistik_tambahan": {"mean_delay_ms": 9.96, "p999_jitter_ms": 0.0, "plr_pct": 0.0},
      "centroid_asli": {"delay_ms": 9.9, "jitter_ms": 5.9, "fai_pct": 0.1},
      "batas_jarak": 0.443, "jarak_sendiri": {"median": 0.06, "p95": 0.124, "maks": 0.52},
      "deskripsi": {"ringkas": "Delay ≤150 ms · Jitter ≤50 ms · Packet Loss ≤1 %", "teks": "...", "melewati_ambang": []},
      "metrik_melewati_ambang": []
    }
  ]
}

(statistik dan boxplot berisi delay_ms, jitter_ms, fai_pct; klaster Noise bernomor -1, nama "Noise", tanpa centroid dan batas)

10.3 Riwayat (T2)
reports/riwayat/jendela_<id_sesi>.csv (append tiap jendela): waktu_jendela, id_sesi, flow_id, delay_ms, jitter_ms, fwd, bwd, fai_pct, status (Terklaster | Di luar klaster | Data Belum Cukup | Sinyal SIP), klaster_terdekat, jarak_terdekat, di_luar_klaster, jarak_ke_klaster (format "1:0.123;2:1.402;…", titik desimal), klaster_kmeans5 (klaster terdekat menurut K-Means pembanding, k = jumlah klaster HDBSCAN), pc1, pc2, feature_mode, diimputasi
reports/riwayat/panggilan_<id_sesi>.csv (satu baris per panggilan selesai, MON-9): flow_id, waktu_mulai, waktu_selesai, durasi_s, total_fwd, total_bwd, delay_ms, jitter_ms, fai_pct, klaster_akhir, jarak_akhir, di_luar_klaster_akhir, status_akhir, jumlah_jendela, persen_per_klaster ("1:62.5;2:12.5", jendela Terklaster per klaster), persen_di_luar_klaster, persen_belum_cukup (ketiganya berjumlah ±100%), klaster_terbanyak, pernah_di_luar_klaster, catatan
reports/riwayat/riwayat_pemantauan_<YYYYMMDD_HHMMSS>.csv (ringkasan sesi saat Stop): id_sesi_pemantauan, waktu_mulai, waktu_selesai, total_aliran, jumlah_klaster, skor_silhouette, skor_dbcv, skor_chi, noise_ratio, status_sesi (Selesai | Dihentikan tidak normal | Tidak ada data), lokasi_laporan, jumlah_panggilan, panggilan_per_klaster ("1:3;2:1"), panggilan_di_luar_klaster, persen_di_luar_klaster (dari panggilan yang dinilai), klaster_mayoritas, panggilan_belum_cukup
jumlah_klaster dan skor_* diambil dari model aktif (hasil pelatihan). Tidak ada status "Kritis" dan tidak ada kritis_ratio.
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
  min_cluster_size: 3
  min_samples: 8
  cluster_selection_method: eom
  grid_search:
    enabled: true
    min_cluster_size: [3, 5, 8, 10, 15, 20, 30, 50]
    min_samples: [1, 2, 3, 5, 8, 10, 15]
    silhouette_sample_size: 10000
  selection:
    max_noise_ratio: 0.20
    min_clusters: 2
    min_ari_mean: 0.90
  stability:
    n_bootstrap: 50
    fraction: 0.8
kmeans:
  k_min: 2
  k_max: 10
  n_init: 10
  rank_metrics: [silhouette, davies_bouldin]
assignment:
  limit_percentile: 95
  limit_rule: max_cluster_global   # per_cluster | global | max_cluster_global
recommendations:
  enabled: true
evaluation:
  silhouette_bands: [0.70, 0.50, 0.25]
  noise_ideal: [0.01, 0.10]
  noise_too_strict: 0.20
  target_silhouette: 0.50
  pca_sample_size: 5000
  dbcv_max_points: 5000
profiling:
  jitter_percentile: 99.9
  max_outliers: 500
retrain:
  log_source:
    exclude_insufficient: true
    exclude_outside_cluster: true
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
paths:
  artifacts_dir: artifacts
  reports_dir: reports
  logs_dir: logs
  recommendations_file: config/recommendations.yaml

Konfigurasi aktif dikelola di st.session_state; Save Configuration menulis config/user_config.yaml (bukan database).

12. Non-Fungsional
Jalan lokal: streamlit run app/Home.py; tanpa internet saat runtime
Klaster terdekat tampil ≤ window_sec + refresh_sec setelah paket masuk
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
Deskripsi (T2)	Tidak ada label kualitas; deskripsi posisi median terhadap ambang ITU dan metrik_melewati_ambang (K0: delay di dalam batas baik; K1 dan Noise: delay > 400 ms)		

Tambahan: FAI s1 = 1.33, s2 = 20.45, s10 = 52.00; pagar delay K0 = 57.69 / 158.25; whisker K0 = 89.40 / 135.20; noise ratio = 30%. (T2: uji label Tabel 2.3 dihapus; digantikan uji deskripsi posisi ITU, penugasan centroid, batas aturan (c), di luar klaster, nomor klaster 1..k, kolom CSV, dan perbandingan K-Means.)

Integrasi:

Pelatihan pada data sintetis → semua artefak terbentuk dan bisa di-load; penugasan Euclidean konsisten dengan label pelatihan
Replay pcap SIP/RTP → CSV jendela terisi, dashboard terupdate, Stop menulis ringkasan

Target model (dilaporkan, bukan test gagal): Silhouette > 0,50 dan Noise Ratio ≤ 20%. Model 20261027_005150: Silhouette 0,8767, noise 13,36%.

14. Milestone (kerjakan berurutan, berhenti untuk review di tiap ⛔)
M0 Scaffold folder, requirements, config loader, pytest jalan
M1 Loader + filter VoIP + laporan EDA (jumlah flow, distribusi 4 fitur) ⛔ laporkan jumlah flow VoIP
M2 Features, cleaning, FAI, profil, pelabelan + test Lampiran 5 lulus ⛔
M3 Training + grid search + evaluasi + artefak + CLI ⛔ laporkan skor
M4 Inference engine + mode replay PCAP (tanpa UI)
M5 UI 4 halaman memakai mode replay ⛔
M6 Live capture pyshark/scapy + penanganan izin
M7 Rekomendasi, riwayat, ekspor CSV/PDF
M8 Latih ulang dari UI + versioning + Terapkan Ulang Pelabelan (T2: diganti Perbarui Deskripsi Klaster)
M9 README, screenshot tiap halaman (untuk Bab/Lampiran)
15. Risiko Teknis
Catatan T2 (27 Okt 2026): butir risiko tentang label QoS (Varian C, ambang loss 3 level) berlaku untuk riwayat pelabelan lama; aplikasi kini tidak melabeli kualitas.
Hasil Varian C (19 Okt 2026, model 20261019_175647: 8 klaster + noise 13,77%, Silhouette 0,8866): tidak ada klaster "QoS Baik"; semua klaster berlabel "QoS Buruk" dan noise "Anomali". Klaster terbesar (123 flow, delay ≈ 10 ms) berlabel Buruk hanya karena PLR 0,1053% sedikit di atas ambang 0,1%. Mode window menghasilkan FAI ≈ 0,13% pada RTP tanpa loss karena efek batas jendela (selisih satu paket antara fwd dan bwd di tepi jendela), sehingga cumulative dijadikan default (19 Okt 2026). Resolusi FAI vs ambang 0,1%: FAI diskret (selisih satu paket pada flow berisi ~6000 paket = 0,0167%, dan pada flow < 1000 paket satu paket selisih sudah > 0,1%), sehingga ambang 0,1% nyaris berarti "tidak ada selisih paket sama sekali" dan sangat sensitif terhadap satu-dua paket. P99,9 jitter pada klaster kecil (n < 1000) berinterpolasi ke ujung atas data sehingga ≈ nilai maksimum klaster (rasio P99,9/maks 0,90–1,00 pada model ini), bukan persentil yang stabil. Ambang tidak diubah. Dataset bukan dirancang khusus untuk VoIP; jumlah flow lolos filter kecil → dicek di M1 (19 Okt 2026): 3.566 flow berlabel VOIP di Darknet.CSV, 247 flow setelah exclude Tor dan fwd/bwd ≥ 5 (193 UDP, 54 TCP; NonVPN 153, VPN 94), sehingga min_flows_warning diturunkan ke 200 dan grid min_cluster_size dibatasi ≤ 25
Semantik fitur: (diubah 19 Okt 2026: delay kini memakai Flow IAT Mean karena Flow Duration menempel ke batas 120 detik dan jauh di atas ambang ITU) Flow Duration CIC = durasi seluruh flow, sedangkan live dihitung per jendela → distribusi bisa bergeser; bandingkan window vs cumulative saat M4
Ambang loss 3 level (diubah 22 Okt 2026, disetujui dosen): ≤1% baik, 1–3% cukup, >3% buruk; sumber ITU-T G.1010 (conversational voice, PLR < 3%) + panduan Cisco VoIP (≤ 1%). Alasan: ambang Y.1541 0,1% terlalu ketat terhadap estimasi FAI yang diskret (lihat catatan 19 Okt 2026) sehingga tidak ada klaster "QoS Baik". Hasil relabel model 20261019_204042 → 20261022_120925 (tanpa refit): QoS Baik 49,8% (klaster 5, 123 flow), Cukup 0%, Buruk 36,4%, Anomali 13,8%; hanya klaster 5 (→ Baik) dan 7 (→ "Jitter Melebihi Batas") berubah label. "QoS Cukup" tetap kosong pada model ini karena klaster dengan loss level 2 (klaster 7) juga melanggar jitter. Rincian skenario S0–S3: reports/experiments/labeling_scenarios.md. Lampiran 5: label K0 kini "QoS Baik" (PLR 0,99% ≤ 1%).
min_cluster_size kecil pada data besar → klaster sangat banyak; andalkan grid search
Silhouette O(n²) → wajib sample_size
Live capture butuh hak Administrator/root dan tshark/Npcap
M10 (T2, 27 Okt 2026) Klasterisasi murni: N1 training/artefak, N2 inferensi Euclidean + CSV baru, N3 UI, N4 dokumentasi dan tangkapan layar
