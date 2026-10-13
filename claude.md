Baca PRD.md sebelum mengerjakan apa pun. PRD adalah sumber kebenaran; jika ambigu, tanya dulu.
Kerjakan per milestone (PRD §14). Berhenti di setiap ⛔ dan ringkas hasilnya.
Pakai paket hdbscan, bukan sklearn.cluster.HDBSCAN.
Satuan internal: delay & jitter dalam ms, loss dalam %. Data CIC dalam µs → bagi 1000.
Semua ambang dan parameter dari config, jangan hardcode.
Setiap fungsi perhitungan wajib punya unit test; test Lampiran 5 (PRD §13) harus lulus.
Jalankan pytest setelah setiap perubahan logika.
Jangan commit data/raw, *.pcap, .pcapng, artifacts// yang besar.
Perintah: streamlit run app/Home.py | python scripts/train.py --config config/default_config.yaml