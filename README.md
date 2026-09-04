# Submission 1: Proyek Pengembangan dan Pengoperasian Sistem Machine Learning

**Nama:** Ahmad Fatih Mu'Alim  
**Username Dicoding:** mualim-fth  

<br>

| Kolom | Deskripsi |
| :--- | :--- |
| **Dataset** | Menggunakan dataset **Telco Customer Churn** dari direktori lokal penyimpanan proyek (`data/`). Dataset ini berisi informasi profil pelanggan telekomunikasi, layanan yang digunakan, rincian tagihan, masa berlangganan (*tenure*), serta kolom target `Churn`. |
| **Masalah** | Perusahaan telekomunikasi sering kali mengalami kerugian finansial akibat tingginya tingkat perpindahan pelanggan (*churn rate*). Tantangannya adalah bagaimana mengidentifikasi pelanggan yang berpotensi melakukan *churn* secara akurat dan otomatis agar strategi retensi dapat segera diterapkan secara proaktif. |
| **Solusi machine learning** | Membangun sistem *machine learning pipeline* otomatis secara *end-to-end* menggunakan **TensorFlow Extended (TFX)** yang mencakup komponen ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Resolver, Evaluator, dan Pusher. Sistem ini mengotomatisasi seluruh siklus mulai dari validasi data, pencarian hyperparameter, hingga model serving. |
| **Metode pengolahan** | Menggunakan TensorFlow Transform (TFT) di dalam modul `transform.py` untuk melakukan konversi dan pembersihan fitur. Fitur kategorikal diolah menggunakan `compute_and_apply_vocabulary` (one-hot/integer encoding), sedangkan fitur numerikal (*tenure*, *MonthlyCharges*, *TotalCharges*) dinormalisasi menggunakan teknik *z-score scaling* (`scale_to_z_score`) untuk menjaga konsistensi antara data pelatihan dan data *serving*. |
| **Arsitektur model** | Menggunakan arsitektur *Neural Network* (Deep Learning) berbasis *Dense Layers* yang dikonfigurasi melalui Keras Tuner (`RandomSearch`). Arsitektur terdiri dari lapisan *Input* gabungan fitur numerikal dan kategorikal, lapisan *Concatenate*, lapisan *Dense* dengan fungsi aktivasi *ReLU* (berdasarkan pencarian *hyperparameter* optimal: 32/48 unit), lapisan *Dropout* untuk regularisasi, dan lapisan *output Dense* tunggal dengan fungsi aktivasi *Sigmoid* untuk klasifikasi biner. Model dikompilasi menggunakan *optimizer* Adam dan fungsi *loss* `binary_crossentropy`. |
| **Metrik evaluasi** | Evaluasi model menggunakan TensorFlow Model Analysis (TFMA) dengan metrik utama **Binary Accuracy** dan metrik pendukung **ExampleCount**, dengan ambang batas (*threshold*) minimal kelulusan sebesar **0.5 (50%)** untuk memastikan performa prediksi andal. |
| **Performa model** | Model berhasil dilatih dan dievaluasi dengan hasil yang memuaskan. Berdasarkan hasil eksekusi *pipeline* terakhir, model mencapai skor akurasi validasi (*val_accuracy*) terbaik sekitar **78.5% hingga 79.5%**, serta berhasil mendapatkan status *blessed* dari komponen Evaluator dan diekspor oleh Pusher ke direktori *serving model*. |
| **Opsi deployment** | Model di-*deploy* menggunakan kontainer Docker ke platform komputasi cloud **Railway** sehingga dapat diakses secara publik melalui HTTP *REST API endpoint* untuk melayani permintaan prediksi (*inference*) secara *real-time*. |
| **Web app** | [Tautan Web App / Endpoint Railway Anda](https://mualim-fth-pipeline-production.up.railway.app/v1/models/churn-model:predict) *(Contoh format tautan)* |
| **Monitoring** | Menggunakan **Prometheus** untuk mengumpulkan metrik performa sistem, latensi, dan ketersediaan *endpoint* secara kontinu, yang kemudian dihubungkan dan divisualisasikan melalui dasbor **Grafana** untuk memantau kesehatan operasional model di cloud. |