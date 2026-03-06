# Machine Learning Preprocessing - Cervical Cancer Risk Factors

## Deskripsi Proyek
Project ini merupakan implementasi tahapan preprocessing dalam Machine Learning menggunakan dataset **Risk Factors for Cervical Cancer** dari UCI Machine Learning Repository.

Tahapan preprocessing yang dilakukan meliputi:
- Penanganan **Missing Value**
- **Normalisasi data menggunakan MinMaxScaler**
- **Seleksi fitur menggunakan ANOVA**
- Penanganan **Imbalanced Data menggunakan SMOTE-ENN**

Tujuan dari project ini adalah mempersiapkan dataset agar lebih optimal untuk proses pelatihan model Machine Learning.
---

# Dataset

Dataset yang digunakan adalah **Risk Factors for Cervical Cancer Dataset** dari UCI Machine Learning Repository.

Link dataset:
https://archive.ics.uci.edu/ml/datasets/risk+factors+cervical+cancer

Dataset ini berisi berbagai faktor risiko yang berkaitan dengan diagnosis kanker serviks.

Contoh fitur dalam dataset:

- Age
- Number of sexual partners
- Num of pregnancies
- Smokes
- Hormonal Contraceptives
- STDs
- Dx:HPV
- Dx:Cancer
- Biopsy (Target)

Target yang digunakan dalam analisis adalah **Biopsy**.

---

# Tahapan Preprocessing

## 1. Handling Missing Value

Dataset memiliki beberapa nilai kosong yang ditandai dengan simbol `?`.

Langkah penanganan missing value:

- Membaca dataset menggunakan **pandas**
- Mengubah simbol `?` menjadi **NaN**
- Mengganti nilai kosong menggunakan:
  - **Median** untuk fitur numerik
  - **Mode** untuk fitur kategorikal
- Menyimpan dataset yang telah dibersihkan ke file baru

Output file:
```
cervical_cleaned.csv
```

---

## 2. Normalisasi Data (MinMaxScaler)

Normalisasi dilakukan untuk menyetarakan skala fitur numerik agar berada pada rentang **0 – 1**.

Library yang digunakan:

```
sklearn.preprocessing.MinMaxScaler
```

Tujuan normalisasi:

- Menghindari bias pada fitur dengan skala besar
- Mempercepat proses pembelajaran model
- Meningkatkan stabilitas algoritma Machine Learning

Output file:

```
risk_factors_cervical_cancer_scaled.csv
```

---

## 3. Seleksi Fitur Menggunakan ANOVA

Seleksi fitur dilakukan menggunakan metode **ANOVA F-Test** untuk menentukan fitur yang paling berpengaruh terhadap target **Biopsy**.

Library yang digunakan:

```
sklearn.feature_selection
```

Metode yang digunakan:

- `SelectKBest`
- `f_classif`

Fitur dengan nilai **F-Score tertinggi** dianggap memiliki pengaruh terbesar terhadap target.

Contoh fitur dengan kontribusi tinggi:

- Schiller
- Hinselmann
- Citology
- Dx:HPV
- Dx:Cancer
- Dx
- STDs:genital herpes

---

## 4. Penanganan Imbalanced Data (SMOTE-ENN)

Dataset memiliki ketidakseimbangan kelas pada target **Biopsy**.

Distribusi awal:

- Kelas mayoritas ≈ 800 data
- Kelas minoritas ≈ 50 data

Metode yang digunakan:

### SMOTE
Synthetic Minority Oversampling Technique

Menambah data sintetis pada kelas minoritas.

### ENN
Edited Nearest Neighbor

Menghapus data noisy setelah proses oversampling.

Library yang digunakan:

```
imblearn.combine.SMOTEENN
```

Hasil setelah resampling:

- Kedua kelas menjadi hampir seimbang
- Sekitar **750 – 780 data pada masing-masing kelas**
---

# Alur Preprocessing

Tahapan preprocessing yang dilakukan dalam project ini:

```
Dataset
   ↓
Check Missing Value
   ↓
Imputasi Median / Mode
   ↓
Normalisasi Data (MinMaxScaler)
   ↓
Seleksi Fitur (ANOVA)
   ↓
Penanganan Imbalanced Data (SMOTE-ENN)
   ↓
Dataset Siap untuk Training Model
```

---

# Kesimpulan

Berdasarkan hasil praktikum Machine Learning:

1. Missing value berhasil ditangani menggunakan metode **imputasi median dan mode**.
2. Normalisasi menggunakan **MinMaxScaler** berhasil menyetarakan skala seluruh fitur ke rentang **0–1**.
3. Seleksi fitur menggunakan **ANOVA** menunjukkan beberapa fitur yang memiliki pengaruh signifikan terhadap target **Biopsy**.
4. Metode **SMOTE-ENN** berhasil menyeimbangkan distribusi kelas pada dataset sehingga lebih optimal untuk pelatihan model Machine Learning.
