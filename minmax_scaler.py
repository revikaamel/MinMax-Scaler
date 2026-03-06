import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Baca dataset, ganti '?' jadi NaN
df = pd.read_csv("risk_factors_cervical_cancer.csv", na_values=["?"])

# 2. Isi nilai NaN dengan median setiap kolom numerik
df = df.fillna(df.median(numeric_only=True))
print("Data setelah bersih (5 baris pertama):")
print(df.head())

# 3. Normalisasi pakai MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("\nData setelah normalisasi (5 baris pertama):")
print(df_scaled.head())

# 4. Simpan hasil ke file CSV baru
df_scaled.to_csv("risk_factors_cervical_cancer_scaled.csv", index=False)
print("\nHasil normalisasi disimpan ke: risk_factors_cervical_cancer_scaled.csv")

# 5. Visualisasi distribusi sebelum & sesudah scaling (contoh 5 kolom pertama)
for col in df.columns[:5]:
    plt.figure(figsize=(10, 4))

    # sebelum scaling
    plt.subplot(1, 2, 1)
    df[col].hist(bins=30)
    plt.title(f"Sebelum Scaling: {col}")

    # sesudah scaling
    plt.subplot(1, 2, 2)
    df_scaled[col].hist(bins=30)
    plt.title(f"Sesudah Scaling: {col}")

    plt.tight_layout()
    plt.savefig(f"hist_{col}.png")  # menyimpan per kolom ke PNG
    plt.close()

print("\nGrafik distribusi disimpan dengan nama hist_namakolom.png")
