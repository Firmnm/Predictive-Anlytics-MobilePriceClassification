# -*- coding: utf-8 -*-
"""Mobile_Price_Classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FRVwk5qX56qFKRv_PAqSJQ6XtmXVKXL8

# Import Library & Setup

*   Import library
*   Setup kredensial Kaggle API
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.filterwarnings('ignore')

# Upload file kaggle.json yang berisi API Key
from google.colab import files
files.upload()

"""Kode ini untuk upload file `kaggle.json` dari perangkat lokal ke Colab, yang diperlukan untuk akses API Kaggle.

# Data Loading
"""

# Simpan kaggle.json ke direktori yang sesuai
os.makedirs('/root/.kaggle', exist_ok=True)
!cp kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

"""Kode ini menyiapkan konfigurasi untuk mengakses Kaggle API dengan menyalin file `kaggle.json` (API token) ke direktori yang diperlukan dan mengatur izin aksesnya.

*   Unduh dataset dari Kaggle
*   Load dataset train.csv dan test.csv
*   Tampilkan informasi awal dan statistik
"""

#Download dataset dari Kaggle
!kaggle datasets download -d iabhishekofficial/mobile-price-classification

# Ekstrak file zip
!unzip -q mobile-price-classification.zip -d mobile_price_dataset

"""Download dataset dari Kaggle"""

# Load data
train_df = pd.read_csv('/content/mobile_price_dataset/train.csv')
test_df = pd.read_csv('/content/mobile_price_dataset/test.csv')

""" Dataset proyek ini terdiri dari dataset Train untuk melatih model dan dataset Test untuk menguji model."""

train_df.head()

"""Menampilkan 5 baris data pada dataset Train"""

train_df.info()

"""Output ini menunjukkan DataFrame dengan 2000 entri dan 21 kolom, tanpa nilai yang hilang. Kolom `price_range` adalah target, sementara kolom lainnya berisi fitur-fitur ponsel seperti daya baterai, ukuran layar, RAM, dll.

"""

train_df.describe().T

"""Output ini merangkum statistik deskriptif dataset, seperti jumlah data (count), rata-rata (mean), standar deviasi (std), nilai minimum, maksimum, dan kuartil untuk setiap fitur.

Pada kolom px_height dan sc_w terdapat nilai 0, yang mengindikasikan nilai tidak valid. Tinggi dan lebar layar seharusnya memiliki nilai lebih besar dari 0, sehingga perlu dilakukan penanganan lebih lanjut pada tahap preprocessing data.
"""

test_df.head()

"""Melihat 5 baris data pada dataset Test"""

test_df.info()

test_df.describe().T

"""Pada dataset Test juga, pada kolom px_height dan sc_w terdapat nilai 0, yang mengindikasikan nilai tidak valid. Tinggi dan lebar layar seharusnya memiliki nilai lebih besar dari 0, sehingga perlu dilakukan penanganan lebih lanjut pada tahap preprocessing data.

# Exploratory Data Analysis (EDA)

* Distribusi target (kelas harga)

* Korelasi antar fitur

* Visualisasi fitur penting terhadap target

* Pengecekan Nilai missing value (null)

* Cek nilai tidak Valid
"""

# Visualisasi distribusi target
sns.countplot(data=train_df, x='price_range')
plt.title("Distribusi Kelas Harga Smartphone")
plt.show()

"""Dari hasil distribusi label di atas, setiap label terdistribusi secara merata di antara empat kategori harga smartphone, yaitu 'murah', 'sedang', 'mahal', dan 'sangat mahal'."""

# Korelasi antar fitur
plt.figure(figsize=(12, 10))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.title("Matriks Korelasi")
plt.show()

"""Dari hasil heatmap di atas, dapat dilihat bahwa terdapat beberapa fitur yang memiliki **korelasi yang signifikan** dengan variabel target `price_range`. Misalnya, fitur-fitur seperti `ram`, `fc`, `pc`, `three_g`,dan `four_g` menunjukkan korelasi yang cukup kuat dengan harga smartphone, yang mengindikasikan bahwa faktor-faktor ini memiliki pengaruh besar terhadap penentuan harga. Sebaliknya, beberapa fitur lainnya seperti `sc_w` (screen width) dan `px_height` (pixel height) memiliki korelasi yang lebih lemah terhadap target, yang berarti mereka mungkin tidak terlalu berkontribusi dalam memprediksi harga smartphone.

"""

# Contoh visualisasi fitur penting
plt.figure(figsize=(8,6))
sns.boxplot(x='price_range', y='ram', data=train_df)
plt.title("Distribusi RAM (MB) terhadap Price Range")
plt.show()

"""Boxplot diatas menunjukkan bahwa RAM meningkat seiring kenaikan kategori price_range, menunjukkan hubungan positif antara RAM dan harga."""

# Contoh visualisasi fitur penting
plt.figure(figsize=(8,6))
sns.boxplot(x='price_range', y='fc', data=train_df)
plt.title("Distribusi Front Camera (Mega Pixel) terhadap Price Range")
plt.show()

"""Dari boxplot di atas, distribusi kamera depan (MP) tampak stabil di semua kategori **price\_range**, tanpa pola peningkatan yang jelas.

"""

train_df.isna().sum()

"""Terlihat tidak terdapat nilai NaN atau nilai kosong."""

# Cek nilai 0 yang mencurigakan (bisa jadi missing value)
print('Jumlah nilai 0 pada fitur px_height = ', len(train_df[train_df.px_height == 0]))
print('Jumlah nilai 0 pada fitur sc_w = ', len(train_df[train_df.sc_w == 0]))

"""Pada dataset Train, kolom px_height terdapat 2 nilai 0, dan pada kolom sc_w terdapat 180 nilai 0. Nilai 0 pada kedua kolom ini menunjukkan data yang tidak valid, karena seharusnya tinggi dan lebar layar memiliki nilai lebih besar dari 0. Oleh karena itu, nilai-nilai 0 ini perlu ditangani lebih lanjut pada tahap preprocessing data."""

# Cek nilai 0 yang mencurigakan (bisa jadi missing value)
print('Jumlah nilai 0 pada fitur px_height = ', len(test_df[test_df.px_height == 0]))
print('Jumlah nilai 0 pada fitur sc_w = ', len(test_df[test_df.sc_w == 0]))

"""Pada dataset Test, kolom px_height terdapat 2 nilai 0, dan pada kolom sc_w terdapat 112 nilai 0. Nilai 0 pada kedua kolom ini menunjukkan data yang tidak valid, karena seharusnya tinggi dan lebar layar memiliki nilai lebih besar dari 0. Oleh karena itu, nilai-nilai 0 ini perlu ditangani lebih lanjut pada tahap preprocessing data.

# Data Preprocessing

* Penanganan nilai 0 pada fitur tertentu

* Split data train dan validasi

* Standarisasi fitur numerik
"""

# Ganti nilai 0 dengan nilai rata-rata
train_df['sc_w'] = train_df['sc_w'].astype(float)
train_df['px_height'] = train_df['px_height'].astype(float)
test_df['sc_w'] = test_df['sc_w'].astype(float)
test_df['px_height'] = test_df['px_height'].astype(float)

train_df.loc[train_df['sc_w'] == 0, 'sc_w'] = train_df['sc_w'].mean()
train_df.loc[train_df['px_height'] == 0, 'px_height'] = train_df['px_height'].mean()
test_df.loc[test_df['sc_w'] == 0, 'sc_w'] = test_df['sc_w'].mean()
test_df.loc[test_df['px_height'] == 0, 'px_height'] = test_df['px_height'].mean()

"""Kode di atas mengganti nilai 0 pada kolom sc_w dan px_height dengan rata-rata masing-masing kolom setelah mengubah tipe data menjadi float."""

test_df.drop('id', axis=1, inplace=True)

"""Pada dataset Test, menghapus kolom id karena tidak diperlukan

# Data Preparation
"""

# Pisahkan fitur dan label
X = train_df.drop('price_range', axis=1)
y = train_df['price_range']

"""Kode di atas memisahkan label pada dataset train untuk diproses pada tahap selanjutnya.

Standarisasi
"""

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

"""Kode di atas melakukan standarisasi data menggunakan **StandardScaler**. Fitur **X** pada dataset **Train** diubah dengan `fit_transform()` untuk menyesuaikan rata-rata dan standar deviasi, sedangkan dataset  **Test** diubah dengan `transform()` berdasarkan parameter yang dihitung dari **train**. Tujuannya adalah agar skala fitur seragam, sehingga model dapat bekerja lebih optimal.

**Split Dataset Train**
"""

# Split Data (Train/Test)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

"""Kode di atas membagi dataset Train menjadi set pelatihan (**X\_train**, **y\_train**) dan set validasi (**X\_val**, **y\_val**) dengan proporsi 80% untuk pelatihan dan 20% untuk validasi. Pembagian ini dilakukan menggunakan **train\_test\_split()** untuk menguji model pada data yang belum pernah dilihat sebelumnya.

# Model Training & Hyperparameter Tuning

* Definisi dan pelatihan beberapa model (Decision Tree, Random Forest, SVM, LogReg)

* Grid Search untuk tuning hyperparameter

* Pilih model terbaik berdasarkan akurasi
"""

# Modeling - Beberapa Model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC()
}

"""Kode di atas mendefinisikan beberapa model klasifikasi dengan parameter sebagai berikut:

* Logistic Regression: max_iter=1000 untuk memastikan konvergensi model dalam 1000 iterasi.

* Decision Tree: Tanpa parameter tambahan, menggunakan pengaturan default.

* Random Forest: n_estimators=100 untuk menentukan jumlah pohon dalam hutan.

* SVM: Tanpa parameter tambahan, menggunakan pengaturan default.
"""

# Hyperparameter Tuning (Grid Search CV)
param_grids = {
    "Decision Tree": {
        'max_depth': [3, 5, 10, None],
        'criterion': ['gini', 'entropy']
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}

"""Kode di atas mendefinisikan grid parameter untuk **Grid Search CV** dengan tujuan mencari kombinasi hyperparameter terbaik:

* **Decision Tree**:

  * `max_depth`: Menentukan kedalaman maksimum pohon.
  * `criterion`: Menentukan kriteria pemisahan (gini atau entropy).
* **Random Forest**:

  * `n_estimators`: Jumlah pohon dalam hutan.
  * `max_depth`: Kedalaman maksimum pohon.
* **SVM**:

  * `C`: Parameter regularisasi yang mengontrol kesalahan model.
  * `kernel`: Jenis fungsi kernel yang digunakan (linear atau rbf).

"""

best_models = {}

# Tuning tiap model
for name in param_grids:
    print(f"Tuning {name}...")
    grid = GridSearchCV(models[name], param_grids[name], cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print(f"Best Params for {name}: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.4f}\n")

    # Menyimpan model terbaik
    best_models[name] = grid.best_estimator_

# Menampilkan hanya parameter yang di-tuning
print("\nBest Tuned Parameters for Each Model:")
for name, model in best_models.items():
    print(f"{name}:")
    for param in param_grids[name]:
        print(f"  {param}: {model.get_params()[param]}")
    print()

"""Kode di atas melakukan **hyperparameter tuning** untuk setiap model menggunakan **GridSearchCV**. Tujuannya adalah untuk menemukan kombinasi hyperparameter terbaik berdasarkan **cross-validation (cv=5)** dan **skor akurasi**. Setelah tuning, parameter terbaik dan skor terbaik untuk setiap model ditampilkan, dan model terbaik disimpan dalam dictionary `best_models`.

Didapatkan bahwa hasil parameter terbaiknya tiap model:

* Decision Tree:
  
        max_depth: None

        criterion: entropy

* Random Forest:

        n_estimators: 200

        max_depth: None

* SVM:

        C: 10

        kernel: linear

"""

# Melatih Model Lain tanpa Hyperparameter Tuning (Logistic Regression)
best_models["Logistic Regression"] = models["Logistic Regression"].fit(X_train, y_train)

"""Kode di atas melatih model **Logistic Regression** tanpa melakukan hyperparameter tuning, menggunakan data pelatihan **X\_train** dan **y\_train**, lalu menyimpan model yang telah dilatih ke dalam dictionary `best_models`.

# Model Evaluation

* Evaluasi model pada validation set

* Confusion Matrix & Classification Report

* Visualisasi akurasi antar model

* Cross-validation
"""

# Evaluasi Model
final_results = {}

for name, model in best_models.items():
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    final_results[name] = acc

    print(f"=== {name} ===")
    print(f"Akurasi: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print()

"""Hasil evaluasi menunjukkan bahwa **Logistic Regression** memiliki akurasi tertinggi (97.75%), diikuti oleh **SVM** (97.25%) dan **Random Forest** (89%). **Decision Tree** memiliki akurasi 84%. Metrik lainnya, seperti precision, recall, dan f1-score, juga menunjukkan performa terbaik pada Logistic Regression dan SVM.

"""

# Visualisasi Hasil Evaluasi Akurasi Model
plt.figure(figsize=(8, 5))
plt.bar(final_results.keys(), final_results.values())
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Perbandingan Akurasi Model Setelah Tuning')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

"""Dari hasil pelatihan model di atas, dapat disimpulkan bahwa model Logistic Regression menunjukkan kinerja yang cukup baik dalam memprediksi harga smartphone berdasarkan berbagai fitur yang ada. Meskipun tidak melalui proses tuning hyperparameter yang mendalam, model ini tetap mampu memberikan akurasi yang memadai dibandingkan dengan model lainnya. Hal ini menunjukkan bahwa Logistic Regression bisa menjadi pilihan yang efektif, terutama ketika kecepatan eksekusi dan interpretasi hasil model menjadi faktor yang penting dalam pengambilan keputusan."""

# Pilih Model Terbaik dan Prediksi Data Test
best_model_name = max(final_results, key=final_results.get)
final_model = best_models[best_model_name]
print(f"Model Terbaik: {best_model_name}")

"""Kode ini memilih model dengan performa terbaik berdasarkan hasil akhir (`final_results`) dan kemudian menggunakan model tersebut untuk melakukan prediksi pada data uji. Nama model terbaik dicetak untuk referensi.

# Prediction & Submission

* Prediksi menggunakan model terbaik

* Distribusi hasil prediksi

* Simpan hasil ke CSV untuk submission
"""

# Prediksi pada data test
test_predictions = final_model.predict(test_scaled)

"""Kode ini melakukan prediksi menggunakan model terbaik (`final_model`) pada dataset Test yang sudah distandarisasi (`test_scaled`). Hasil prediksi disimpan dalam variabel `test_predictions`."""

# Mapping
label_mapping = {
    0: "murah",
    1: "sedang",
    2: "mahal",
    3: "sangat mahal"
}

# Lihat distribusi kelas yang diprediksi pada data test
unique_classes, counts = np.unique(test_predictions, return_counts=True)

# Visualisasi
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
plt.bar([label_mapping[cls] for cls in unique_classes], counts, color=colors[:len(unique_classes)])
plt.xlabel('Kelas')
plt.ylabel('Jumlah Sampel')
plt.title('Distribusi Kelas Prediksi pada Data Test')
plt.show()

for cls, count in zip(unique_classes, counts):
    print(f"\nKelas '{label_mapping[cls]}': {count} sampel")

"""**Kesimpulan**

Distribusi kelas prediksi pada data test menunjukkan hasil yang relatif seimbang di antara kelas harga yang ada:

* Murah: 254 sampel

* Sedang: 232 sampel

* Mahal: 255 sampel

* Sangat Mahal: 259 sampel

Model berhasil mengklasifikasikan sampel hampir secara merata, dengan sedikit perbedaan jumlah sampel per kelas, yang menunjukkan tidak ada bias signifikan antar kelas. Hal ini menandakan bahwa model bekerja dengan baik dalam membedakan rentang harga pada data yang ada.

"""

# Menyimpan hasil prediksi ke CSV
submission = pd.DataFrame({
    "predicted_price_range": test_predictions
})
submission.to_csv("submission_tuned.csv", index=False)
print("Prediksi telah disimpan dalam 'submission_tuned.csv'")

"""Kode di atas menyimpan hasil prediksi ke dalam sebuah file CSV. Data hasil prediksi yang disimpan berupa kolom "predicted\_price\_range", yang berisi nilai prediksi untuk setiap sampel pada data uji. File CSV tersebut disimpan dengan nama "submission\_tuned.csv", tanpa menyertakan indeks baris.

"""