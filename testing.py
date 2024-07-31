import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

def prediksi_tj():
    start_time = time.time()

    # Muat dataset
    tj_data = pd.read_csv('/kaggle/input/indathon-round1-2024/training_jumlah_penumpang_tj.csv', sep=';')  # Data penumpang TJ dari 2015-2023
    other_data = pd.read_csv('/kaggle/input/gabungan/data_gabungan.csv', sep=';')  # Data armada TJ, penumpang dan perjalanan LRT & MRT 2023-2024

    # Memisahkan data berdasarkan tahun
    tj_train = tj_data[tj_data['tahun'] < 2024].copy()  # Buat salinan untuk menghindari SettingWithCopyWarning
    tj_test = tj_data[tj_data['tahun'] == 2023].copy()  # Salinan eksplisit
    other_train = other_data[other_data['tahun'] == 2023].copy()  # Salinan eksplisit
    other_future = other_data[(other_data['tahun'] == 2024) & (other_data['bulan'] <= 5)].copy()  # Salinan eksplisit

    # Pastikan data tidak mengandung nilai NaN
    tj_train.fillna(0, inplace=True)
    tj_test.fillna(0, inplace=True)
    other_train.fillna(0, inplace=True)
    other_future.fillna(0, inplace=True)

    # Menyiapkan fitur dan target
    # Karena dataset berbeda skala waktu, kita asumsikan fitur lain hanya dipakai sebagai variabel penjelas
    X_train = other_train[['jumlah_armada_tj', 'jumlah_penumpang_lrt', 'jumlah_penumpang_mrt', 'jumlah_perjalanan_lrt', 'jumlah_perjalanan_mrt']]
    y_train = tj_train['jumlah_penumpang'][-len(other_train):]  # Cocokkan panjang data
    X_test = other_train[['jumlah_armada_tj', 'jumlah_penumpang_lrt', 'jumlah_penumpang_mrt', 'jumlah_perjalanan_lrt', 'jumlah_perjalanan_mrt']]
    y_test = tj_test['jumlah_penumpang']
    X_future = other_future[['jumlah_armada_tj', 'jumlah_penumpang_lrt', 'jumlah_penumpang_mrt', 'jumlah_perjalanan_lrt', 'jumlah_perjalanan_mrt']]

    # Melatih model regresi linear
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Prediksi dengan model regresi linear
    y_pred = linear_model.predict(X_test)

    # Evaluasi model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse}')
    print(f'R-squared: {r2}')

    # Plot hasil prediksi vs data aktual
    plt.figure(figsize=(10, 6))
    plt.plot(tj_test.index, y_test, label='Data Aktual', marker='o')
    plt.plot(tj_test.index, y_pred, label='Prediksi', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Jumlah Penumpang TJ')
    plt.title('Prediksi Jumlah Penumpang TJ Tahun 2023')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prediksi untuk bulan Januari hingga Mei 2024
    y_pred_future = linear_model.predict(X_future)

    # Prediksi trend sederhana untuk bulan Juni 2024
    # Hitung rata-rata perubahan bulanan dari bulan-bulan sebelumnya
    avg_monthly_change = np.mean(np.diff(y_pred_future))
    june_prediction = y_pred_future[-1] + avg_monthly_change

    # Hasil prediksi termasuk bulan Juni
    # Menggunakan `other_future` untuk mengambil nilai 'bulan' yang ada
    bulan_future = list(other_future['bulan'])  # Pastikan kolom 'bulan' ada di sini
    bulan_future.append(6)  # Tambahkan bulan Juni

    prediksi_2024 = pd.DataFrame({
        'tahun': 2024,
        'bulan': bulan_future,
        'prediksi_jumlah_penumpang_tj': list(y_pred_future) + [june_prediction]
    })

    print(prediksi_2024)

    # Hitung waktu pemrosesan
    processing_time = time.time() - start_time
    print(f"Total execution time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    prediksi_tj()
