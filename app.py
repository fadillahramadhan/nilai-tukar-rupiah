from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    bulan = int(request.form['bulan'])

    # ambil data
    df = pd.read_csv('dataset.csv')
    y = df['Price'].astype(str).str.replace(',', '').astype(float)

    # buat index waktu
    x = np.arange(1, len(y)+1)

    # prediksi
    prediksi = model.predict(np.array([[bulan]]))[0]

    # prediksi garis sampai bulan input
    x_pred = np.arange(1, bulan+1)
    y_pred = model.predict(x_pred.reshape(-1,1))

    # plot grafik
    plt.figure()
    plt.plot(x, y, label='Data Asli')
    plt.plot(x_pred, y_pred, linestyle='--', label='Prediksi')
    plt.scatter(bulan, prediksi, label='Titik Prediksi')

    plt.title("Grafik Kurs Rupiah (Real + Prediksi)")
    plt.xlabel("Bulan")
    plt.ylabel("Kurs")
    plt.legend()

    plt.savefig('static/grafik.png')
    plt.close()

    return render_template('index.html',
                           hasil=f"Prediksi Kurs: Rp {int(prediksi)}",
                           gambar='grafik.png')

if __name__ == '__main__':
    app.run(debug=True)

import os

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))