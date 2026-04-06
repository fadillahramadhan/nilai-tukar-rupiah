from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import matplotlib

matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)

try:
    model = pickle.load(open('model.pkl', 'rb'))
except:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    bulan = int(request.form['bulan'])

    df = pd.read_csv('dataset.csv')
    y = df['Price'].astype(str).str.replace(',', '').astype(float)

    x = np.arange(1, len(y)+1)

    if model:
        prediksi = model.predict(np.array([[bulan]]))[0]
        x_pred = np.arange(1, bulan+1)
        y_pred = model.predict(x_pred.reshape(-1,1))
    else:
        prediksi = 0
        x_pred = x
        y_pred = y

    plt.figure()
    plt.plot(x, y, label='Data Asli')
    plt.plot(x_pred, y_pred, linestyle='--', label='Prediksi')
    plt.scatter(bulan, prediksi, label='Titik Prediksi')

    plt.title("Grafik Kurs Rupiah")
    plt.xlabel("Bulan")
    plt.ylabel("Kurs")
    plt.legend()

    plt.savefig('static/grafik.png')
    plt.close()

    return render_template('index.html',
                           hasil=f"Prediksi Kurs: Rp {int(prediksi)}",
                           gambar='grafik.png')

import os

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
