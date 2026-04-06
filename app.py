from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except:
    model = None

@app.route('/')
def home():
    return render_template('index.html', hasil=None, gambar=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bulan = int(request.form['bulan'])

        df = pd.read_csv('dataset.csv')

        # bersihin data
        df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
        y = df['Price']

        x = np.arange(1, len(y)+1)

        if model:
            prediksi = model.predict(np.array([[bulan]]))[0]
            x_pred = np.arange(1, bulan+1)
            y_pred = model.predict(x_pred.reshape(-1,1))
        else:
            prediksi = 0
            x_pred = x
            y_pred = y

        # bikin folder static kalau belum ada
        os.makedirs('static', exist_ok=True)

        # plot grafik
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

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
