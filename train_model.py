import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('dataset.csv')

print("Kolom:", df.columns)

df['date'] = range(1, len(df)+1)

X = df[['date']]
y = df['Price']  

y = y.astype(str).str.replace(',', '')
y = y.astype(float)

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))

print("Model berhasil dibuat!")