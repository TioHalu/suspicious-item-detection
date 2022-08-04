import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import seaborn as sns
# sns.set()

st.title("Pendeteksi Item Mencurigakan")

data = pd.read_csv('./DatasetAlpha0.3.csv')
st.subheader("Data training")
data

st.subheader("Visualisasi data training")
fig, ax = plt.subplots()
for PCU, d in data.groupby('Potential_Criminal_Usage'):
    ax.scatter(d['Harga(Rp)'], d['Weight(g)'], label=PCU)
st.pyplot(fig)

x_train = np.array(data[['Harga(Rp)','Weight(g)']])
y_train = np.array(data['Potential_Criminal_Usage'])

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

from sklearn.neighbors import KNeighborsClassifier

# test = {1,0,1,0,1,1,1,0,1}
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)


# data baru predikti
st.title("contoh prediksi satu data")
"berat 850 harga 1800000"
berat = 850
harga = 1800000
x_new = np.array([[harga, berat]]).reshape(1, -1)
y_new = model.predict(x_new)
test = lb.inverse_transform(y_new)
test

# prediksi
st.subheader("contoh prediksi beberapa item")
"table struk belanja alexwilliams"
datanew = pd.read_csv('./Receipts/Receipts_AlexWilliams.csv')
datanew
x_test = np.array(datanew[['Price(IDR)','Weight']])
y_pred = model.predict(x_test)
trans = lb.inverse_transform(y_pred)
lenght_pred = len(trans)
    
def probability(criminal):
    jumlah = 0
    persentase = 0
    for i in range(lenght_pred):
        if trans[i] == criminal:
            jumlah = jumlah + 1
    persentase = (jumlah/lenght_pred)*100
    criminal 
    persentase , "%"

probability("Kidnap")
probability("Remove Evidence")
probability("Narcotics")
probability("Craft Explosive")