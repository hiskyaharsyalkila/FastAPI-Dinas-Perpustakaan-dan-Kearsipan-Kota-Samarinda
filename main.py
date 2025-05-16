from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Inisialisasi FastAPI
app = FastAPI(
    title="API Prediksi Klaster Pemustaka",
    description="API ini memprediksi perilaku peminjaman buku berdasarkan data demografi dan keterlambatan",
    version="1.0.0"
)

# Definisi struktur input
class InputData(BaseModel):
    jumlah_hari_telat: float
    nomor_klass: int
    jenis_kelamin: int  # 0 = Laki-Laki, 1 = Perempuan
    kelompok_umur: int
    status_keterlambatan: int  # 0 = Tepat Waktu, 1 = Terlambat

# Fungsi interpretasi hasil klaster (bisa kamu sesuaikan)
def interpret_result(cluster: int) -> str:
    if cluster == 0:
        return "Pemustaka tepat waktu dan dari kelompok umur muda"
    elif cluster == 1:
        return "Pemustaka cukup sering telat dan aktif meminjam"
    elif cluster == 2:
        return "Kelompok pemustaka rawan keterlambatan"
    elif cluster == 3:
        return "Pemustaka dewasa dengan preferensi buku tertentu"
    elif cluster == 4:
        return "Pemustaka anak-anak atau pelajar disiplin"
    elif cluster == 5:
        return "Kelompok pemustaka pasif namun tepat waktu"
    elif cluster == 6:
        return "Pemustaka sering telat meminjam buku tertentu"
    elif cluster == 7:
        return "Pemustaka senior dengan keterlambatan bervariasi"
    elif cluster == 8:
        return "Karakteristik pemustaka campuran"
    else:
        return "Belum dapat diidentifikasi"

# Endpoint utama untuk prediksi
@app.post("/predict_cluster")
def predict_cluster(data: InputData):
    input_array = np.array([[data.jumlah_hari_telat,
                             data.nomor_klass,
                             data.jenis_kelamin,
                             data.kelompok_umur,
                             data.status_keterlambatan]])

    # Scaling input
    scaled_input = scaler.transform(input_array)

    # Prediksi
    cluster = model.predict(scaled_input)[0]
    interpretasi = interpret_result(cluster)

    return {
        "cluster": int(cluster),
        "interpretasi": interpretasi
    }