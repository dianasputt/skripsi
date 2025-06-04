import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Prediksi Musim Jawa Timur",
    page_icon="⛅"
)

# Load model
model_311 = tf.keras.models.load_model('Data/model/model_311.h5', custom_objects={'mse': tf.keras.losses.mse})
model_303 = tf.keras.models.load_model('Data/model/model_303.h5', custom_objects={'mse': tf.keras.losses.mse})
model_349 = tf.keras.models.load_model('Data/model/model_349.h5', custom_objects={'mse': tf.keras.losses.mse})

# Load data
data_311 = pd.read_excel("Data/data/data_311.xlsx")
data_303 = pd.read_excel("Data/data/data_303.xlsx")
data_349 = pd.read_excel("Data/data/data_349.xlsx")

# Load scalers
scaler_x_311 = joblib.load('Data/scaler/scaler_x_311.pkl')
scaler_x_303 = joblib.load('Data/scaler/scaler_x_303.pkl')
scaler_x_349 = joblib.load('Data/scaler/scaler_x_349.pkl')

scaler_y_311 = joblib.load('Data/scaler/scaler_y_311.pkl')
scaler_y_303 = joblib.load('Data/scaler/scaler_y_303.pkl')
scaler_y_349 = joblib.load('Data/scaler/scaler_y_349.pkl')

def reset_prediksi():
    st.session_state["prediksi_ditekan"] = False

def pilih_topografi():
    dropdown = st.selectbox(
        "Pilih Topografi yang diinginkan:",
        ("Dataran Tinggi", "Dataran Rendah", "Pesisir"),
        on_change=reset_prediksi
    )
    
    if dropdown == 'Dataran Tinggi':
        model = model_311
        data = data_311
        scaler_x = scaler_x_311
        scaler_y = scaler_y_311
        zona = '311'
    elif dropdown == 'Dataran Rendah':
        model = model_303
        data = data_303
        scaler_x = scaler_x_303
        scaler_y = scaler_y_303
        zona = '303'
    elif dropdown == 'Pesisir':
        model = model_349
        data = data_349
        scaler_x = scaler_x_349
        scaler_y = scaler_y_349
        zona = '349'
    st.write(f'Jenis topografi **{dropdown}** telah dipilih.')
    return model, data, scaler_x, scaler_y, zona

def pilih_musim():
    musim = st.selectbox(
        "Pilih Musim yang Ingin Diprediksi:",
        ("Musim Kemarau", "Musim Hujan"),
        on_change=reset_prediksi
    )
    st.write(f"Musim yang dipilih: **{musim}**")
    return musim

def buat_sequence(data_x, look_back=36):
    X_seq = []
    for i in range(len(data_x) - look_back):
        X_seq.append(data_x[i:i+look_back])
    return np.array(X_seq)

def prediksi_recursive(model, scaler_y, X_test_seq, n_future=36):
    recursive_input = X_test_seq[-1].copy()  # shape (look_back, fitur)
    predictions = []
    for _ in range(n_future):
        input_reshaped = recursive_input.reshape(1, recursive_input.shape[0], recursive_input.shape[1])
        next_pred = model.predict(input_reshaped, verbose=0)
        predictions.append(next_pred[0, 0])
        # Update recursive_input:
        # Geser 1 langkah ke depan, tambahkan prediksi baru sebagai fitur RR (kolom 0)
        # Pertahankan fitur lain (kolom 1 dan seterusnya) dari baris terakhir
        last_features = recursive_input[-1, 1:]
        new_row = np.concatenate(([next_pred[0, 0]], last_features))
        recursive_input = np.vstack((recursive_input[1:], new_row))
    pred_array = np.array(predictions).reshape(-1, 1)
    pred_array_rescaled = scaler_y.inverse_transform(pred_array)
    return pred_array_rescaled

def detect_seasons(pred_array_rescaled, start_date, days_per_dasarian=10):
    n = len(pred_array_rescaled)
    dates = [start_date + timedelta(days=days_per_dasarian * i) for i in range(n)]

    df = pd.DataFrame({
        'Tanggal': dates,
        'RR': pred_array_rescaled.flatten()
    })
    df['Bulan'] = df['Tanggal'].dt.month

    # Awal musim hujan
    awal_hujan = None
    for i in range(n - 2):
        window = pred_array_rescaled[i:i+3].flatten()
        if (window[0] >= 50) and (np.all(window >= 50) or np.sum(window) >= 150):
            awal_hujan = i
            break

    # Puncak musim hujan
    puncak_hujan_idx = None
    max_total = -np.inf
    for i in range(n - 2):
        window = pred_array_rescaled[i:i+3].flatten()
        total = np.sum(window)
        if total > max_total:
            max_total = total
            puncak_hujan_idx = i

    bulan_window_hujan = df['Bulan'][puncak_hujan_idx:puncak_hujan_idx+3]
    bulan_dominan_hujan = bulan_window_hujan.mode().iloc[0]

    # Awal musim kemarau
    awal_kemarau = None
    if awal_hujan is not None:
        for i in range(awal_hujan + 3, n - 2):
            window = pred_array_rescaled[i:i+3].flatten()
            if (window[0] < 50) and (np.all(window < 50) or np.sum(window) < 150):
                awal_kemarau = i
                break

    # Puncak musim kemarau
    puncak_kemarau_idx = None
    min_total = np.inf
    for i in range(n - 2):
        window = pred_array_rescaled[i:i+3].flatten()
        total = np.sum(window)
        if total < min_total:
            min_total = total
            puncak_kemarau_idx = i

    window_zero = pred_array_rescaled[puncak_kemarau_idx:puncak_kemarau_idx+3].flatten()
    if np.all(window_zero == 0):
        puncak_kemarau_idx += 1  # dasarian tengah

    bulan_window_kemarau = df['Bulan'][puncak_kemarau_idx:puncak_kemarau_idx+3]
    bulan_dominan_kemarau = bulan_window_kemarau.mode().iloc[0]

    # Durasi musim hujan
    durasi_hujan = None
    if awal_hujan is not None and awal_kemarau is not None:
        durasi_hujan = awal_kemarau - awal_hujan

    # Durasi musim kemarau
    durasi_kemarau = None
    if awal_kemarau is not None:
        akhir_kemarau_idx = awal_hujan if (awal_hujan is not None and awal_hujan > awal_kemarau) else n
        durasi_kemarau = akhir_kemarau_idx - awal_kemarau

    def idx_to_str(idx):
        if idx is None:
            return "Tidak terdeteksi"
        dasarian_ke = idx + 1
        tanggal = start_date + timedelta(days=days_per_dasarian * idx)
        return f"{tanggal.strftime('%Y-%m-%d')} (dasarian ke-{dasarian_ke})"

    result = {
        "musim_hujan": {
            "awal": idx_to_str(awal_hujan),
            "puncak": f"{idx_to_str(puncak_hujan_idx)}, Bulan dominan: {bulan_dominan_hujan}",
            "durasi (dasarian)": durasi_hujan if durasi_hujan is not None else "Tidak terdeteksi"
        },
        "musim_kemarau": {
            "awal": idx_to_str(awal_kemarau),
            "puncak": f"{idx_to_str(puncak_kemarau_idx)}, Bulan dominan: {bulan_dominan_kemarau}",
            "durasi (dasarian)": durasi_kemarau if durasi_kemarau is not None else "Tidak terdeteksi"
        }
    }
    return result

def set_background_and_style():
    # Define a single, consistent color palette
    background_color = "#FFFBDE"
    primary_color = "#90D1CA"
    secondary_color = "#096B68"
    text_color_header = "#096B68"
    text_color_general = "#096B68"
    dropdown_text_color = "#129990"

    table_header_bg_color = "#7AE2CF"
    table_cell_bg_color = "#FFFBDE"
    table_border_color = "#90D1CA"

    # --- TAMBAHKAN URL RAW GAMBAR HEADER DI SINI ---
    header_image_url = "https://raw.githubusercontent.com/IkmalKadafi/app_predict/main/Data/bg/header1.jpg" # GANTI DENGAN URL GAMBAR

    css = f"""
    <style>
    /* Targetkan kontainer utama aplikasi Streamlit */
    [data-testid="stAppViewContainer"], [data-testid="stDecoration"] {{
        background-color: {background_color} !important;
    }}

    /* Styling umum untuk teks */
    body, .stMarkdown, .stTextInput > div > div > input, .stTextArea > div > div > textarea {{
        color: {text_color_general} !important;
    }}

    h1, h2, h3, h4, h5, h6, .stSubheader {{
        color: {text_color_header} !important;
        font-weight: bold;
    }}

    /* Styling untuk Label Widget */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stRadio"] label,
    div[data-testid="stDateInput"] label,
    div[data-testid="stTimeInput"] label,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stSlider"] label {{
        color: {text_color_general} !important;
    }}

    /* Tombol */
    div[data-testid="stButton"] > button {{
        background-color: {primary_color} !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: 1px solid {primary_color} !important;
        padding: 0.6em 1.5em !important;
        transition: background-color 0.3s ease, color 0.3s ease, transform 0.2s ease !important;
    }}

    div[data-testid="stButton"] > button:hover {{
        background-color: {secondary_color} !important;
        color: {primary_color} !important;
        border: 1px solid {primary_color} !important;
        transform: scale(1.03);
    }}

    div[data-testid="stButton"] > button:active {{
        transform: scale(0.98);
    }}

    /* Dropdown (Selectbox) */
    div[data-testid="stSelectbox"] > div {{
        background-color: {secondary_color} !important;
        border-radius: 6px !important;
        border: 1px solid {primary_color} !important;
    }}

    div[data-testid="stSelectbox"] .st-bb,
    div[data-testid="stSelectbox"] .st-cq,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] input {{
        color: {dropdown_text_color} !important;
    }}
    
    div[data-baseweb="popover"] ul li {{
        background-color: {secondary_color} !important;
        color: {dropdown_text_color} !important;
    }}
    div[data-baseweb="popover"] ul li:hover {{
        background-color: {primary_color} !important;
        color: white !important;
    }}

    /* Header default Streamlit dengan Gambar Latar Belakang */
    [data-testid="stHeader"] {{
        background-image: url('{header_image_url}') !important; /* <--- Kembalikan baris ini */
        background-size: cover !important;
        background-position: center center !important;
        background-repeat: no-repeat !important;
        height: 80px !important;
    }}

    /* --- STYLING TABEL --- */
    [data-testid="stTable"] table {{
        width: 100%;
        border-collapse: collapse;
        background-color: {table_cell_bg_color} !important;
    }}

    [data-testid="stTable"] table th,
    [data-testid="stTable"] table td {{
        color: {text_color_general} !important;
        border: 1px solid {table_border_color} !important;
        padding: 8px !important;
        text-align: left !important;
    }}

    [data-testid="stTable"] table td {{
        background-color: {table_cell_bg_color} !important;
    }}

    [data-testid="stTable"] table th {{
        background-color: {table_header_bg_color} !important;
        color: {text_color_header} !important;
        font-weight: bold;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def main():
    st.title("Prediksi Musim di Jawa Timur")
    set_background_and_style()
    # Inisialisasi state jika belum ada
    if "prediksi_ditekan" not in st.session_state:
        st.session_state["prediksi_ditekan"] = False

    model, data, scaler_x, scaler_y, zona = pilih_topografi()
    musim = pilih_musim()
    start_date = pd.to_datetime("2024-10-01")
    look_back = 36

    if st.button("Prediksi Musim"):
        st.session_state["prediksi_ditekan"] = True  # tombol ditekan = hide tabel
        X_all = data[['TAVG', 'FF_AVG']]  # sesuaikan fitur
        X_scaled = scaler_x.transform(X_all)
        X_test_seq = buat_sequence(X_scaled, look_back=look_back)
        pred_array_rescaled = prediksi_recursive(model, scaler_y, X_test_seq, n_future=36)
        result = detect_seasons(pred_array_rescaled, start_date)

        st.subheader("Hasil Deteksi Musim:")
        if musim == "Musim Hujan":
            st.write("=== MUSIM HUJAN ===")
            st.write(f"Awal     : {result['musim_hujan']['awal']}")
            st.write(f"Puncak   : {result['musim_hujan']['puncak']}")
            st.write(f"Durasi   : {result['musim_hujan']['durasi (dasarian)']} dasarian\n")
        else:
            st.write("=== MUSIM KEMARAU ===")
            st.write(f"Awal     : {result['musim_kemarau']['awal']}")
            st.write(f"Puncak   : {result['musim_kemarau']['puncak']}")
            st.write(f"Durasi   : {result['musim_kemarau']['durasi (dasarian)']} dasarian\n")

        normal_musim = {
            'musim_hujan': {
                '303 (Dataran Rendah)': {'awal': 'November III', 'akhir': 'April III', 'durasi': 16},
                '311 (Dataran Tinggi)': {'awal': 'November I', 'akhir': 'April III', 'durasi': 18},
                '349 (Pesisir)': {'awal': 'November II', 'akhir': 'Mei I', 'durasi': 18},
            },
            'musim_kemarau': {
                '303 (Dataran Rendah)': {'awal': 'April III', 'akhir': 'November II', 'durasi': 21},
                '311 (Dataran Tinggi)': {'awal': 'April III', 'akhir': 'Oktober III', 'durasi': 19},
                '349 (Pesisir)': {'awal': 'Mei I', 'akhir': 'November I', 'durasi': 19},
            }
        }

        # Tampilkan data normal musim sesuai zona dan musim yang dipilih
        # Tampilkan data normal musim untuk SEMUA zona, tergantung musim yang dipilih
        if musim == "Musim Hujan":
            df_normal = pd.DataFrame.from_dict({
                'Zona': ['303 (Dataran Rendah)', '311 (Dataran Tinggi)', '349 (Pesisir)'],
                'Awal Musim Hujan': [
                    normal_musim['musim_hujan']['303 (Dataran Rendah)']['awal'],
                    normal_musim['musim_hujan']['311 (Dataran Tinggi)']['awal'],
                    normal_musim['musim_hujan']['349 (Pesisir)']['awal']
                ],
                'Akhir Musim Hujan': [
                    normal_musim['musim_hujan']['303 (Dataran Rendah)']['akhir'],
                    normal_musim['musim_hujan']['311 (Dataran Tinggi)']['akhir'],
                    normal_musim['musim_hujan']['349 (Pesisir)']['akhir']
                ],
                'Durasi (Dasarian)': [
                    normal_musim['musim_hujan']['303 (Dataran Rendah)']['durasi'],
                    normal_musim['musim_hujan']['311 (Dataran Tinggi)']['durasi'],
                    normal_musim['musim_hujan']['349 (Pesisir)']['durasi']
                ]
            })
            st.subheader("Data Normal Musim Hujan untuk Semua Zona")
        else:
            df_normal = pd.DataFrame.from_dict({
                'Zona': ['303 (Dataran Rendah)', '311 (Dataran Tinggi)', '349 (Pesisir)'],
                'Awal Musim Kemarau': [
                    normal_musim['musim_kemarau']['303 (Dataran Rendah)']['awal'],
                    normal_musim['musim_kemarau']['311 (Dataran Tinggi)']['awal'],
                    normal_musim['musim_kemarau']['349 (Pesisir)']['awal']
                ],
                'Akhir Musim Kemarau': [
                    normal_musim['musim_kemarau']['303 (Dataran Rendah)']['akhir'],
                    normal_musim['musim_kemarau']['311 (Dataran Tinggi)']['akhir'],
                    normal_musim['musim_kemarau']['349 (Pesisir)']['akhir']
                ],
                'Durasi (Dasarian)': [
                    normal_musim['musim_kemarau']['303 (Dataran Rendah)']['durasi'],
                    normal_musim['musim_kemarau']['311 (Dataran Tinggi)']['durasi'],
                    normal_musim['musim_kemarau']['349 (Pesisir)']['durasi']
                ]
            })
            st.subheader("Data Normal Musim Kemarau untuk Semua Zona")
        # Hide the index by setting it to empty strings
        df_normal.index = [''] * len(df_normal)
        st.table(df_normal)

    # Tabel Normal Musim hanya tampil jika tombol BELUM ditekan
    if not st.session_state["prediksi_ditekan"]:
        st.subheader("Tabel Data Normal Musim (Durasi dalam Dasarian)")
        normal_df = pd.DataFrame({
            "Zona": ["303 (Dataran Rendah)", "311 (Dataran Tinggi)", "349 (Pesisir)"],
            "Awal Hujan": ["November III", "November I", "November II"],
            "Akhir Hujan": ["April III", "April III", "Mei I"],
            "Durasi Hujan": [16, 18, 18],
            "Awal Kemarau": ["April III", "April III", "Mei I"],
            "Akhir Kemarau": ["November II", "Oktober III", "November I"],
            "Durasi Kemarau": [21, 19, 19]
        })
        normal_df.index = [''] * len(normal_df)
        st.table(normal_df)

if __name__ == "__main__":
    main()
