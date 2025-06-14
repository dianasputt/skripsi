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
model_311 = tf.keras.models.load_model('Data/model/model_311_2.h5', custom_objects={'mse': tf.keras.losses.mse})
model_303 = tf.keras.models.load_model('Data/model/model_303_3.h5', custom_objects={'mse': tf.keras.losses.mse})
model_349 = tf.keras.models.load_model('Data/model/model_349_5.h5', custom_objects={'mse': tf.keras.losses.mse})

# Load data
data_311 = pd.read_excel("Data/data/data_311.xlsx")
data_303 = pd.read_excel("Data/data/data_303.xlsx")
data_349 = pd.read_excel("Data/data/data_349.xlsx")

# Load scalers
scaler_x_311 = joblib.load('Data/scaler/scaler_x_311_2.pkl')
scaler_x_303 = joblib.load('Data/scaler/scaler_x_303_3.pkl')
scaler_x_349 = joblib.load('Data/scaler/scaler_x_349_5.pkl')

scaler_y_311 = joblib.load('Data/scaler/scaler_y_311_2.pkl')
scaler_y_303 = joblib.load('Data/scaler/scaler_y_303_3.pkl')
scaler_y_349 = joblib.load('Data/scaler/scaler_y_349_5.pkl')

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

def create_direct_multistep_dataset(X_data, y_data, lag=36, n_future=36):
    X, Y = [], []
    for i in range(len(X_data) - lag - n_future + 1):
        X_seq = X_data[i:(i + lag), :]
        Y_seq = y_data[i + lag : i + lag + n_future]
        X.append(X_seq)
        Y.append(Y_seq.flatten())
    return np.array(X), np.array(Y)

def prediksi_multistep(model, scaler_y, X_test_seq, y_test_seq):
    y_pred = model.predict(X_test_seq, verbose=0)  # shape: (samples, n_future)
    
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_true_rescaled = scaler_y.inverse_transform(y_test_seq)

    return y_pred_rescaled, y_true_rescaled


def detect_seasons(y_pred, start_date, days_per_dasarian=10, threshold=50):
    y_pred = np.array(y_pred)
    # Jika 2D, ambil deret terakhir
    if y_pred.ndim > 1:
        y_pred = y_pred[-1]
    y_pred = y_pred.flatten()

    n = len(y_pred)
    dates = [start_date + timedelta(days=days_per_dasarian * i) for i in range(n)]

    # DataFrame bantu untuk bulan dan tanggal
    df = pd.DataFrame({
        'Tanggal': dates,
        'RR': y_pred
    })
    df['Bulan'] = df['Tanggal'].dt.month

    # Awal musim hujan
    awal_hujan = None
    for i in range(n - 2):  # cek window 3 dasarian
        window = y_pred[i:i+3]
        if (window[0] >= threshold) and (np.all(window >= threshold) or np.sum(window) >= threshold*3):
            awal_hujan = i
            break

    # Akhir Musim Hujan
    akhir_hujan = None
    if awal_hujan is not None:
        for i in range(awal_hujan + 3, n - 2):
            window = y_pred[i:i+3]
            if not ((window[0] >= threshold) and (np.all(window >= threshold) or np.sum(window) >= threshold*3)):
                akhir_hujan = i
                break
        if akhir_hujan is None:
            akhir_hujan = n
            
    # Durasi Musim Hujan
    durasi_hujan = None
    if awal_hujan is not None and akhir_hujan is not None and akhir_hujan > awal_hujan:
        durasi_hujan = akhir_hujan - awal_hujan

    # Puncak musim hujan
    puncak_hujan_idx = None
    if awal_hujan is not None and akhir_hujan is not None:
        max_total = -np.inf
        for i in range(awal_hujan, min(akhir_hujan, n - 2)):
            window = y_pred[i:i+3]
            total = np.sum(window)
            if total > max_total:
                max_total = total
                puncak_hujan_idx = i

    if puncak_hujan_idx is not None:
        bulan_window_hujan = df['Bulan'][puncak_hujan_idx:puncak_hujan_idx+3]
        bulan_dominan_hujan = bulan_window_hujan.mode().iloc[0]
    else:
        bulan_dominan_hujan = None

    # Awal musim kemarau
    awal_kemarau = None
    if awal_hujan is not None:
        for i in range(awal_hujan + 3, n - 2):
            window = y_pred[i:i+3]
            if (window[0] < threshold) and (np.all(window < threshold) or np.sum(window) < threshold*3):
                awal_kemarau = i
                break
                
    # Akhir Musim Kemarau
    akhir_kemarau = None
    if awal_kemarau is not None:
        for i in range(awal_kemarau + 3, n - 2):
            window = y_pred[i:i+3]
            if not ((window[0] < threshold) and (np.all(window < threshold) or np.sum(window) < threshold*3)):
                akhir_kemarau = i
                break
        if akhir_kemarau is None:
            akhir_kemarau = n

    # Durasi Musim Kemarau
    durasi_kemarau = None
    if awal_kemarau is not None and akhir_kemarau is not None and akhir_kemarau > awal_kemarau:
        durasi_kemarau = akhir_kemarau - awal_kemarau

    # Puncak musim kemarau
    puncak_kemarau_idx = None
    if awal_kemarau is not None and akhir_kemarau is not None:
        min_total = np.inf
        for i in range(awal_kemarau, min(akhir_kemarau, n - 2)):
            window = y_pred[i:i+3]
            total = np.sum(window)
            if total < min_total:
                min_total = total
                puncak_kemarau_idx = i

    if puncak_kemarau_idx is not None:
        bulan_window_kemarau = df['Bulan'][puncak_kemarau_idx:puncak_kemarau_idx+3]
        bulan_dominan_kemarau = bulan_window_kemarau.mode().iloc[0]
    else:
        bulan_dominan_kemarau = None

    def idx_to_str(idx):
        if idx is None:
            return "Tidak terdeteksi"
        dasarian_ke = idx + 1
        tanggal = start_date + timedelta(days=days_per_dasarian * idx)
        return f"{tanggal.strftime('%Y-%m-%d')} (dasarian ke-{dasarian_ke})"

    result = {
        "musim_hujan": {
            "awal": idx_to_str(awal_hujan),
            "puncak": f"{idx_to_str(puncak_hujan_idx)}, Bulan dominan: {bulan_dominan_hujan}" if puncak_hujan_idx is not None else "Tidak terdeteksi",
            "durasi (dasarian)": durasi_hujan if durasi_hujan is not None else "Tidak terdeteksi"
        },
        "musim_kemarau": {
            "awal": idx_to_str(awal_kemarau),
            "puncak": f"{idx_to_str(puncak_kemarau_idx)}, Bulan dominan: {bulan_dominan_kemarau}" if puncak_kemarau_idx is not None else "Tidak terdeteksi",
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

    if "prediksi_ditekan" not in st.session_state:
        st.session_state["prediksi_ditekan"] = False

    model, data, scaler_x, scaler_y, zona = pilih_topografi()
    musim = pilih_musim()
    start_date = pd.to_datetime("2024-10-01")
    look_back = 36
    n_future = 36

    if st.button("Prediksi Musim"):
        st.session_state["prediksi_ditekan"] = True

        # Persiapkan input dan output
        X_all = data[['TAVG', 'FF_AVG']]
        y_all = data[['RR']]

        X_scaled = scaler_x.transform(X_all)
        y_scaled = scaler_y.transform(y_all)

        # Buat sequence multi-step direct
        X_seq, y_seq = create_direct_multistep_dataset(X_scaled, y_scaled, lag=look_back, n_future=n_future)

        # Prediksi langsung
        y_pred_scaled = model.predict(X_seq, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # Deteksi musim dari prediksi terakhir
        pred_final = y_pred[-1].reshape(-1, 1)
        result = detect_seasons(pred_final, start_date)

        # Tampilkan hasil deteksi
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

        # Perbandingan dengan data normal
        normal_musim = {
            'musim_hujan': {
                '303': {'awal': 'November III', 'akhir': 'April III', 'durasi': 16},
                '311': {'awal': 'November I', 'akhir': 'April III', 'durasi': 18},
                '349': {'awal': 'November II', 'akhir': 'Mei I', 'durasi': 18},
            },
            'musim_kemarau': {
                '303': {'awal': 'April III', 'akhir': 'November II', 'durasi': 21},
                '311': {'awal': 'April III', 'akhir': 'Oktober III', 'durasi': 19},
                '349': {'awal': 'Mei I', 'akhir': 'November I', 'durasi': 19},
            }
        }

        zona_kode = zona.split()[0]  # Ekstrak '303', '311', atau '349'
        st.subheader(f"Data Normal Musim Zona {zona_kode}")
        st.write("Musim Hujan:")
        st.write(f"Awal Normal  : {normal_musim['musim_hujan'][zona_kode]['awal']}")
        st.write(f"Akhir Normal : {normal_musim['musim_hujan'][zona_kode]['akhir']}")
        st.write(f"Durasi       : {normal_musim['musim_hujan'][zona_kode]['durasi']} dasarian\n")

        st.write("Musim Kemarau:")
        st.write(f"Awal Normal  : {normal_musim['musim_kemarau'][zona_kode]['awal']}")
        st.write(f"Akhir Normal : {normal_musim['musim_kemarau'][zona_kode]['akhir']}")
        st.write(f"Durasi       : {normal_musim['musim_kemarau'][zona_kode]['durasi']} dasarian")


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
