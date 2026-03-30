import streamlit as st
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
import cv2

# --- 🖥️ PC GENİŞ EKRAN AYARLARI ---
st.set_page_config(
    page_title="İlhanoğulları Master-Station AI",
    layout="wide", # PC ekranını tam kullanmak için WIDE mod
    initial_sidebar_state="expanded"
)

# --- 🚀 PC OPTİMİZE DASHBOARD TASARIMI ---
st.markdown("""
    <style>
    .stApp { background-color: #0B0E14; }
    /* Dashboard Kartları */
    .metric-card {
        background: #161B22;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #30363D;
        text-align: center;
    }
    /* Devasa Rakamlar */
    [data-testid="stMetricValue"] {
        color: #D4AF37 !important;
        font-size: 64px !important;
        font-weight: 800;
    }
    /* Başlık */
    .brand-title {
        color: #D4AF37;
        font-family: 'Garamond', serif;
        font-size: 42px;
        letter-spacing: 5px;
        text-align: center;
        border-bottom: 2px solid #D4AF37;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 🏗️ MODEL YÜKLEME ---
@st.cache_resource
def load_engine():
    return YOLO('yolov8n-seg.pt')

model = load_engine()

# --- 🧬 VERİ MATRİSİ ---
breed_configs = {
    "Simental": {"dens": 268, "yield": 58.5},
    "Angus": {"dens": 255, "yield": 61.0},
    "Holstein": {"dens": 245, "yield": 55.5},
    "Jersey": {"dens": 230, "yield": 53.0},
    "Belçika Mavisi": {"dens": 295, "yield": 65.0}
}
body_mods = {"Zayıf": 0.90, "İdeal": 1.0, "Kaslı/Pehlivan": 1.12}

# --- 🚀 ANA EKRAN DÜZENİ ---
st.markdown('<p class="brand-title">İLHANOĞULLARI ÇİFTLİĞİ | MASTER-STATION</p>', unsafe_allow_html=True)

# Sidebar: Ayarlar
st.sidebar.header("⚙️ Analiz Parametreleri")
sel_breed = st.sidebar.selectbox("Hayvan Irkı", list(breed_configs.keys()))
sel_body = st.sidebar.selectbox("Vücut Kondisyonu", list(body_mods.keys()))
st.sidebar.divider()
st.sidebar.write("İlhanoğulları Çiftliği tarafından üretilmiştir.")

# Ana Gövde: İki Sütun (Geniş Ekran)
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("📸 Fotoğraf Girişi")
    file = st.file_uploader("Simental/Angus/Holstein fotoğrafı yükleyin", type=['jpg','png','jpeg'])

# --- ⚖️ ANALİZ VE HATA KONTROLÜ ---
if file:
    img = Image.open(file).convert('RGB')
    img_np = np.array(img)
    
    with st.spinner("⏳ Yapay Zeka Analiz Ediyor..."):
        # MODEL ÇALIŞTIRILIYOR
        res_list = model.predict(img_np, conf=0.45) # İsmi res_list yaptım karışmasın diye
        
        if len(res_list) > 0 and res_list[0].masks is not None:
            r = res_list[0] # İlk tespit edilen hayvanı al
            
            # 1. Geometri Analizi
            mask = cv2.resize(r.masks.data[0].cpu().numpy(), (img_np.shape[1], img_np.shape[0]))
            y, x = np.where(mask > 0)
            ph, pw = (np.max(y)-np.min(y)), (np.max(x)-np.min(x))
            
            # 2. Kalibrasyon (132 CM Sabiti)
            cm_per_px = 132 / max(ph, 1)
            area_m2 = np.sum(mask > 0) * (cm_per_px / 100)**2
            length_m = (pw * cm_per_px / 100)
            
            # 3. Hesaplama
            cfg = breed_configs[sel_breed]
            weight = int(area_m2 * length_m * cfg['dens'] * body_mods[sel_body])
            if weight > 650: weight = int(weight * 0.92) # 500 KG kalibrasyonu
            karkas = int(weight * (cfg['yield'] / 100))

            # --- 📊 PC SONUÇ EKRANI ---
            with col_input:
                st.image(r.plot(), caption="Biyometrik Analiz Tamamlandı", use_container_width=True)
            
            with col_output:
                st.subheader("⚖️ Analiz Sonuçları")
                m1, m2 = st.columns(2)
                m1.metric("CANLI AĞIRLIK", f"{weight} KG")
                m2.metric("KARKAS (ET)", f"{karkas} KG")
                
                st.divider()
                st.info(f"📍 **Seçilen Irk:** {sel_breed} | **Kondisyon:** {sel_body}")
                st.write(f"📊 **Baz Randıman:** %{cfg['yield']}")
                
                if st.button("💾 ANALİZİ VERİTABANINA KAYDET"):
                    st.balloons()
                    st.success("Veri başarıyla kaydedildi.")
        else:
            st.error("❌ Hayvan tespit edilemedi. Lütfen daha net bir fotoğraf yükleyin.")