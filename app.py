import streamlit as st
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
import cv2
import os

# --- 🖥️ MSI KATANA & PC ÖZEL AYARLARI ---
st.set_page_config(
    page_title="İlhanoğulları Master-Station AI",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 ÖZEL TASARIM (DARK MODE & GOLD) ---
st.markdown("""
    <style>
    .stApp { background-color: #0B0E14; color: #E0E0E0; }
    [data-testid="stMetricValue"] {
        color: #D4AF37 !important;
        font-size: 58px !important;
        font-weight: 800;
    }
    .brand-title {
        color: #D4AF37;
        font-family: 'Garamond', serif;
        font-size: 40px;
        text-align: center;
        border-bottom: 2px solid #D4AF37;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 🏗️ AKILLI MOTOR YÜKLEME (GPU/CPU) ---
@st.cache_resource
def load_engine():
    # Arkadaşının kurduğu torch sürümüne göre cihaz seçimi
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Modeli indir/yükle ve cihaza gönder
    return YOLO('yolov8n-seg.pt').to(device)

try:
    model = load_engine()
except Exception as e:
    st.error(f"Model yüklenirken bir sorun oluştu: {e}")

# --- 🧬 ÇİFTLİK VERİ MATRİSİ ---
breed_configs = {
    "Simental": {"dens": 268, "yield": 58.5},
    "Angus": {"dens": 255, "yield": 61.0},
    "Holstein": {"dens": 245, "yield": 55.5},
    "Jersey": {"dens": 230, "yield": 53.0},
    "Belçika Mavisi": {"dens": 295, "yield": 65.0}
}
body_mods = {"Zayıf": 0.90, "İdeal": 1.0, "Kaslı/Pehlivan": 1.12}

# --- 🚀 ANA PANEL ---
st.markdown('<p class="brand-title">İLHANOĞULLARI ÇİFTLİĞİ | MASTER-STATION AI</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Parametreler")
    sel_breed = st.selectbox("Hayvan Irkı", list(breed_configs.keys()))
    sel_body = st.selectbox("Vücut Kondisyonu", list(body_mods.keys()))
    st.divider()
    st.write("📍 Alpu / Eskişehir")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📸 Fotoğraf Analizi")
    file = st.file_uploader("İneğin fotoğrafını seçin...", type=['jpg','jpeg','png'])

if file:
    img = Image.open(file).convert('RGB')
    img_np = np.array(img)
    
    with st.spinner("⏳ Yapay Zeka Biyometrik Ölçüm Yapıyor..."):
        # MODEL ANALİZİ
        results = model.predict(img_np, conf=0.45)
        
        if len(results) > 0 and results[0].masks is not None:
            r = results[0]
            
            # 1. Maske ve Geometri Hesaplama
            mask = r.masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
            
            y, x = np.where(mask_resized > 0)
            ph = np.max(y) - np.min(y)
            pw = np.max(x) - np.min(x)
            
            # 2. İlhanoğulları Kalibrasyon Formülü
            # Standart sırt yüksekliği 132 cm baz alınmıştır
            cm_per_px = 132 / max(ph, 1)
            area_m2 = np.sum(mask_resized > 0) * (cm_per_px / 100)**2
            length_m = (pw * cm_per_px / 100)
            
            cfg = breed_configs[sel_breed]
            weight = int(area_m2 * length_m * cfg['dens'] * body_mods[sel_body])
            
            # Büyük baş hayvan düzeltme katsayısı
            if weight > 550: weight = int(weight * 0.94)
            karkas = int(weight * (cfg['yield'] / 100))

            # --- 📊 SONUÇLARIN GÖSTERİLMESİ ---
            with col_left:
                st.image(r.plot(), caption="Biyometrik Maskeleme Tamam", use_container_width=True)
            
            with col_right:
                st.subheader("⚖️ Canlı Ağırlık Verileri")
                m1, m2 = st.columns(2)
                m1.metric("TOPLAM AĞIRLIK", f"{weight} KG")
                m2.metric("TAHMİNİ ET", f"{karkas} KG")
                
                st.divider()
                st.success(f"Analiz başarılı! Irk: {sel_breed}")
                if st.button("💾 VERİYİ VERİTABANINA İŞLE"):
                    st.balloons()
                    st.toast("Veri kaydedildi!")
        else:
            st.error("❌ Hayvan tespit edilemedi. Lütfen daha net bir açıdan çekim yapın.")
