import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hızlı Kaçak Risk Sorgulama",
    page_icon="💧",
    layout="centered",
)

MODEL_YOLU = "rf_kacak_suphe_final.joblib"

# Sunumda kullanıcıya sorulacak en kritik 5 değişken
ONEMLI_5_SUTUN = [
    "tuketim_yaz_m3",
    "tuketim_kis_m3",
    "sayac_yasi",
    "abone_tipi",
    "sayac_tipi",
]

ETIKETLER = {
    "tuketim_yaz_m3": "Yaz Dönemi Tüketimi (m³)",
    "tuketim_kis_m3": "Kış Dönemi Tüketimi (m³)",
    "sayac_yasi": "Sayaç Yaşı (Yıl)",
    "abone_tipi": "Abone Tipi",
    "sayac_tipi": "Sayaç Tipi",
}


@st.cache_resource
def model_yukle(yol: str):
    return joblib.load(yol)


try:
    paket = model_yukle(MODEL_YOLU)
except FileNotFoundError:
    st.error(f"'{MODEL_YOLU}' dosyası bulunamadı. Lütfen aynı klasöre koyun.")
    st.stop()

pipe = paket["pipeline"]
beklenen_sutunlar = paket["ozellik_sutunlari"]
ornek_degerler = paket.get("ornek_degerler", {})
yuksek_esik = float(paket.get("esik", 0.50))
orta_esik = float(paket.get("orta_risk_esigi", 0.20))


def varsayilan_satir_olustur() -> dict:
    satir = {}
    for col in beklenen_sutunlar:
        meta = ornek_degerler.get(col)
        if meta and meta.get("tip") == "kat":
            secenekler = meta.get("secenekler", [])
            satir[col] = secenekler[0] if secenekler else ""
        elif meta and meta.get("tip") == "num":
            val = meta.get("medyan", 0)
            satir[col] = int(round(val)) if meta.get("is_int") else round(float(val), 2)
        else:
            satir[col] = 0
    return satir


st.title("💧 Hızlı Kaçak Su Risk Tahmini")
st.caption("En kritik 5 bilgiyi girin; kayıtlı yapay zekâ modeli kaçak riskini anında hesaplasın.")

# Hazır senaryo butonları (Başkana sunum yaparken tek tıkla geçiş için)
col_s1, col_s2 = st.columns(2)
if col_s1.button("🔴 Yüksek Riskli Örnek Doldur"):
    st.session_state["tuketim_yaz_m3"] = 4.5
    st.session_state["tuketim_kis_m3"] = 3.0
    st.session_state["sayac_yasi"] = 16
    st.session_state["abone_tipi"] = "Tarım-Hayvancılık"
    st.session_state["sayac_tipi"] = "Mekanik"

if col_s2.button("🟢 Düşük Riskli Örnek Doldur"):
    st.session_state["tuketim_yaz_m3"] = 16.0
    st.session_state["tuketim_kis_m3"] = 12.0
    st.session_state["sayac_yasi"] = 3
    st.session_state["abone_tipi"] = "Konut"
    st.session_state["sayac_tipi"] = "Akıllı"

girdiler = {}
for col in ONEMLI_5_SUTUN:
    if col not in beklenen_sutunlar:
        continue
    baslik = ETIKETLER.get(col, col)
    meta = ornek_degerler.get(col, {})

    if meta.get("tip") == "kat":
        secenekler = meta.get("secenekler", [])
        mevcut = st.session_state.get(col, secenekler[0] if secenekler else "")
        idx = secenekler.index(mevcut) if mevcut in secenekler else 0
        girdiler[col] = st.selectbox(baslik, options=secenekler, index=idx, key=f"inp_{col}")
    else:
        is_int = bool(meta.get("is_int", False))
        min_v = meta.get("min", 0)
        max_v = meta.get("max", 1000)
        med_v = meta.get("medyan", 10)

        if is_int:
            varsayilan_val = int(st.session_state.get(col, round(med_v)))
            girdiler[col] = st.number_input(
                baslik,
                min_value=int(min_v),
                max_value=int(max_v),
                value=max(int(min_v), min(int(max_v), varsayilan_val)),
                step=1,
                key=f"inp_{col}",
            )
        else:
            varsayilan_val = float(st.session_state.get(col, round(float(med_v), 2)))
            girdiler[col] = st.number_input(
                baslik,
                min_value=round(float(min_v), 2),
                max_value=round(float(max_v), 2),
                value=max(round(float(min_v), 2), min(round(float(max_v), 2), varsayilan_val)),
                step=0.5,
                key=f"inp_{col}",
            )

if st.button("Tahmin Et", type="primary", use_container_width=True):
    tam_satir = varsayilan_satir_olustur()
    tam_satir.update(girdiler)

    df_model = pd.DataFrame([tam_satir])[beklenen_sutunlar]
    olas = float(pipe.predict_proba(df_model)[:, 1][0])

    st.divider()
    st.metric("Kaçak Şüphe Olasılığı", f"%{olas * 100:.1f}")

    if olas >= yuksek_esik:
        st.error("🔴 YÜKSEK RİSK — Saha denetimine öncelikli alınmalı.")
    elif olas >= orta_esik:
        st.warning("🟡 ORTA RİSK — Tüketim takibi ve kontrol önerilir.")
    else:
        st.success("🟢 DÜŞÜK RİSK — Normal tüketim görünümü.")
