import joblib
import numpy as np
import pandas as pd
import streamlit as st

# scikit-learn 1.6.1 ile kaydedilmiş pipeline için uyumluluk yaması
import sklearn.compose._column_transformer as _ct

if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

st.set_page_config(
    page_title="Kaçak Su Risk Analiz ve Karar Paneli",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(28, 131, 225, 0.06);
            border: 1px solid rgba(28, 131, 225, 0.18);
            padding: 0.85rem 1rem;
            border-radius: 0.65rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_YOLU = "rf_kacak_suphe_final.joblib"

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

# Ekrandaki metin -> Verideki sayısal kod eşleşmesi (kendi verinize göre düzenleyin)
ABONE_TIPI_MAP = {
    "Konut": 0,
    "Tarım-Hayvancılık": 1,
    "İşyeri": 2,
    "Resmi": 3,
    "Sanayi": 4,
}

SAYAC_TIPI_MAP = {
    "Mekanik": 0,
    "Akıllı": 1,
}

KOD_SOZLUKLERI = {
    "abone_tipi": ABONE_TIPI_MAP,
    "sayac_tipi": SAYAC_TIPI_MAP,
}

# Görsel kıyas için referans medyan değerler
REFERANS_TUKETIM = {
    "normal_yaz": 14.8,
    "normal_kis": 11.0,
    "riskli_yaz": 8.3,
    "riskli_kis": 5.9,
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

if "sorgu_gecmisi" not in st.session_state:
    st.session_state["sorgu_gecmisi"] = []


def varsayilan_satir_olustur() -> dict:
    satir = {}
    for col in beklenen_sutunlar:
        meta = ornek_degerler.get(col)
        if meta and meta.get("tip") == "kat":
            secenekler = meta.get("secenekler", [])
            satir[col] = secenekler[0] if secenekler else 0
        elif meta and meta.get("tip") == "num":
            val = meta.get("medyan", 0)
            satir[col] = int(round(val)) if meta.get("is_int") else round(float(val), 2)
        else:
            satir[col] = 0
    return satir


# Üst Başlık Bandı
bas_sol, bas_sag = st.columns([3, 1])
with bas_sol:
    st.title("💧 Kaçak Su Risk Analiz ve Saha Öncelik Paneli")
    st.caption("5 kritik abone ve tüketim göstergesi üzerinden hızlı kaçak şüphe değerlendirmesi")
with bas_sag:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**Aktif Eşikler**\n\nOrta: `%{orta_esik*100:.0f}` • Yüksek: `%{yuksek_esik*100:.0f}`")

st.divider()

# Ana İki Sütunlu Ekran
sol_panel, sag_panel = st.columns([1.05, 1.35], gap="large")

girdiler = {}
secilen_etiketler = {}

with sol_panel:
    st.subheader("1. Abone ve Tüketim Bilgileri")

    col_s1, col_s2, col_s3 = st.columns(3)
    if col_s1.button("🔴 Yüksek Risk", use_container_width=True):
        st.session_state["tuketim_yaz_m3"] = 2.5
        st.session_state["tuketim_kis_m3"] = 1.5
        st.session_state["sayac_yasi"] = 17
        st.session_state["abone_tipi"] = "Tarım-Hayvancılık"
        st.session_state["sayac_tipi"] = "Mekanik"

    if col_s2.button("🟡 Orta Risk", use_container_width=True):
        st.session_state["tuketim_yaz_m3"] = 8.5
        st.session_state["tuketim_kis_m3"] = 6.5
        st.session_state["sayac_yasi"] = 11
        st.session_state["abone_tipi"] = "Konut"
        st.session_state["sayac_tipi"] = "Mekanik"

    if col_s3.button("🟢 Düşük Risk", use_container_width=True):
        st.session_state["tuketim_yaz_m3"] = 17.5
        st.session_state["tuketim_kis_m3"] = 13.0
        st.session_state["sayac_yasi"] = 3
        st.session_state["abone_tipi"] = "Konut"
        st.session_state["sayac_tipi"] = "Akıllı"

    st.markdown("<br>", unsafe_allow_html=True)

    for col in ONEMLI_5_SUTUN:
        if col not in beklenen_sutunlar:
            continue
        baslik = ETIKETLER.get(col, col)
        meta = ornek_degerler.get(col, {})

        if col in KOD_SOZLUKLERI:
            sozluk = KOD_SOZLUKLERI[col]
            secenek_isimleri = list(sozluk.keys())
            mevcut_isim = st.session_state.get(col, secenek_isimleri[0])
            idx = secenek_isimleri.index(mevcut_isim) if mevcut_isim in secenek_isimleri else 0
            secilen_isim = st.selectbox(baslik, options=secenek_isimleri, index=idx, key=f"inp_{col}")
            girdiler[col] = sozluk[secilen_isim]
            secilen_etiketler[col] = secilen_isim

        elif meta.get("tip") == "kat":
            secenekler = meta.get("secenekler", [])
            mevcut = st.session_state.get(col, secenekler[0] if secenekler else "")
            idx = secenekler.index(mevcut) if mevcut in secenekler else 0
            secilen = st.selectbox(baslik, options=secenekler, index=idx, key=f"inp_{col}")
            girdiler[col] = secilen
            secilen_etiketler[col] = str(secilen)

        else:
            is_int = bool(meta.get("is_int", False))
            min_v = meta.get("min", 0)
            max_v = meta.get("max", 1000)
            med_v = meta.get("medyan", 10)

            if is_int:
                varsayilan_val = int(st.session_state.get(col, round(med_v)))
                val = st.number_input(
                    baslik,
                    min_value=int(min_v),
                    max_value=int(max_v),
                    value=max(int(min_v), min(int(max_v), varsayilan_val)),
                    step=1,
                    key=f"inp_{col}",
                )
            else:
                varsayilan_val = float(st.session_state.get(col, round(float(med_v), 2)))
                val = st.number_input(
                    baslik,
                    min_value=round(float(min_v), 2),
                    max_value=round(float(max_v), 2),
                    value=max(round(float(min_v), 2), min(round(float(max_v), 2), varsayilan_val)),
                    step=0.5,
                    key=f"inp_{col}",
                )
            girdiler[col] = val
            secilen_etiketler[col] = val

    tahmin_basildi = st.button("🔍 Risk Skorunu Hesapla", type="primary", use_container_width=True)

with sag_panel:
    st.subheader("2. Risk Değerlendirmesi ve Tüketim Görünümü")

    tam_satir = varsayilan_satir_olustur()
    tam_satir.update(girdiler)
    df_model = pd.DataFrame([tam_satir])[beklenen_sutunlar]
    olas = float(pipe.predict_proba(df_model)[:, 1][0])

    if olas >= yuksek_esik:
        risk_etiket = "Yüksek Risk"
        aksiyon = "Öncelikli saha denetimi, sayaç bağlantısı ve tüketim geçmişi incelemesi önerilir."
    elif olas >= orta_esik:
        risk_etiket = "Orta Risk"
        aksiyon = "İzleme listesine alınıp saha kapasitesine göre ikincil kontrol planlanabilir."
    else:
        risk_etiket = "Düşük Risk"
        aksiyon = "Ek saha müdahalesi gerektirmeyen olağan tüketim profili."

    if tahmin_basildi:
        st.session_state["sorgu_gecmisi"].insert(
            0,
            {
                "Abone Tipi": secilen_etiketler.get("abone_tipi", "-"),
                "Sayaç Tipi": secilen_etiketler.get("sayac_tipi", "-"),
                "Sayaç Yaşı": secilen_etiketler.get("sayac_yasi", "-"),
                "Yaz (m³)": secilen_etiketler.get("tuketim_yaz_m3", "-"),
                "Kış (m³)": secilen_etiketler.get("tuketim_kis_m3", "-"),
                "Risk Olasılığı (%)": round(olas * 100, 1),
                "Risk Seviyesi": risk_etiket,
            },
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Kaçak Şüphe Olasılığı", f"%{olas * 100:.1f}")
    m2.metric("Risk Kademesi", risk_etiket)
    m3.metric("Sayaç Yaşı", f"{girdiler.get('sayac_yasi', 0)} yıl")

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(min(max(olas, 0.0), 1.0))

    if risk_etiket == "Yüksek Risk":
        st.error(f"🔴 **YÜKSEK RİSK** — {aksiyon}")
    elif risk_etiket == "Orta Risk":
        st.warning(f"🟡 **ORTA RİSK** — {aksiyon}")
    else:
        st.success(f"🟢 **DÜŞÜK RİSK** — {aksiyon}")

    notlar = []
    if float(girdiler.get("tuketim_yaz_m3", 99)) < 8.0:
        notlar.append(f"Yaz tüketimi düşük ({girdiler['tuketim_yaz_m3']} m³)")
    if float(girdiler.get("tuketim_kis_m3", 99)) < 6.0:
        notlar.append(f"Kış tüketimi düşük ({girdiler['tuketim_kis_m3']} m³)")
    if float(girdiler.get("sayac_yasi", 0)) >= 12:
        notlar.append(f"Sayaç yaşı yüksek ({girdiler['sayac_yasi']} yıl)")
    if girdiler.get("sayac_tipi") == SAYAC_TIPI_MAP.get("Mekanik"):
        notlar.append("Mekanik sayaç")
    if girdiler.get("abone_tipi") == ABONE_TIPI_MAP.get("Tarım-Hayvancılık"):
        notlar.append("Tarım-Hayvancılık grubu")

    if notlar:
        st.caption("**Öne çıkan sinyaller:** " + " • ".join(notlar))

    st.markdown("**Dönemsel Tüketim Profil Kıyaslaması (m³)**")
    df_kiyas = pd.DataFrame(
        {
            "Girilen Abone": [
                float(girdiler.get("tuketim_yaz_m3", 0)),
                float(girdiler.get("tuketim_kis_m3", 0)),
            ],
            "Normal Profil": [
                REFERANS_TUKETIM["normal_yaz"],
                REFERANS_TUKETIM["normal_kis"],
            ],
            "Kaçak Şüphe Profili": [
                REFERANS_TUKETIM["riskli_yaz"],
                REFERANS_TUKETIM["riskli_kis"],
            ],
        },
        index=["Yaz Tüketimi (m³)", "Kış Tüketimi (m³)"],
    )
    st.bar_chart(df_kiyas, use_container_width=True)

    with st.expander("Modele gönderilen sayısal kod ve girdi özeti"):
        st.json({ETIKETLER.get(k, k): v for k, v in girdiler.items()})

# Alt Bölüm: Sunum Sorgu Geçmişi ve Dışa Aktarma
if st.session_state["sorgu_gecmisi"]:
    st.divider()
    gecmis_bas, gecmis_but = st.columns([3, 1])
    with gecmis_bas:
        st.subheader("3. Sunum Sorgu Geçmişi")
    with gecmis_but:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Geçmişi Temizle", use_container_width=True):
            st.session_state["sorgu_gecmisi"] = []
            st.rerun()

    df_gecmis = pd.DataFrame(st.session_state["sorgu_gecmisi"])
    st.dataframe(df_gecmis, use_container_width=True, hide_index=True)

    csv_bytes = df_gecmis.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Sorgu Geçmişini CSV İndir",
        data=csv_bytes,
        file_name="kacak_risk_sorgu_gecmisi.csv",
        mime="text/csv",
    )
