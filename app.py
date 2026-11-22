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
    page_title="Kaçak Su Risk Analiz ve Saha Öncelik Paneli",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2.2rem;
            max-width: 1280px;
        }
        .ust-banner {
            background: linear-gradient(120deg, #0f2942 0%, #155e75 65%, #0e7490 100%);
            color: #ffffff;
            padding: 1.35rem 1.6rem;
            border-radius: 14px;
            margin-bottom: 1.1rem;
            box-shadow: 0 8px 24px rgba(15, 41, 66, 0.14);
        }
        .ust-banner h1 {
            color: #ffffff !important;
            font-size: 1.65rem !important;
            margin: 0 0 0.3rem 0 !important;
            font-weight: 700 !important;
        }
        .ust-banner p {
            color: #e0f2fe !important;
            margin: 0 !important;
            font-size: 0.95rem !important;
        }
        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 0.85rem 1rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: #475569 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #0f172a !important;
        }
        .rozet-yuksek {
            display: inline-block;
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
            padding: 0.32rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.88rem;
        }
        .rozet-orta {
            display: inline-block;
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
            padding: 0.32rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.88rem;
        }
        .rozet-dusuk {
            display: inline-block;
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            padding: 0.32rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.88rem;
        }
        .alt-bilgi {
            margin-top: 1.5rem;
            padding: 0.85rem 1.1rem;
            border-radius: 10px;
            background-color: #f1f5f9;
            color: #334155;
            font-size: 0.86rem;
            text-align: center;
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


def risk_skala_html(olas_val: float, orta_val: float, yuksek_val: float) -> str:
    pct = min(max(olas_val * 100.0, 0.0), 100.0)
    orta_pct = min(max(orta_val * 100.0, 1.0), 95.0)
    yuksek_pct = min(max(yuksek_val * 100.0, orta_pct + 2.0), 99.0)
    d_w = orta_pct
    o_w = max(yuksek_pct - orta_pct, 1.0)
    y_w = max(100.0 - yuksek_pct, 1.0)

    return f"""
    <div style="margin: 0.4rem 0 0.8rem 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#64748b; margin-bottom:4px;">
            <span>Düşük (&lt;%{orta_pct:.0f})</span>
            <span>Orta (%{orta_pct:.0f}–%{yuksek_pct:.0f})</span>
            <span>Yüksek (≥%{yuksek_pct:.0f})</span>
        </div>
        <div style="position:relative; height:12px; border-radius:999px; overflow:hidden; display:flex; border:1px solid #cbd5e1;">
            <div style="width:{d_w}%; background:#86efac;"></div>
            <div style="width:{o_w}%; background:#fde047;"></div>
            <div style="width:{y_w}%; background:#fca5a5;"></div>
        </div>
        <div style="position:relative; height:14px; margin-top:2px;">
            <div style="position:absolute; left:calc({pct}% - 6px); font-size:0.8rem; color:#0f172a; font-weight:700;">▲ %{pct:.1f}</div>
        </div>
    </div>
    """


# Üst Kurumsal Banner
st.markdown(
    """
    <div class="ust-banner">
        <h1>💧 Kaçak Su Risk Analiz ve Saha Öncelik Paneli</h1>
        <p>Abone tüketim profili, sayaç yaşı ve abone tipine göre kaçak şüphe skorunu anlık puanlayan saha karar destek ekranı</p>
    </div>
    """,
    unsafe_allow_html=True,
)

sol_panel, sag_panel = st.columns([1.02, 1.38], gap="large")

girdiler = {}
secilen_etiketler = {}

with sol_panel:
    st.subheader("1. Abone Bilgileri ve Hazır Senaryolar")

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

    tahmin_basildi = st.button("🚀 Risk Skorunu Hesapla", type="primary", use_container_width=True)

with sag_panel:
    st.subheader("2. Risk Değerlendirmesi ve Saha Görünümü")

    tam_satir = varsayilan_satir_olustur()
    tam_satir.update(girdiler)
    df_model = pd.DataFrame([tam_satir])[beklenen_sutunlar]
    olas = float(pipe.predict_proba(df_model)[:, 1][0])

    if olas >= yuksek_esik:
        risk_etiket = "Yüksek Risk"
        rozet_html = '<span class="rozet-yuksek">● YÜKSEK RİSK</span>'
        aksiyon = "Öncelikli saha denetimi, sayaç bağlantısı ve hat/mühür kontrolü önerilir."
        kontrol_maddeleri = [
            "Sayaç önü bağlantı ve by-pass / hat müdahalesi kontrolü",
            "Sayaç camı, mühür ve fiziksel hasar incelemesi",
            "Son 12 aylık dönemlik tüketim dalgalanmasının sahada doğrulanması",
        ]
    elif olas >= orta_esik:
        risk_etiket = "Orta Risk"
        rozet_html = '<span class="rozet-orta">● ORTA RİSK</span>'
        aksiyon = "İzleme listesine alınıp saha ekibi kapasitesine göre ikincil kontrol planlanabilir."
        kontrol_maddeleri = [
            "Önümüzdeki okuma döneminde tüketim artış/azalış takibi",
            "Sayaç yaşı ve tipi dikkate alınarak rutin kontrol sırasına ekleme",
        ]
    else:
        risk_etiket = "Düşük Risk"
        rozet_html = '<span class="rozet-dusuk">● DÜŞÜK RİSK</span>'
        aksiyon = "Ek saha müdahalesi gerektirmeyen olağan tüketim profili."
        kontrol_maddeleri = [
            "Olağan periyodik sayaç okuma ve faturalandırma akışı sürdürülür",
        ]

    if tahmin_basildi:
        st.session_state["sorgu_gecmisi"].insert(
            0,
            {
                "Abone Tipi": secilen_etiketler.get("abone_tipi", "-"),
                "Sayaç Tipi": secilen_etiketler.get("sayac_tipi", "-"),
                "Sayaç Yaşı": secilen_etiketler.get("sayac_yasi", "-"),
                "Yaz (m³)": secilen_etiketler.get("tuketim_yaz_m3", "-"),
                "Kış (m³)": secilen_etiketler.get("tuketim_kis_m3", "-"),
                "Risk (%)": round(olas * 100, 1),
                "Kademe": risk_etiket,
            },
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Kaçak Şüphe Skoru", f"%{olas * 100:.1f}")
    m2.metric("Yüksek Risk Eşiği", f"%{yuksek_esik * 100:.0f}")
    m3.metric("Sayaç Yaşı", f"{girdiler.get('sayac_yasi', 0)} yıl")

    st.markdown(f"<div style='margin: 0.5rem 0 0.25rem 0;'>{rozet_html}</div>", unsafe_allow_html=True)
    st.markdown(risk_skala_html(olas, orta_esik, yuksek_esik), unsafe_allow_html=True)

    st.info(f"**Saha Önerisi:** {aksiyon}")

    with st.container(border=True):
        st.markdown("**Saha Kontrol Odakları**")
        for madde in kontrol_maddeleri:
            st.markdown(f"- {madde}")

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

st.markdown(
    """
    <div class="alt-bilgi">
        Karar Destek Modeli: Random Forest Sınıflandırıcı • Test ROC-AUC: 0.92 • Canlı Eşik ve Saha Önceliklendirme Ekranı
    </div>
    """,
    unsafe_allow_html=True,
)
