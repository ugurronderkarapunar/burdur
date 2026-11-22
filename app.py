import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
            max-width: 1300px;
        }
        .ust-banner {
            background: linear-gradient(120deg, #0f2942 0%, #155e75 65%, #0e7490 100%);
            color: #ffffff;
            padding: 1.3rem 1.6rem;
            border-radius: 14px;
            margin-bottom: 1.1rem;
            box-shadow: 0 8px 24px rgba(15, 41, 66, 0.14);
        }
        .ust-banner h1 {
            color: #ffffff !important;
            font-size: 1.65rem !important;
            margin: 0 0 0.25rem 0 !important;
            font-weight: 700 !important;
        }
        .ust-banner p {
            color: #e0f2fe !important;
            margin: 0 !important;
            font-size: 0.95rem !important;
        }
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 0.85rem 1rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
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
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.86rem;
        }
        .rozet-orta {
            display: inline-block;
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.86rem;
        }
        .rozet-dusuk {
            display: inline-block;
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.86rem;
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


def powerbi_risk_kadran(olas_val: float, orta_val: float, yuksek_val: float) -> go.Figure:
    pct = round(min(max(olas_val * 100.0, 0.0), 100.0), 1)
    orta_pct = round(min(max(orta_val * 100.0, 1.0), 95.0), 1)
    yuksek_pct = round(min(max(yuksek_val * 100.0, orta_pct + 2.0), 99.0), 1)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 28, "color": "#0f172a"}},
            title={"text": "Kaçak Şüphe Skoru", "font": {"size": 14, "color": "#334155"}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 11}},
                "bar": {"color": "#0f2942", "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, orta_pct], "color": "#dcfce7"},
                    {"range": [orta_pct, yuksek_pct], "color": "#fef3c7"},
                    {"range": [yuksek_pct, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 3},
                    "thickness": 0.85,
                    "value": yuksek_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=38, b=10),
        paper_bgcolor="#ffffff",
    )
    return fig


def powerbi_tuketim_grafigi(yaz_val: float, kis_val: float) -> go.Figure:
    donemler = ["Yaz Dönemi", "Kış Dönemi"]
    girilen = [yaz_val, kis_val]
    normal = [REFERANS_TUKETIM["normal_yaz"], REFERANS_TUKETIM["normal_kis"]]
    riskli = [REFERANS_TUKETIM["riskli_yaz"], REFERANS_TUKETIM["riskli_kis"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Girilen Abone",
            x=donemler,
            y=girilen,
            text=[f"{v:.1f} m³" for v in girilen],
            textposition="outside",
            marker_color="#1d4ed8",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Normal Profil (Medyan)",
            x=donemler,
            y=normal,
            text=[f"{v:.1f} m³" for v in normal],
            textposition="outside",
            marker_color="#059669",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Kaçak Şüphe Profili",
            x=donemler,
            y=riskli,
            text=[f"{v:.1f} m³" for v in riskli],
            textposition="outside",
            marker_color="#dc2626",
        )
    )

    max_y = max(girilen + normal + riskli) * 1.22
    fig.update_layout(
        title=dict(
            text="Dönemsel Tüketim Kıyaslaması (m³)",
            font=dict(size=15, color="#0f172a"),
            x=0.0,
        ),
        barmode="group",
        bargap=0.24,
        bargroupgap=0.08,
        height=310,
        margin=dict(l=15, r=15, t=48, b=15),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis=dict(
            range=[0, max(max_y, 5.0)],
            showgrid=True,
            gridcolor="#e2e8f0",
            gridwidth=1,
            title="Tüketim (m³)",
        ),
        xaxis=dict(showgrid=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11),
        ),
    )
    return fig


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

sol_panel, sag_panel = st.columns([1.0, 1.42], gap="large")

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
    st.subheader("2. Power BI Görünümlü Risk ve Tüketim Paneli")

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
            "Son 12 aylık dönemlik tüketim sapmasının sahada doğrulanması",
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

    g_sol, g_sag = st.columns([0.9, 1.2], gap="medium")
    with g_sol:
        st.plotly_chart(powerbi_risk_kadran(olas, orta_esik, yuksek_esik), use_container_width=True)
        st.markdown(f"<div style='text-align:center;'>{rozet_html}</div>", unsafe_allow_html=True)
    with g_sag:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Saha Önerisi:** {aksiyon}")
        with st.container(border=True):
            st.markdown("**Saha Kontrol Odakları**")
            for madde in kontrol_maddeleri:
                st.markdown(f"- {madde}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(
        powerbi_tuketim_grafigi(
            float(girdiler.get("tuketim_yaz_m3", 0)),
            float(girdiler.get("tuketim_kis_m3", 0)),
        ),
        use_container_width=True,
    )

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
