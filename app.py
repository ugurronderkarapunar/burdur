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
        .rozet-yuksek {
            display: inline-block;
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
            padding: 0.32rem 0.78rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.88rem;
        }
        .rozet-orta {
            display: inline-block;
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
            padding: 0.32rem 0.78rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.88rem;
        }
        .rozet-dusuk {
            display: inline-block;
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            padding: 0.32rem 0.78rem;
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
    "tuketim_kis_m3",
    "hane_buyuklugu",
    "tuketim_yaz_m3",
    "sayac_yasi",
    "abone_tipi",
]

ETIKETLER = {
    "tuketim_kis_m3": "Kış Dönemi Tüketimi (m³)",
    "hane_buyuklugu": "Hane Büyüklüğü (Kişi)",
    "tuketim_yaz_m3": "Yaz Dönemi Tüketimi (m³)",
    "sayac_yasi": "Sayaç Yaşı (Yıl)",
    "abone_tipi": "Abone Tipi",
}

REFERANS_TUKETIM = {
    "normal_kis": 11.0,
    "normal_yaz": 14.8,
    "riskli_kis": 5.9,
    "riskli_yaz": 8.3,
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
yuksek_esik = float(paket.get("esik", 0.62))
orta_esik = float(paket.get("orta_risk_esigi", 0.25))
test_f1 = float(paket.get("test_f1", 0.6786))

if "sorgu_gecmisi" not in st.session_state:
    st.session_state["sorgu_gecmisi"] = []


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


def senaryo_uygula(kis: float, hane: int, yaz: float, yas: int, abone_adaylari: list[str]):
    st.session_state["tuketim_kis_m3"] = kis
    st.session_state["hane_buyuklugu"] = hane
    st.session_state["tuketim_yaz_m3"] = yaz
    st.session_state["sayac_yasi"] = yas

    kat_meta = ornek_degerler.get("abone_tipi", {})
    secenekler = [str(x) for x in kat_meta.get("secenekler", [])]
    if secenekler:
        for aday in abone_adaylari:
            if aday in secenekler:
                st.session_state["abone_tipi"] = aday
                break
        else:
            st.session_state["abone_tipi"] = secenekler[0]


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


def powerbi_tuketim_grafigi(kis_val: float, yaz_val: float) -> go.Figure:
    donemler = ["Kış Dönemi", "Yaz Dönemi"]
    girilen = [kis_val, yaz_val]
    normal = [REFERANS_TUKETIM["normal_kis"], REFERANS_TUKETIM["normal_yaz"]]
    riskli = [REFERANS_TUKETIM["riskli_kis"], REFERANS_TUKETIM["riskli_yaz"]]

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
    f"""
    <div class="ust-banner">
        <h1>💧 Kaçak Su Risk Analiz ve Saha Öncelik Paneli</h1>
        <p>Kış/yaz tüketimi, hane büyüklüğü ve sayaç sinyalleriyle kaçak şüphesini puanlayan karar destek paneli • Karar Eşiği: %{yuksek_esik*100:.0f} • Test F1: {test_f1:.4f}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

sol_panel, sag_panel = st.columns([1.0, 1.42], gap="large")

girdiler = {}
secilen_etiketler = {}

with sol_panel:
    st.subheader("1. Kritik Abone Girdileri ve Senaryolar")

    col_s1, col_s2, col_s3 = st.columns(3)
    if col_s1.button("🔴 Yüksek Risk", use_container_width=True):
        senaryo_uygula(1.5, 6, 3.0, 17, ["Tarım-Hayvancılık", "Sanayi", "İşyeri"])

    if col_s2.button("🟡 Orta Risk", use_container_width=True):
        senaryo_uygula(6.5, 4, 9.0, 11, ["Konut", "İşyeri"])

    if col_s3.button("🟢 Düşük Risk", use_container_width=True):
        senaryo_uygula(13.0, 3, 17.5, 3, ["Konut"])

    st.markdown("<br>", unsafe_allow_html=True)

    for col in ONEMLI_5_SUTUN:
        if col not in beklenen_sutunlar:
            continue
        baslik = ETIKETLER.get(col, col)
        meta = ornek_degerler.get(col, {})

        if col == "abone_tipi":
            secenekler = [str(x) for x in meta.get("secenekler") or [
                "Konut",
                "Tarım-Hayvancılık",
                "İşyeri",
                "Resmi",
                "Sanayi",
            ]]
            mevcut = str(st.session_state.get(col, secenekler[0]))
            idx = secenekler.index(mevcut) if mevcut in secenekler else 0
            secilen = st.selectbox(baslik, options=secenekler, index=idx, key=f"inp_{col}")
            girdiler[col] = secilen
            secilen_etiketler[col] = secilen

        elif meta.get("tip") == "kat":
            secenekler = [str(x) for x in meta.get("secenekler", [])]
            if not secenekler:
                secenekler = ["Varsayılan"]
            mevcut = str(st.session_state.get(col, secenekler[0]))
            idx = secenekler.index(mevcut) if mevcut in secenekler else 0
            secilen = st.selectbox(baslik, options=secenekler, index=idx, key=f"inp_{col}")
            girdiler[col] = secilen
            secilen_etiketler[col] = secilen

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
                    value=max(round(float(min_v), 2), min(
