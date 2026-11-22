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
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #eef2f6 !important;
            color: #0f172a !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.4rem;
            max-width: 1360px;
        }

        /* Streamlit form ve etiket okunabilirliği */
        label, .stMarkdown, .stText, p, span, div {
            color: #0f172a;
        }
        div[data-testid="stWidgetLabel"] label,
        div[data-testid="stWidgetLabel"] p {
            color: #0f172a !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input[type="number"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
        }
        button[kind="secondary"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            font-weight: 700 !important;
        }
        button[kind="secondary"]:hover {
            background-color: #f8fafc !important;
            border-color: #0f4c64 !important;
            color: #0f172a !important;
        }
        button[kind="primary"] {
            background: linear-gradient(120deg, #0b1f3a 0%, #0e7490 100%) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
        }
        button[kind="primary"] p,
        button[kind="primary"] span {
            color: #ffffff !important;
        }

        /* Üst Banner */
        .hero-card {
            background: linear-gradient(125deg, #0b1f3a 0%, #0f4c64 55%, #0e7490 100%);
            border-radius: 18px;
            padding: 1.45rem 1.8rem;
            color: #ffffff !important;
            box-shadow: 0 12px 28px rgba(11, 31, 58, 0.16);
            margin-bottom: 1.1rem;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.18);
            color: #ffffff !important;
            font-size: 0.76rem;
            font-weight: 700;
            padding: 0.25rem 0.72rem;
            border-radius: 999px;
            margin-bottom: 0.5rem;
            letter-spacing: 0.03em;
        }
        .hero-title {
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0 0 0.28rem 0;
            line-height: 1.2;
            color: #ffffff !important;
        }
        .hero-sub {
            font-size: 0.95rem;
            color: #f0f9ff !important;
            margin: 0;
        }

        /* KPI Kartları */
        .kpi-card {
            background: #ffffff !important;
            border: 1px solid #dbe3ee;
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        }
        .kpi-label {
            font-size: 0.76rem;
            font-weight: 700;
            color: #475569 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #0f172a !important;
            line-height: 1.15;
        }
        .kpi-hint {
            font-size: 0.8rem;
            color: #475569 !important;
            margin-top: 0.25rem;
        }

        /* Panel Kartları */
        .panel-card {
            background: #ffffff !important;
            border: 1px solid #dbe3ee;
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }
        .panel-title {
            font-size: 1.06rem;
            font-weight: 800;
            color: #0f172a !important;
            margin: 0 0 0.85rem 0;
        }

        /* Karar Kartları */
        .karar-yuksek {
            background: #fff5f5 !important;
            border-left: 5px solid #dc2626;
            border-top: 1px solid #fecaca;
            border-right: 1px solid #fecaca;
            border-bottom: 1px solid #fecaca;
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
        }
        .karar-orta {
            background: #fffbeb !important;
            border-left: 5px solid #d97706;
            border-top: 1px solid #fde68a;
            border-right: 1px solid #fde68a;
            border-bottom: 1px solid #fde68a;
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
        }
        .karar-dusuk {
            background: #f0fdf4 !important;
            border-left: 5px solid #16a34a;
            border-top: 1px solid #bbf7d0;
            border-right: 1px solid #bbf7d0;
            border-bottom: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
        }
        .karar-baslik {
            font-weight: 800;
            color: #0f172a !important;
            margin-bottom: 0.25rem;
            font-size: 0.95rem;
        }
        .karar-metin {
            font-size: 0.9rem;
            color: #1e293b !important;
            line-height: 1.45;
        }

        /* Rozetler */
        .rozet-yuksek {
            display: inline-block;
            background-color: #fee2e2 !important;
            color: #7f1d1d !important;
            border: 1px solid #f87171;
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.84rem;
        }
        .rozet-orta {
            display: inline-block;
            background-color: #fef3c7 !important;
            color: #78350f !important;
            border: 1px solid #fbbf24;
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.84rem;
        }
        .rozet-dusuk {
            display: inline-block;
            background-color: #dcfce7 !important;
            color: #14532d !important;
            border: 1px solid #4ade80;
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.84rem;
        }

        .footer-bar {
            margin-top: 1.2rem;
            padding: 0.9rem 1.2rem;
            border-radius: 12px;
            background-color: #ffffff !important;
            border: 1px solid #dbe3ee;
            color: #334155 !important;
            font-size: 0.85rem;
            font-weight: 600;
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

OZELLIK_ETKI_OZETI = [
    ("Kış Tüketimi", 0.337),
    ("Hane Büyüklüğü", 0.220),
    ("Yaz Tüketimi", 0.062),
    ("Sayaç Yaşı", 0.017),
    ("Abone Tipi (Tarım-Hayv.)", 0.012),
]


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
            number={"suffix": "%", "font": {"size": 30, "color": "#0f172a"}},
            title={"text": "Kaçak Şüphe Olasılığı", "font": {"size": 14, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 11, "color": "#334155"}},
                "bar": {"color": "#0b1f3a", "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
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
        height=235,
        margin=dict(l=18, r=18, t=38, b=8),
        paper_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
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
            textfont=dict(color="#0f172a", size=12),
            marker=dict(color="#1d4ed8", cornerradius=6),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Normal Profil",
            x=donemler,
            y=normal,
            text=[f"{v:.1f} m³" for v in normal],
            textposition="outside",
            textfont=dict(color="#0f172a", size=12),
            marker=dict(color="#059669", cornerradius=6),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Kaçak Şüphe Profili",
            x=donemler,
            y=riskli,
            text=[f"{v:.1f} m³" for v in riskli],
            textposition="outside",
            textfont=dict(color="#0f172a", size=12),
            marker=dict(color="#dc2626", cornerradius=6),
        )
    )

    max_y = max(girilen + normal + riskli) * 1.24
    fig.update_layout(
        title=dict(
            text="Dönemsel Tüketim Kıyaslaması (m³)",
            font=dict(size=15, color="#0f172a"),
            x=0.0,
        ),
        barmode="group",
        bargap=0.26,
        bargroupgap=0.08,
        height=295,
        margin=dict(l=12, r=12, t=46, b=12),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
        yaxis=dict(
            range=[0, max(max_y, 5.0)],
            showgrid=True,
            gridcolor="#e2e8f0",
            title=dict(text="Tüketim (m³)", font=dict(color="#0f172a")),
            tickfont=dict(color="#334155"),
        ),
        xaxis=dict(showgrid=False, tickfont=dict(color="#0f172a")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#0f172a"),
        ),
    )
    return fig


def powerbi_etki_grafigi() -> go.Figure:
    sirali = list(reversed(OZELLIK_ETKI_OZETI))
    isimler = [x[0] for x in sirali]
    degerler = [x[1] for x in sirali]

    fig = go.Figure(
        go.Bar(
            x=degerler,
            y=isimler,
            orientation="h",
            text=[f"+{v:.3f}" for v in degerler],
            textposition="outside",
            textfont=dict(color="#0f172a", size=12),
            marker=dict(color="#0e7490", cornerradius=6),
        )
    )
    fig.update_layout(
        title=dict(
            text="Modelde Kararı Etkileyen Başlıca Sinyaller (ΔF1)",
            font=dict(size=14, color="#0f172a"),
            x=0.0,
        ),
        height=230,
        margin=dict(l=10, r=46, t=38, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
        xaxis=dict(
            range=[0, max(degerler) * 1.24],
            showgrid=True,
            gridcolor="#e2e8f0",
            title=dict(text="Test F1 Katkısı", font=dict(color="#0f172a")),
            tickfont=dict(color="#334155"),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color="#0f172a")),
    )
    return fig


# Hero Banner
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">SAHA KARAR DESTEK SİSTEMİ</div>
        <h1 class="hero-title">💧 Kaçak Su Risk Analiz ve Önceliklendirme Paneli</h1>
        <p class="hero-sub">Tüketim davranışı, hane büyüklüğü ve sayaç göstergelerini analiz ederek yüksek şüpheli aboneleri önceliklendirir</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Üst KPI Şeridi
k1, k2, k3, k4 = st.columns(4, gap="medium
