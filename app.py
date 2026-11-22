import folium
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

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

MODEL_YOLU = "rf_kacak_suphe2_final.joblib"

ONEMLI_SUTUNLAR = [
    "ilce",
    "tuketim_kis_m3",
    "hane_buyuklugu",
    "tuketim_yaz_m3",
    "sayac_yasi",
    "abone_tipi",
]

ETIKETLER = {
    "ilce": "İlçe",
    "tuketim_kis_m3": "Kış Dönemi Tüketimi (m³)",
    "hane_buyuklugu": "Hane Büyüklüğü (Kişi)",
    "tuketim_yaz_m3": "Yaz Dönemi Tüketimi (m³)",
    "sayac_yasi": "Sayaç Yaşı (Yıl)",
    "abone_tipi": "Abone Tipi",
}

ILCE_KOORDINATLARI = {
    "Merkez": (37.7203, 30.2908),
    "Bucak": (37.4594, 30.5944),
    "Gölhisar": (37.1475, 29.6050),
    "Yeşilova": (37.5067, 29.7583),
    "Çavdır": (37.1528, 29.6694),
    "Tefenni": (37.3156, 29.7764),
    "Ağlasun": (37.6514, 30.5311),
    "Karamanlı": (37.3678, 29.8569),
    "Kemer": (37.3267, 30.0683),
    "Altınyayla": (36.9964, 29.5467),
    "Çeltikçi": (37.5425, 30.4658),
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
test_f1 = float(paket.get("test_f1", 0.6740))

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


def folium_ilce_haritasi(ilce_adi: str, olas_val: float, risk_etiket: str) -> folium.Map:
    varsayilan_koord = (37.7203, 30.2908)
    koord_bulundu = False

    if ilce_adi in ILCE_KOORDINATLARI:
        lat, lon = ILCE_KOORDINATLARI[ilce_adi]
        koord_bulundu = True
    else:
        for k, v in ILCE_KOORDINATLARI.items():
            if k.lower() == str(ilce_adi).strip().lower():
                lat, lon = v
                koord_bulundu = True
                break
        if not koord_bulundu:
            lat, lon = varsayilan_koord

    renk = "#dc2626" if olas_val >= yuksek_esik else ("#d97706" if olas_val >= orta_esik else "#16a34a")

    m = folium.Map(location=[lat, lon], zoom_start=11 if koord_bulundu else 9, tiles="CartoDB positron")

    popup_html = (
        f"<b>İlçe:</b> {ilce_adi}<br>"
        f"<b>Risk Skoru:</b> %{olas_val * 100:.1f}<br>"
        f"<b>Kademe:</b> {risk_etiket}"
    )

    folium.Circle(
        location=[lat, lon],
        radius=3500,
        color=renk,
        fill=True,
        fill_color=renk,
        fill_opacity=0.22,
        weight=2,
    ).add_to(m)

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=240),
        tooltip=f"{ilce_adi} • %{olas_val * 100:.1f} ({risk_etiket})",
        icon=folium.Icon(
            color="red" if olas_val >= yuksek_esik else ("orange" if olas_val >= orta_esik else "green"),
            icon="tint",
            prefix="fa",
        ),
    ).add_to(m)

    return m


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
        <h1 class="hero-title">💧 Kaçak Su Risk Analiz ve Bölgesel Harita Paneli</h1>
        <p class="hero-sub">İlçe bazlı model puanlaması ve canlı harita işaretlemesiyle yüksek şüpheli bölgeleri önceliklendirir</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Üst KPI Şeridi
k1, k2, k3, k4 = st.columns(4, gap="medium")
k1.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-label">Karar Eşiği</div>
        <div class="kpi-value">%{yuksek_esik * 100:.0f}</div>
        <div class="kpi-hint">Yüksek öncelikli saha barajı</div>
    </div>
    """,
    unsafe_allow_html=True,
)
k2.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-label">Test F1 Skoru</div>
        <div class="kpi-value">{test_f1:.4f}</div>
        <div class="kpi-hint">İlçe dahil Random Forest modeli</div>
    </div>
    """,
    unsafe_allow_html=True,
)
k3.markdown(
    """
    <div class="kpi-card">
        <div class="kpi-label">Birincil Sinyal</div>
        <div class="kpi-value">Kış & Hane</div>
        <div class="kpi-hint">En yüksek ayrım gücü sağlayan ikili</div>
    </div>
    """,
    unsafe_allow_html=True,
)
k4.markdown(
    """
    <div class="kpi-card">
        <div class="kpi-label">Harita Modu</div>
        <div class="kpi-value">Folium Canlı</div>
        <div class="kpi-hint">Seçilen ilçeyi haritada işaretler</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

sol_panel, sag_panel = st.columns([0.96, 1.44], gap="large")

girdiler = {}
secilen_etiketler = {}

with sol_panel:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">1. İlçe ve Abone Girdileri</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3, gap="small")
    if col_s1.button("🔴 Yüksek Risk", use_container_width=True):
        senaryo_uygula(1.5, 6, 3.0, 17, ["Tarım-Hayvancılık", "Sanayi", "İşyeri"])

    if col_s2.button("🟡 Orta Risk", use_container_width=True):
        senaryo_uygula(6.5, 4, 9.0, 11, ["Konut", "İşyeri"])

    if col_s3.button("🟢 Düşük Risk", use_container_width=True):
        senaryo_uygula(13.0, 3, 17.5, 3, ["Konut"])

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    for col in ONEMLI_SUTUNLAR:
        if col not in beklenen_sutunlar:
            continue
        baslik = ETIKETLER.get(col, col)
        meta = ornek_degerler.get(col, {})

        if col == "ilce":
            secenekler = [str(x) for x in meta.get("secenekler", [])] or list(ILCE_KOORDINATLARI.keys())
            mevcut = str(st.session_state.get(col, secenekler[0]))
            idx = secenekler.index(mevcut) if mevcut in secenekler else 0
            secilen = st.selectbox(baslik, options=secenekler, index=idx, key=f"inp_{col}")
            girdiler[col] = secilen
            secilen_etiketler[col] = secilen

        elif col == "abone_tipi":
            secenekler = [
                str(x)
                for x in (
                    meta.get("secenekler")
                    or ["Konut", "Tarım-Hayvancılık", "İşyeri", "Resmi", "Sanayi"]
                )
            ]
            mevcut = str(st.session_state.get(col, secenekler[0]))
            idx = secenekler.index(mevcut) if mevcut in secenekler else 0
            secilen = st.selectbox(baslik, options=secenekler, index=idx, key=f"inp_{col}")
            girdiler[col] = secilen
            secilen_etiketler[col] = secilen

        elif meta.get("tip") == "kat":
            secenekler = [str(x) for x in meta.get("secenekler", [])] or ["Varsayılan"]
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
                    value=max(round(float(min_v), 2), min(round(float(max_v), 2), varsayilan_val)),
                    step=0.5,
                    key=f"inp_{col}",
                )
            girdiler[col] = val
            secilen_etiketler[col] = val

    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
    tahmin_basildi = st.button("🚀 Risk Skorunu Hesapla", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with sag_panel:
    tam_satir = varsayilan_satir_olustur()
    tam_satir.update(girdiler)
    df_model = pd.DataFrame([tam_satir])[beklenen_sutunlar]
    olas = float(pipe.predict_proba(df_model)[:, 1][0])
    secili_ilce = str(girdiler.get("ilce", "Merkez"))

    if olas >= yuksek_esik:
        risk_etiket = "Yüksek Risk"
        rozet_html = '<span class="rozet-yuksek">● YÜKSEK RİSK — ÖNCELİKLİ SAHA</span>'
        karar_class = "karar-yuksek"
        aksiyon = f"{secili_ilce} ilçesinde öncelikli saha denetimi, sayaç bağlantısı ve hat/mühür kontrolü önerilir."
        kontrol_maddeleri = [
            f"{secili_ilce} bölgesinde düşük kış/yaz tüketimi ve hane büyüklüğü uyumsuzluğunun kontrolü",
            "Sayaç önü bağlantı, mühür ve olası by-pass / hat müdahalesi incelemesi",
            "Sayaç yaşı ve abone grubu dikkate alınarak sahada tüketim doğrulaması",
        ]
    elif olas >= orta_esik:
        risk_etiket = "Orta Risk"
        rozet_html = '<span class="rozet-orta">● ORTA RİSK — İZLEME LİSTESİ</span>'
        karar_class = "karar-orta"
        aksiyon = f"{secili_ilce} ilçesi izleme listesine alınıp saha ekibi kapasitesine göre ikincil kontrol planlanabilir."
        kontrol_maddeleri = [
            "Önümüzdeki okuma döneminde tüketim ve hane profili sapmasının takibi",
            "Sayaç yaşı ve abone tipine göre rutin kontrol planına dahil etme",
        ]
    else:
        risk_etiket = "Düşük Risk"
        rozet_html = '<span class="rozet-dusuk">● DÜŞÜK RİSK — OLAĞAN PROFİL</span>'
        karar_class = "karar-dusuk"
        aksiyon = f"{secili_ilce} için ek saha müdahalesi gerektirmeyen olağan tüketim profili."
        kontrol_maddeleri = [
            "Olağan periyodik sayaç okuma ve faturalandırma akışı sürdürülür",
        ]

    if tahmin_basildi:
        st.session_state["sorgu_gecmisi"].insert(
            0,
            {
                "İlçe": secili_ilce,
                "Abone Tipi": secilen_etiketler.get("abone_tipi", "-"),
                "Kış (m³)": secilen_etiketler.get("tuketim_kis_m3", "-"),
                "Hane": secilen_etiketler.get("hane_buyuklugu", "-"),
                "Yaz (m³)": secilen_etiketler.get("tuketim_yaz_m3", "-"),
                "Sayaç Yaşı": secilen_etiketler.get("sayac_yasi", "-"),
                "Risk (%)": round(olas * 100, 1),
                "Kademe": risk_etiket,
            },
        )

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">2. Risk Skoru ve Saha Karar Özeti</div>', unsafe_allow_html=True)

    g_sol, g_sag = st.columns([0.92, 1.28], gap="medium")
    with g_sol:
        st.plotly_chart(powerbi_risk_kadran(olas, orta_esik, yuksek_esik), use_container_width=True)
        st.markdown(f"<div style='text-align:center;'>{rozet_html}</div>", unsafe_allow_html=True)
    with g_sag:
        st.markdown(
            f"""
            <div class="{karar_class}">
                <div class="karar-baslik">Saha Aksiyon Önerisi ({secili_ilce})</div>
                <div class="karar-metin">{aksiyon}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Saha Kontrol Maddeleri**")
            for madde in kontrol_maddeleri:
                st.markdown(f"- {madde}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">3. Bölgesel Harita Görünümü — {secili_ilce
