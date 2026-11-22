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

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stApp {
            background-color: #f4f7fb;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1360px;
        }
        .hero-card {
            background: linear-gradient(125deg, #0b1f3a 0%, #0f4c64 55%, #0e7490 100%);
            border-radius: 18px;
            padding: 1.5rem 1.85rem;
            color: #ffffff;
            box-shadow: 0 12px 28px rgba(11, 31, 58, 0.16);
            margin-bottom: 1.15rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.14);
            color: #e0f2fe;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.55rem;
            letter-spacing: 0.02em;
        }
        .hero-title {
            font-size: 1.75rem;
            font-weight: 800;
            margin: 0 0 0.3rem 0;
            line-height: 1.2;
            color: #ffffff;
        }
        .hero-sub {
            font-size: 0.96rem;
            color: #dbeafe;
            margin: 0;
        }
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e6ecf5;
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }
        .kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.3rem;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.15;
        }
        .kpi-hint {
            font-size: 0.78rem;
            color: #475569;
            margin-top: 0.25rem;
        }
        .panel-card {
            background: #ffffff;
            border: 1px solid #e6ecf5;
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }
        .panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0 0 0.85rem 0;
        }
        .karar-yuksek {
            background: linear-gradient(90deg, #fef2f2 0%, #ffffff 100%);
            border-left: 5px solid #dc2626;
            border-top: 1px solid #fecaca;
            border-right: 1px solid #fecaca;
            border-bottom: 1px solid #fecaca;
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
            margin-top: 0.6rem;
        }
        .karar-orta {
            background: linear-gradient(90deg, #fffbeb 0%, #ffffff 100%);
            border-left: 5px solid #d97706;
            border-top: 1px solid #fde68a;
            border-right: 1px solid #fde68a;
            border-bottom: 1px solid #fde68a;
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
            margin-top: 0.6rem;
        }
        .karar-dusuk {
            background: linear-gradient(90deg, #f0fdf4 0%, #ffffff 100%);
            border-left: 5px solid #16a34a;
            border-top: 1px solid #bbf7d0;
            border-right: 1px solid #bbf7d0;
            border-bottom: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
            margin-top: 0.6rem;
        }
        .rozet-yuksek {
            display: inline-block;
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .rozet-orta {
            display: inline-block;
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .rozet-dusuk {
            display: inline-block;
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .footer-bar {
            margin-top: 1.2rem;
            padding: 0.9rem 1.2rem;
            border-radius: 12px;
            background-color: #ffffff;
            border: 1px solid #e6ecf5;
            color: #475569;
            font-size: 0.84rem;
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

# Son analizdeki test F1 etki ağırlıkları (görsel kıyas için)
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
            number={"suffix": "%", "font": {"size": 30, "color": "#0f172a", "family": "Plus Jakarta Sans"}},
            title={"text": "Kaçak Şüphe Olasılığı", "font": {"size": 14, "color": "#475569"}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 11, "color": "#64748b"}},
                "bar": {"color": "#0b1f3a", "thickness": 0.28},
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
        height=235,
        margin=dict(l=18, r=18, t=38, b=8),
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
        yaxis=dict(
            range=[0, max(max_y, 5.0)],
            showgrid=True,
            gridcolor="#eef2f6",
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
        margin=dict(l=10, r=42, t=38, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            range=[0, max(degerler) * 1.22],
            showgrid=True,
            gridcolor="#eef2f6",
            title="Test F1 Katkısı",
        ),
        yaxis=dict(showgrid=False),
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
        <div class="kpi-hint">Doğrulanmış Random Forest modeli</div>
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
        <div class="kpi-label">Operasyon Modu</div>
        <div class="kpi-value">Canlı Puanlama</div>
        <div class="kpi-hint">Anlık risk skoru ve saha önerisi</div>
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
    st.markdown('<div class="panel-title">1. Abone Bilgileri ve Hazır Senaryolar</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3, gap="small")
    if col_s1.button("🔴 Yüksek Risk", use_container_width=True):
        senaryo_uygula(1.5, 6, 3.0, 17, ["Tarım-Hayvancılık", "Sanayi", "İşyeri"])

    if col_s2.button("🟡 Orta Risk", use_container_width=True):
        senaryo_uygula(6.5, 4, 9.0, 11, ["Konut", "İşyeri"])

    if col_s3.button("🟢 Düşük Risk", use_container_width=True):
        senaryo_uygula(13.0, 3, 17.5, 3, ["Konut"])

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    for col in ONEMLI_5_SUTUN:
        if col not in beklenen_sutunlar:
            continue
        baslik = ETIKETLER.get(col, col)
        meta = ornek_degerler.get(col, {})

        if col == "abone_tipi":
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

    if olas >= yuksek_esik:
        risk_etiket = "Yüksek Risk"
        rozet_html = '<span class="rozet-yuksek">● YÜKSEK RİSK — ÖNCELİKLİ SAHA</span>'
        karar_class = "karar-yuksek"
        aksiyon = "Öncelikli saha denetimi, sayaç bağlantısı ve hat/mühür kontrolü önerilir."
        kontrol_maddeleri = [
            "Düşük kış/yaz tüketimi ve hane büyüklüğü uyumsuzluğunun yerinde kontrolü",
            "Sayaç önü bağlantı, mühür ve olası by-pass / hat müdahalesi incelemesi",
            "Sayaç yaşı ve abone grubu dikkate alınarak sahada tüketim doğrulaması",
        ]
    elif olas >= orta_esik:
        risk_etiket = "Orta Risk"
        rozet_html = '<span class="rozet-orta">● ORTA RİSK — İZLEME LİSTESİ</span>'
        karar_class = "karar-orta"
        aksiyon = "İzleme listesine alınıp saha ekibi kapasitesine göre ikincil kontrol planlanabilir."
        kontrol_maddeleri = [
            "Önümüzdeki okuma döneminde tüketim ve hane profili sapmasının takibi",
            "Sayaç yaşı ve abone tipine göre rutin kontrol planına dahil etme",
        ]
    else:
        risk_etiket = "Düşük Risk"
        rozet_html = '<span class="rozet-dusuk">● DÜŞÜK RİSK — OLAĞAN PROFİL</span>'
        karar_class = "karar-dusuk"
        aksiyon = "Ek saha müdahalesi gerektirmeyen olağan tüketim profili."
        kontrol_maddeleri = [
            "Olağan periyodik sayaç okuma ve faturalandırma akışı sürdürülür",
        ]

    if tahmin_basildi:
        st.session_state["sorgu_gecmisi"].insert(
            0,
            {
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
                <div style="font-weight:700; color:#0f172a; margin-bottom:0.25rem;">Saha Aksiyon Önerisi</div>
                <div style="font-size:0.9rem; color:#334155;">{aksiyon}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Saha Kontrol Maddeleri**")
            for madde in kontrol_maddeleri:
                st.markdown(f"- {madde}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.plotly_chart(
        powerbi_tuketim_grafigi(
            float(girdiler.get("tuketim_kis_m3", 0)),
            float(girdiler.get("tuketim_yaz_m3", 0)),
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.plotly_chart(powerbi_etki_grafigi(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Modele gönderilen 5 temel değişken"):
        st.json({ETIKETLER.get(k, k): v for k, v in girdiler.items()})

if st.session_state["sorgu_gecmisi"]:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    gecmis_bas, gecmis_but = st.columns([3, 1])
    with gecmis_bas:
        st.markdown('<div class="panel-title">3. Sunum Sorgu Geçmişi</div>', unsafe_allow_html=True)
    with gecmis_but:
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
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="footer-bar">
        Karar Destek Modeli: Random Forest Sınıflandırıcı • Test F1: {test_f1:.4f} • Karar Eşiği: %{yuksek_esik*100:.0f} • Kurumsal Saha Önceliklendirme Ekranı
    </div>
    """,
    unsafe_allow_html=True,
)
