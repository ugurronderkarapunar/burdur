import folium
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

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
    popup_html = f"<b>İlçe:</b> {ilce_adi}<br><b>Risk Skoru:</b> %{olas_val * 100:.1f}<br><b>Kademe:</b> {risk_etiket}"

    folium.Circle(location=[lat, lon], radius=3500, color=renk, fill=True, fill_color=renk, fill_opacity=0.22, weight=2).add_to(m)
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=240),
        tooltip=f"{ilce_adi} • %{olas_val * 100:.1f} ({risk_etiket})",
        icon=folium.Icon(color="red" if olas_val >= yuksek_esik else ("orange" if olas_val >= orta_esik else "green"), icon="tint", prefix="fa"),
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
                    "thickness": 0.8,
                    "value": yuksek_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans"},
    )
    return fig


def risk_kademesi(p: float):
    if p >= yuksek_esik:
        return "Yüksek Risk", "yuksek"
    if p >= orta_esik:
        return "Orta Risk", "orta"
    return "Düşük Risk", "dusuk"


KARAR_METINLERI = {
    "yuksek": ("Öncelikli saha kontrolü", "Abone için sayaç ve tesisat kontrolü öncelikli planlanmalı. Kaçak kullanım şüphesi yüksek."),
    "orta": ("İzleme listesine al", "Tüketim davranışı izlenmeli. Sonraki okuma döneminde sapma sürerse saha kontrolüne alınabilir."),
    "dusuk": ("Rutin takip", "Mevcut verilere göre belirgin bir kaçak şüphesi yok. Rutin sayaç okuma yeterli."),
}

# ------------------------------------------------------------------ #
# Başlangıç değerleri
# ------------------------------------------------------------------ #
varsayilan = varsayilan_satir_olustur()

ilce_secenekleri = [str(x) for x in ornek_degerler.get("ilce", {}).get("secenekler", [])] or list(ILCE_KOORDINATLARI.keys())
abone_secenekleri = [str(x) for x in ornek_degerler.get("abone_tipi", {}).get("secenekler", [])] or ["Konut"]

for col in ONEMLI_SUTUNLAR:
    if col not in st.session_state:
        st.session_state[col] = varsayilan.get(col, 0)

if st.session_state["ilce"] not in ilce_secenekleri:
    st.session_state["ilce"] = ilce_secenekleri[0]
if st.session_state["abone_tipi"] not in abone_secenekleri:
    st.session_state["abone_tipi"] = abone_secenekleri[0]

st.session_state["tuketim_kis_m3"] = float(st.session_state["tuketim_kis_m3"])
st.session_state["tuketim_yaz_m3"] = float(st.session_state["tuketim_yaz_m3"])
st.session_state["hane_buyuklugu"] = int(st.session_state["hane_buyuklugu"])
st.session_state["sayac_yasi"] = int(st.session_state["sayac_yasi"])

if "son_sonuc" not in st.session_state:
    st.session_state["son_sonuc"] = None

# ------------------------------------------------------------------ #
# Başlık
# ------------------------------------------------------------------ #
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">BURDUR • SU ŞEBEKESİ</div>
        <div class="hero-title">Kaçak Su Risk Analiz ve Saha Öncelik Paneli</div>
        <p class="hero-sub">Abone tüketim davranışından kaçak şüphesi olasılığını hesaplar ve saha ekibi için öncelik önerir.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3 = st.columns(3)
for kol, etiket, deger, ipucu in [
    (k1, "Model Test F1", f"{test_f1:.3f}", "Test verisi üzerinde"),
    (k2, "Yüksek Risk Eşiği", f"%{yuksek_esik * 100:.0f}", "Bu değer ve üstü: saha kontrolü"),
    (k3, "Orta Risk Eşiği", f"%{orta_esik * 100:.0f}", "Bu değer ve üstü: izleme"),
]:
    kol.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{etiket}</div>
            <div class="kpi-value">{deger}</div>
            <div class="kpi-hint">{ipucu}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

sol, sag = st.columns([1, 1.2], gap="large")

# ------------------------------------------------------------------ #
# Sol: girdi paneli
# ------------------------------------------------------------------ #
with sol:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Abone Bilgileri</div>', unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        b1.button(
            "Normal abone örneği",
            use_container_width=True,
            on_click=senaryo_uygula,
            args=(REFERANS_TUKETIM["normal_kis"], 4, REFERANS_TUKETIM["normal_yaz"], 5, ["Konut", "Mesken"]),
        )
        b2.button(
            "Riskli abone örneği",
            use_container_width=True,
            on_click=senaryo_uygula,
            args=(REFERANS_TUKETIM["riskli_kis"], 1, REFERANS_TUKETIM["riskli_yaz"], 15, ["Tarım-Hayvancılık", "Tarım", "Hayvancılık"]),
        )

        st.selectbox(ETIKETLER["ilce"], ilce_secenekleri, key="ilce")
        st.selectbox(ETIKETLER["abone_tipi"], abone_secenekleri, key="abone_tipi")

        c1, c2 = st.columns(2)
        c1.number_input(ETIKETLER["tuketim_kis_m3"], min_value=0.0, max_value=500.0, step=0.5, key="tuketim_kis_m3")
        c2.number_input(ETIKETLER["tuketim_yaz_m3"], min_value=0.0, max_value=500.0, step=0.5, key="tuketim_yaz_m3")

        c3, c4 = st.columns(2)
        c3.number_input(ETIKETLER["hane_buyuklugu"], min_value=1, max_value=30, step=1, key="hane_buyuklugu")
        c4.number_input(ETIKETLER["sayac_yasi"], min_value=0, max_value=60, step=1, key="sayac_yasi")

        hesapla = st.button("Riski Hesapla", type="primary", use_container_width=True)

    if hesapla:
        satir = dict(varsayilan)
        for col in ONEMLI_SUTUNLAR:
            satir[col] = st.session_state[col]
        df = pd.DataFrame([satir])[beklenen_sutunlar]

        try:
            olasilik = float(pipe.predict_proba(df)[0, 1])
        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")
            st.stop()

        etiket, seviye = risk_kademesi(olasilik)
        sonuc = {
            "ilce": st.session_state["ilce"],
            "olasilik": olasilik,
            "etiket": etiket,
            "seviye": seviye,
        }
        st.session_state["son_sonuc"] = sonuc
        st.session_state["sorgu_gecmisi"].append(
            {
                "İlçe": st.session_state["ilce"],
                "Abone Tipi": st.session_state["abone_tipi"],
                "Kış (m³)": st.session_state["tuketim_kis_m3"],
                "Yaz (m³)": st.session_state["tuketim_yaz_m3"],
                "Hane": st.session_state["hane_buyuklugu"],
                "Sayaç Yaşı": st.session_state["sayac_yasi"],
                "Risk (%)": round(olasilik * 100, 1),
                "Kademe": etiket,
            }
        )

# ------------------------------------------------------------------ #
# Sağ: sonuç paneli
# ------------------------------------------------------------------ #
with sag:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Risk Sonucu</div>', unsafe_allow_html=True)
        sonuc = st.session_state["son_sonuc"]

        if sonuc is None:
            st.info("Sol taraftan bilgileri girip 'Riski Hesapla' düğmesine bas.")
        else:
            st.plotly_chart(
                powerbi_risk_kadran(sonuc["olasilik"], orta_esik, yuksek_esik),
                use_container_width=True,
            )
            st.markdown(
                f'<span class="rozet-{sonuc["seviye"]}">{sonuc["etiket"]}</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            baslik, metin = KARAR_METINLERI[sonuc["seviye"]]
            st.markdown(
                f"""
                <div class="karar-{sonuc['seviye']}">
                    <div class="karar-baslik">{baslik}</div>
                    <div class="karar-metin">{metin}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            harita = folium_ilce_haritasi(sonuc["ilce"], sonuc["olasilik"], sonuc["etiket"])
            st_folium(harita, height=300, use_container_width=True, returned_objects=[])

# ------------------------------------------------------------------ #
# Alt: model etkisi + geçmiş
# ------------------------------------------------------------------ #
alt1, alt2 = st.columns([1, 1.2], gap="large")

with alt1:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Modelde En Etkili Özellikler</div>', unsafe_allow_html=True)
        etiketler_ozet = [x[0] for x in OZELLIK_ETKI_OZETI][::-1]
        degerler_ozet = [x[1] for x in OZELLIK_ETKI_OZETI][::-1]
        fig_bar = go.Figure(
            go.Bar(x=degerler_ozet, y=etiketler_ozet, orientation="h", marker_color="#0e7490")
        )
        fig_bar.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Plus Jakarta Sans", "color": "#0f172a"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with alt2:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Sorgu Geçmişi</div>', unsafe_allow_html=True)
        if st.session_state["sorgu_gecmisi"]:
            gecmis_df = pd.DataFrame(st.session_state["sorgu_gecmisi"][::-1])
            st.dataframe(gecmis_df, use_container_width=True, hide_index=True)
            if st.button("Geçmişi temizle"):
                st.session_state["sorgu_gecmisi"] = []
                st.rerun()
        else:
            st.caption("Henüz sorgu yapılmadı.")

st.markdown(
    '<div class="footer-bar">Bu panel karar destek amaçlıdır; sentetik veriyle eğitilmiş bir modele dayanır. Saha kararı yetkili personelce verilmelidir.</div>',
    unsafe_allow_html=True,
)
