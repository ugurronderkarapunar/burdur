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
        .panel-title {
            font-size: 1.06rem;
            font-weight: 800;
            color: #0f172a !important;
            margin: 0 0 0.85rem 0;
        }
        .finans-card {
            background: #f8fafc !important;
            border: 1px solid #cbd5e1;
            border-left: 5px solid #0e7490;
            border-radius: 12px;
            padding: 0.9rem 1.05rem;
            margin-top: 0.6rem;
        }
        .finans-baslik {
            font-size: 0.82rem;
            font-weight: 800;
            color: #0f4c64 !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.25rem;
        }
        .finans-tutar {
            font-size: 1.38rem;
            font-weight: 800;
            color: #0f172a !important;
            line-height: 1.2;
        }
        .finans-not {
            font-size: 0.8rem;
            color: #475569 !important;
            margin-top: 0.25rem;
            line-height: 1.35;
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
        .kontrol-liste {
            list-style: none;
            padding-left: 0;
            margin: 0.35rem 0 0 0;
        }
        .kontrol-liste li {
            padding: 0.42rem 0.65rem;
            margin-bottom: 0.4rem;
            border-radius: 8px;
            background-color: #f8fafc !important;
            border: 1px solid #e2e8f0;
            font-size: 0.88rem;
            color: #0f172a !important;
            line-height: 1.4;
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
    st.session_state["tuketim_kis_m3"] = float(kis)
    st.session_state["hane_buyuklugu"] = int(hane)
    st.session_state["tuketim_yaz_m3"] = float(yaz)
    st.session_state["sayac_yasi"] = int(yas)

    kat_meta = ornek_degerler.get("abone_tipi", {})
    secenekler = [str(x) for x in kat_meta.get("secenekler", [])]
    if secenekler:
        for aday in abone_adaylari:
            if aday in secenekler:
                st.session_state["abone_tipi"] = aday
                break
        else:
            st.session_state["abone_tipi"] = secenekler[0]


def aylik_parasal_karsilik_hesapla(
    kis_m3: float,
    yaz_m3: float,
    hane_sayisi: int,
    olasilik: float,
    birim_tarife_tl: float,
    ek_kacak_hacim_m3: float,
) -> tuple[float, float, float]:
    beklenen_aylik_m3 = max(hane_sayisi, 1) * 4.2
    mevcut_ortalama_m3 = (float(kis_m3) + float(yaz_m3)) / 2.0
    tuketim_acigi_m3 = max(0.0, beklenen_aylik_m3 - mevcut_ortalama_m3)
    tahmini_kacak_m3 = round(tuketim_acigi_m3 + float(ek_kacak_hacim_m3), 2)
    aylik_tl = round(tahmini_kacak_m3 * float(birim_tarife_tl) * float(olasilik), 2)
    yillik_tl = round(aylik_tl * 12.0, 2)
    return aylik_tl, yillik_tl, tahmini_kacak_m3


def dinamik_saha_kontrol_listesi(
    seviye: str,
    ilce: str,
    abone_tipi: str,
    kis_m3: float,
    yaz_m3: float,
    hane_sayisi: int,
    sayac_yasi: int,
) -> list[str]:
    beklenen_m3 = max(hane_sayisi, 1) * 4.2
    ort_tuketim = (kis_m3 + yaz_m3) / 2.0
    uyum_notu = (
        f"Hane-tüketim uyumu: {hane_sayisi} kişilik hane için beklenen ~{beklenen_m3:.1f} m³ yerine ortalama {ort_tuketim:.1f} m³ tüketim görünüyor."
        if ort_tuketim < beklenen_m3
        else f"Hane-tüketim uyumu: {hane_sayisi} kişilik hane için ortalama {ort_tuketim:.1f} m³ tüketim kaydedilmiş."
    )
    sayac_notu = (
        f"Sayaç yaşı ({sayac_yasi} yıl): Yaşlı/yıpranmış sayaç ihtimaline karşı kalibrasyon ve mekanik durma kontrolü yapın."
        if sayac_yasi >= 10
        else f"Sayaç yaşı ({sayac_yasi} yıl): Sayaç yaşı görece yeni; bağlantı bütünlüğü ve mühür durumuna odaklanın."
    )

    if seviye == "yuksek":
        return [
            f"<b>Sayaç mührü ve gövde:</b> {ilce} / {abone_tipi} abonesinde mühür kopukluğu, ters sayaç veya cam/kadran müdahalesini kontrol edin.",
            "<b>By-pass ve hat kontrolü:</b> Sayaç öncesi/sonrası kaçak hat, by-pass bağlantısı veya şebeke dışı paralel çekiş olup olmadığını inceleyin.",
            f"<b>{uyum_notu}</b>",
            f"<b>{sayac_notu}</b>",
        ]
    if seviye == "orta":
        return [
            f"<b>Sayaç mührü kontrolü:</b> Rutin denetimde {ilce} abonesinin sayaç mühür ve bağlantı kelepçelerini gözle kontrol edin.",
            "<b>Hat güzergâhı gözlemi:</b> Bahçe, ahır/işlik veya ek yapı yönüne giden şüpheli boru hattı bulunup bulunmadığına bakın.",
            f"<b>{uyum_notu}</b>",
            f"<b>{sayac_notu}</b>",
        ]
    return [
        f"<b>Rutin sayaç okuma:</b> {ilce} abonesi için olağan periyodik okuma ve kayıt doğrulaması yapın.",
        "<b>Görsel mühür teyidi:</b> Okuma sırasında mühürde bariz hasar veya sızıntı varsa not düşün.",
        f"<b>{uyum_notu}</b>",
        f"<b>{sayac_notu}</b>",
    ]


def folium_ilce_haritasi(ilce_adi: str, olas_val: float, risk_etiket: str, aylik_tl: float) -> folium.Map:
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
    m = folium.Map(location=[lat, lon], zoom_start=11 if koord_bulundu else 9, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", attr="OpenStreetMap").add_to(m)

    popup_html = (
        f"<b>İlçe:</b> {ilce_adi}<br>"
        f"<b>Risk Skoru:</b> %{olas_val * 100:.1f}<br>"
        f"<b>Kademe:</b> {risk_etiket}<br>"
        f"<b>Tahmini Aylık Karşılık:</b> {aylik_tl:,.0f} TL/ay"
    )

    folium.Circle(location=[lat, lon], radius=3500, color=renk, fill=True, fill_color=renk, fill_opacity=0.22, weight=2).add_to(m)
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"{ilce_adi} • %{olas_val * 100:.1f} • {aylik_tl:,.0f} TL/ay",
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
        height=240,
        margin=dict(l=20, r=20, t=45, b=10),
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

if "birim_tarife_tl" not in st.session_state:
    st.session_state["birim_tarife_tl"] = 45.0
if "ek_kacak_hacim_m3" not in st.session_state:
    st.session_state["ek_kacak_hacim_m3"] = 5.0

satir = dict(varsayilan)
for col in ONEMLI_SUTUNLAR:
    satir[col] = st.session_state[col]
df_canli = pd.DataFrame([satir])[beklenen_sutunlar]

try:
    olasilik = float(pipe.predict_proba(df_canli)[0, 1])
except Exception as e:
    st.error(f"Tahmin sırasında hata oluştu: {e}")
    st.stop()

etiket, seviye = risk_kademesi(olasilik)
aktif_ilce = st.session_state["ilce"]
aktif_abone = st.session_state["abone_tipi"]
aktif_kis = st.session_state["tuketim_kis_m3"]
aktif_yaz = st.session_state["tuketim_yaz_m3"]
aktif_hane = st.session_state["hane_buyuklugu"]
aktif_yas = st.session_state["sayac_yasi"]

aylik_tl, yillik_tl, tahmini_kacak_m3 = aylik_parasal_karsilik_hesapla(
    kis_m3=aktif_kis,
    yaz_m3=aktif_yaz,
    hane_sayisi=aktif_hane,
    olasilik=olasilik,
    birim_tarife_tl=st.session_state["birim_tarife_tl"],
    ek_kacak_hacim_m3=st.session_state["ek_kacak_hacim_m3"],
)

kontrol_maddeleri = dinamik_saha_kontrol_listesi(
    seviye=seviye,
    ilce=aktif_ilce,
    abone_tipi=aktif_abone,
    kis_m3=aktif_kis,
    yaz_m3=aktif_yaz,
    hane_sayisi=aktif_hane,
    sayac_yasi=aktif_yas,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">BURDUR • SU ŞEBEKESİ</div>
        <div class="hero-title">Kaçak Su Risk Analiz ve Saha Öncelik Paneli</div>
        <p class="hero-sub">Girdiler değiştikçe kaçak şüphesi olasılığını, tahmini aylık parasal karşılığı ve saha kontrol maddelerini anlık günceller.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
for kol, etiket_k, deger_k, ipucu_k in [
    (k1, "Model Test F1", f"{test_f1:.3f}", "Test verisi üzerinde"),
    (k2, "Yüksek Risk Eşiği", f"%{yuksek_esik * 100:.0f}", "Bu değer ve üstü: saha kontrolü"),
    (k3, "Tahmini Aylık Karşılık", f"{aylik_tl:,.0f} TL/ay", f"~{tahmini_kacak_m3:.1f} m³/ay × {st.session_state['birim_tarife_tl']:.0f} TL × %{olasilik*100:.0f}"),
    (k4, "Yıllık Projeksiyon", f"{yillik_tl:,.0f} TL/yıl", "12 aylık tahmini karşılık"),
]:
    kol.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{etiket_k}</div>
            <div class="kpi-value">{deger_k}</div>
            <div class="kpi-hint">{ipucu_k}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

sol, sag = st.columns([1, 1.2], gap="large")

with sol:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Abone ve Tarife Parametreleri (Anlık)</div>', unsafe_allow_html=True)

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

        f1, f2 = st.columns(2)
        f1.number_input("Birim Su Tarifesi (TL/m³)", min_value=1.0, max_value=500.0, step=1.0, key="birim_tarife_tl")
        f2.number_input("Ek Kaçak Hacim Varsayımı (m³/ay)", min_value=0.0, max_value=200.0, step=1.0, key="ek_kacak_hacim_m3")

        gecmise_ekle = st.button("Mevcut senaryoyu geçmişe kaydet", type="primary", use_container_width=True)

if gecmise_ekle:
    st.session_state["sorgu_gecmisi"].append(
        {
            "İlçe": aktif_ilce,
            "Abone Tipi": aktif_abone,
            "Kış (m³)": aktif_kis,
            "Yaz (m³)": aktif_yaz,
            "Hane": aktif_hane,
            "Sayaç Yaşı": aktif_yas,
            "Risk (%)": round(olasilik * 100, 1),
            "Tahmini Aylık (TL)": round(aylik_tl, 0),
            "Kademe": etiket,
        }
    )

with sag:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Canlı Risk, Saha Kontrol Listesi ve Parasal Karşılık</div>', unsafe_allow_html=True)
        st.plotly_chart(
            powerbi_risk_kadran(olasilik, orta_esik, yuksek_esik),
            use_container_width=True,
        )
        st.markdown(
            f'<span class="rozet-{seviye}">{etiket}</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        baslik, metin = KARAR_METINLERI[seviye]
        st.markdown(
            f"""
            <div class="karar-{seviye}">
                <div class="karar-baslik">{baslik} ({aktif_ilce})</div>
                <div class="karar-metin">{metin}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(f"**Saha Kontrol Kontrol Listesi ({etiket})**")
        liste_html = "<ul class='kontrol-liste'>" + "".join(f"<li>✓ {m}</li>" for m in kontrol_maddeleri) + "</ul>"
        st.markdown(liste_html, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="finans-card">
                <div class="finans-baslik">Tahmini Aylık Parasal Karşılık</div>
                <div class="finans-tutar">{aylik_tl:,.0f} TL / ay <span style="font-size:0.9rem;font-weight:600;color:#475569;">({yillik_tl:,.0f} TL / yıl)</span></div>
                <div class="finans-not">Hane başına beklenen tüketime göre açığın ve ek kaçak hacmin ({tahmini_kacak_m3:.1f} m³/ay) birim tarife ({st.session_state['birim_tarife_tl']:.0f} TL/m³) ve model olasılığı (%{olasilik*100:.1f}) ile çarpımıdır.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        harita = folium_ilce_haritasi(aktif_ilce, olasilik, etiket, aylik_tl)
        st_folium(harita, height=290, use_container_width=True, returned_objects=[])

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
        st.markdown('<div class="panel-title">Kaydedilen Sorgu Geçmişi</div>', unsafe_allow_html=True)
        if st.session_state["sorgu_gecmisi"]:
            gecmis_df = pd.DataFrame(st.session_state["sorgu_gecmisi"][::-1])
            st.dataframe(gecmis_df, use_container_width=True, hide_index=True)
            if st.button("Geçmişi temizle"):
                st.session_state["sorgu_gecmisi"] = []
                st.rerun()
        else:
            st.caption("İstediğiniz an 'Mevcut senaryoyu geçmişe kaydet' ile karşılaştırma satırı ekleyebilirsiniz.")

st.markdown(
    '<div class="footer-bar">Bu panel karar destek amaçlıdır; parasal tutarlar tarife ve hacim varsayımlarına dayalı tahmini karşılıktır. Saha kararı yetkili personelce verilmelidir.</div>',
    unsafe_allow_html=True,
)
