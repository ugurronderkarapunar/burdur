import io
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


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Saha_Listesi") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buf.getvalue()


def ornek_abone_dosyasi_olustur(ilce_liste: list[str], abone_liste: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    ilceler = rng.choice(ilce_liste, size=n)
    tipler = rng.choice(abone_liste, size=n)
    haneler = rng.integers(1, 7, size=n)
    yaslar = rng.integers(1, 22, size=n)
    kis_tuk = np.round(rng.uniform(1.5, 18.0, size=n), 1)
    yaz_tuk = np.round(kis_tuk + rng.uniform(-1.0, 9.5, size=n), 1)
    yaz_tuk = np.maximum(yaz_tuk, 1.0)

    return pd.DataFrame(
        {
            "abone_no": [f"ABN-{10000 + i}" for i in range(n)],
            "mahalle": [f"{ilc} Mah. {(i % 8) + 1}" for i, ilc in enumerate(ilceler)],
            "ilce": ilceler,
            "abone_tipi": tipler,
            "hane_buyuklugu": haneler,
            "sayac_yasi": yaslar,
            "tuketim_kis_m3": kis_tuk,
            "tuketim_yaz_m3": yaz_tuk,
        }
    )


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

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">BURDUR • SU ŞEBEKESİ</div>
        <div class="hero-title">Kaçak Su Risk Analiz ve Saha Öncelik Paneli</div>
        <p class="hero-sub">Sentetik veriyle hazırlandı, pilotla doğrulanacak — tekil abone risk analizi ve haftalık saha kontrol planlaması.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "**Pilot ve Veri Notu:** Bu model ve panel **sentetik veriyle hazırlandı, pilotla doğrulanacak**tır. "
    "Parasal geri kazanım ve saha kapasitesi hesapları karar destek amaçlı varsayımlara dayanır; ilk saha pilotu bulgularıyla kalibre edilecektir."
)

tab_tekil, tab_toplu = st.tabs([
    "🔍 Tekil Abone Sorgulama",
    "📋 Toplu Yükleme & Haftalık Saha Planı",
])

with tab_tekil:
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

with tab_toplu:
    with st.container(border=True):
        st.markdown('<div class="panel-title">1. Tüm Abone Dosyasını Yükle ve Haftalık Saha Listesi Oluştur</div>', unsafe_allow_html=True)
        st.caption(
            "Beklenen sütunlar: `ilce`, `tuketim_kis_m3`, `hane_buyuklugu`, `tuketim_yaz_m3`, `sayac_yasi`, `abone_tipi` "
            "(isteğe bağlı `abone_no`, `mahalle`). Eksik sütunlar modelin varsayılan/medyan değerleriyle doldurulur."
        )

        ornek_df = ornek_abone_dosyasi_olustur(ilce_secenekleri, abone_secenekleri)
        st.download_button(
            label="📥 Örnek Abone Listesi Şablonunu İndir (.xlsx)",
            data=df_to_excel_bytes(ornek_df, sheet_name="Abone_Verisi"),
            file_name="ornek_abone_listesi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        yuklenen_dosya = st.file_uploader(
            "Abone dosyasını yükleyin (Excel .xlsx veya .csv)",
            type=["xlsx", "csv"],
            key="toplu_abone_dosyasi",
        )

        if yuklenen_dosya is None:
            st.info(
                "Haftalık saha listesini ve net kazanç aralığını görmek için yukarıdan bir Excel/CSV dosyası yükleyin "
                "veya **Örnek Abone Listesi Şablonunu İndir** ile 120 aboneli test dosyasını deneyin."
            )
        else:
            try:
                if yuklenen_dosya.name.lower().endswith(".csv"):
                    df_ham = pd.read_csv(yuklenen_dosya)
                else:
                    df_ham = pd.read_excel(yuklenen_dosya)
            except Exception as e:
                st.error(f"Dosya okunurken hata oluştu: {e}")
                df_ham = None

            if df_ham is not None and not df_ham.empty:
                df_calisma = df_ham.copy()

                if "ilce" not in df_calisma.columns:
                    df_calisma["ilce"] = ilce_secenekleri[0]
                else:
                    df_calisma["ilce"] = df_calisma["ilce"].fillna(ilce_secenekleri[0]).astype(str)

                if "abone_tipi" not in df_calisma.columns:
                    df_calisma["abone_tipi"] = abone_secenekleri[0]
                else:
                    df_calisma["abone_tipi"] = df_calisma["abone_tipi"].fillna(abone_secenekleri[0]).astype(str)

                for num_col, varsay_val in [
                    ("tuketim_kis_m3", float(varsayilan.get("tuketim_kis_m3", 10.0))),
                    ("tuketim_yaz_m3", float(varsayilan.get("tuketim_yaz_m3", 14.0))),
                    ("hane_buyuklugu", int(varsayilan.get("hane_buyuklugu", 3))),
                    ("sayac_yasi", int(varsayilan.get("sayac_yasi", 6))),
                ]:
                    if num_col not in df_calisma.columns:
                        df_calisma[num_col] = varsay_val
                    else:
                        df_calisma[num_col] = pd.to_numeric(df_calisma[num_col], errors="coerce").fillna(varsay_val)

                df_model_girdi = pd.DataFrame(index=df_calisma.index)
                for col in beklenen_sutunlar:
                    if col in df_calisma.columns:
                        df_model_girdi[col] = df_calisma[col]
                    else:
                        df_model_girdi[col] = varsayilan.get(col, 0)

                try:
                    olas_dizi = pipe.predict_proba(df_model_girdi)[:, 1].astype(float)
                except Exception as e:
                    st.error(f"Toplu tahmin sırasında hata oluştu: {e}")
                    olas_dizi = None

                if olas_dizi is not None:
                    df_sonuc = df_calisma.copy()
                    df_sonuc["risk_olasiligi"] = olas_dizi
                    df_sonuc["risk_skoru_%"] = np.round(olas_dizi * 100.0, 1)

                    kademe_etiketleri = [risk_kademesi(float(p_val))[0] for p_val in olas_dizi]
                    df_sonuc["risk_kademesi"] = kademe_etiketleri

                    df_sirali = df_sonuc.sort_values(by="risk_olasiligi", ascending=False).reset_index(drop=True)
                    df_sirali.insert(0, "saha_sirasi", np.arange(1, len(df_sirali) + 1))

                    st.divider()
                    st.markdown('<div class="panel-title">Saha Kapasitesi ve Maliyet Parametreleri</div>', unsafe_allow_html=True)

                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        gunluk_kapasite = st.slider(
                            "Günlük Kontrol Kapasitesi (Abone/Gün)",
                            min_value=1,
                            max_value=50,
                            value=10,
                            step=1,
                            key="saha_gunluk_kapasite",
                        )
                    with p2:
                        haftalik_gun = st.number_input(
                            "Haftalık Çalışma Günü",
                            min_value=1,
                            max_value=7,
                            value=5,
                            step=1,
                            key="saha_haftalik_gun",
                        )
                    with p3:
                        gunluk_ekip_maliyeti = st.number_input(
                            "Günlük Ekip Maliyeti (TL/Gün)",
                            min_value=500.0,
                            max_value=100000.0,
                            value=4500.0,
                            step=250.0,
                            key="saha_gunluk_maliyet",
                        )
                    with p4:
                        temkinli_oran_pct = st.slider(
                            "Temkinli Tahsilat / Gerçekleşme (%)",
                            min_value=10,
                            max_value=100,
                            value=50,
                            step=5,
                            key="saha_temkinli_oran",
                        )

                    haftalik_hedef = min(int(gunluk_kapasite) * int(haftalik_gun), len(df_sirali))
                    haftalik_liste = df_sirali.head(haftalik_hedef).copy()

                    aylik_kazanim_list = []
                    yillik_kazanim_list = []
                    for _, row in haftalik_liste.iterrows():
                        a_tl, y_tl, _ = aylik_parasal_karsilik_hesapla(
                            kis_m3=float(row["tuketim_kis_m3"]),
                            yaz_m3=float(row["tuketim_yaz_m3"]),
                            hane_sayisi=int(row["hane_buyuklugu"]),
                            olasilik=float(row["risk_olasiligi"]),
                            birim_tarife_tl=float(st.session_state["birim_tarife_tl"]),
                            ek_kacak_hacim_m3=float(st.session_state["ek_kacak_hacim_m3"]),
                        )
                        aylik_kazanim_list.append(a_tl)
                        yillik_kazanim_list.append(y_tl)

                    haftalik_liste["tahmini_aylik_kazanim_tl"] = aylik_kazanim_list
                    haftalik_liste["tahmini_yillik_kazanim_tl"] = yillik_kazanim_list

                    iyimser_aylik_geri_kazanim = float(haftalik_liste["tahmini_aylik_kazanim_tl"].sum())
                    temkinli_aylik_geri_kazanim = iyimser_aylik_geri_kazanim * (float(temkinli_oran_pct) / 100.0)

                    haftalik_saha_maliyeti = float(haftalik_gun) * float(gunluk_ekip_maliyeti)
                    aylik_saha_maliyeti = haftalik_saha_maliyeti * 4.0

                    net_temkinli_aylik = temkinli_aylik_geri_kazanim - aylık_saha_maliyeti
                    net_iyimser_aylik = iyimser_aylik_geri_kazanim - aylık_saha_maliyeti

                    st.divider()
                    st.markdown(
                        f'<div class="panel-title">2. Net Kazanç ve Geri Dönüş Özeti (İlk {haftalik_hedef} Abone)</div>',
                        unsafe_allow_html=True,
                    )

                    n1, n2, n3, n4 = st.columns(4)
                    for kol, etiket_n, deger_n, ipucu_n in [
                        (
                            n1,
                            "Haftalık / 4-Hafta Saha Maliyeti",
                            f"{haftalik_saha_maliyeti:,.0f} TL / hf",
                            f"4 haftada ~{aylik_saha_maliyeti:,.0f} TL ({haftalik_gun} gün × {gunluk_ekip_maliyeti:,.0f} TL)",
                        ),
                        (
                            n2,
                            "Beklenen Aylık Geri Kazanım Aralığı",
                            f"{temkinli_aylik_geri_kazanim:,.0f} – {iyimser_aylik_geri_kazanim:,.0f} TL/ay",
                            f"Temkinli (%{temkinli_oran_pct}) – İyimser (%100)",
                        ),
                        (
                            n3,
                            f"Net Kazanç (Temkinli %{temkinli_oran_pct})",
                            f"{net_temkinli_aylik:,.0f} TL/ay",
                            "Aylık temkinli geri kazanım − 4 haftalık maliyet",
                        ),
                        (
                            n4,
                            "Net Kazanç (İyimser %100)",
                            f"{net_iyimser_aylik:,.0f} TL/ay",
                            "Aylık iyimser geri kazanım − 4 haftalık maliyet",
                        ),
                    ]:
                        kol.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">{etiket_n}</div>
                                <div class="kpi-value">{deger_n}</div>
                                <div class="kpi-hint">{ipucu_n}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.caption(
                        f"Varsayım notu: İlk {haftalik_hedef} abone kontrolü günde {gunluk_kapasite} abone × {haftalik_gun} gün kapasiteyle planlanır. "
                        f"Sentetik veriyle hazırlandı, pilotla doğrulanacak; geri kazanım aralığı {st.session_state['birim_tarife_tl']:.0f} TL/m³ tarife ve "
                        f"%{temkinli_oran_pct} temkinli / %100 iyimser gerçekleşme senaryolarına dayanır."
                    )

                    st.divider()
                    st.markdown(
                        f'<div class="panel-title">Bu Hafta Kontrol Edilecek İlk {haftalik_hedef} Abone (Toplam {len(df_sirali)} Abone İçinden)</div>',
                        unsafe_allow_html=True,
                    )

                    excel_liste = haftalik_liste.copy()
                    excel_liste["saha_sonucu"] = ""
                    excel_liste["not"] = ""

                    st.download_button(
                        label=f"⬇️ Haftalık Saha Listesini Excel İndir (İlk {haftalik_hedef} Abone)",
                        data=df_to_excel_bytes(excel_liste, sheet_name="Haftalik_Saha_Listesi"),
                        file_name=f"haftalik_saha_listesi_{haftalik_hedef}_abone.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

                    st.dataframe(haftalik_liste, use_container_width=True, hide_index=True, height=380)

st.markdown(
    '<div class="footer-bar">Sentetik veriyle hazırlandı, pilotla doğrulanacak. Bu panel karar destek amaçlıdır; parasal tutarlar tarife ve hacim varsayımlarına dayalı tahmini karşılıktır. Saha kararı yetkili personelce verilmelidir.</div>',
    unsafe_allow_html=True,
)
