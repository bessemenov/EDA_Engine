# ---- main.py ----
# Streamlit tabanlı: CSV -> EDA -> Varsayım Kontrol -> Test Öner + Çalıştır
# Python 3.13 uyumlu. core/assumptions.py ve core/recommender.py ile birlikte çalışır.

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.assumptions import shapiro_test, levene_test, qq_points
from core.recommender import recommend_and_test


# -------------------- Sayfa Ayarı --------------------
st.set_page_config(page_title="EDA Engine", layout="wide")
st.title("📊 EDA Engine — CSV → EDA → Varsayım → Test")

st.caption("CSV yükle; önce EDA’yı gör, sonra varsayım kontrollerini yap, en sonda da uygun testi çalıştır.")


# -------------------- CSV Yükleme --------------------
file = st.file_uploader("CSV dosyasını yükle", type=["csv"])
if file is None:
    st.info("Devam etmek için bir .csv yükleyin.")
    st.stop()


@st.cache_data(show_spinner=False)
def read_csv_smart(uploaded) -> pd.DataFrame | str:
    """
    Ayırıcıyı otomatik saptamaya çalışır. (',', ';' vs.)
    """
    data = uploaded.read()
    buf = io.BytesIO(data)
    try:
        # sep=None + engine='python' -> ayırıcıyı infer eder
        return pd.read_csv(buf, sep=None, engine="python")
    except Exception:
        try:
            buf.seek(0)
            return pd.read_csv(buf)  # varsayılan ','
        except Exception as e:
            return f"Hata: {e}"


df = read_csv_smart(file)
if isinstance(df, str):
    st.error(df)
    st.stop()

# Tip setleri
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()


# -------------------- Sekmeler --------------------
tab_eda, tab_assump, tab_test = st.tabs(["🔍 EDA", "🧪 Varsayım Kontrol", "🧭 Test Öner + Çalıştır"])


# ==================== EDA TAB ====================
with tab_eda:
    st.subheader("Veri Önizlemesi")

    preview_mode = st.radio("Önizleme modu", ["İlk N satır", "Rastgele örnek"], horizontal=True)
    n_rows = st.slider("Gösterilecek satır sayısı", 5, 200, 20, key="preview_n")

    if preview_mode == "İlk N satır":
        st.dataframe(df.head(n_rows), use_container_width=True)
    else:
        # sample n > len(df) olmasın
        n_show = min(n_rows, len(df))
        st.dataframe(df.sample(n_show, random_state=42), use_container_width=True)

    # Hızlı meta özet
    st.subheader("📋 Veri Özeti")
    summary = {
        "Gözlem": int(len(df)),
        "Değişken": int(df.shape[1]),
        "Eksik (toplam)": int(df.isna().sum().sum()),
        "Sayısal Değişken": int(len(numeric_cols)),
        "Kategorik Değişken": int(len(cat_cols)),
    }
    st.json(summary)

    # Kategorik dağılım (özellikle 'group' gibi sütunlar için yararlı)
    st.subheader("🔤 Kategorik Değişken Dağılımı")
    if cat_cols:
        default_idx = cat_cols.index("group") if "group" in cat_cols else 0
        cat_pick = st.selectbox("Bir kategorik sütun seç", cat_cols, index=default_idx, key="eda_cat")
        counts = (
            df[cat_pick]
            .astype("category")
            .value_counts(dropna=False)
            .rename_axis(cat_pick)
            .reset_index(name="count")
        )
        st.dataframe(counts, use_container_width=True)
        st.plotly_chart(px.bar(counts, x=cat_pick, y="count", title=f"{cat_pick} dağılımı"), use_container_width=True)
        st.caption(f"`{cat_pick}` benzersiz seviye sayısı: {df[cat_pick].nunique(dropna=True)}")
    else:
        st.info("Kategorik sütun bulunamadı.")

    # Tanımlayıcı istatistikler (sayısal)
    st.subheader("📈 Tanımlayıcı İstatistikler (Sayısal)")
    desc = df.describe(include="number").T
    if not desc.empty:
        st.dataframe(desc, use_container_width=True)
    else:
        st.write("Sayısal değişken bulunamadı.")

    # Görseller
    if numeric_cols:
        st.subheader("📊 Görselleştirmeler (Sayısal)")
        num_pick = st.selectbox("Bir sayısal değişken seç:", numeric_cols, key="eda_num")
        st.plotly_chart(
            px.histogram(df, x=num_pick, nbins=30, title=f"{num_pick} — Histogram"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.box(df, y=num_pick, points="outliers", title=f"{num_pick} — Box Plot"),
            use_container_width=True,
        )

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, text_auto=True, title="Korelasyon Matrisi"), use_container_width=True)
    else:
        st.info("Görselleştirme için sayısal değişken yok.")


# ================= VARSAYIM KONTROL TAB =================
with tab_assump:
    st.subheader("🧪 Normallik & Varyans Homojenliği")

    if not numeric_cols:
        st.info("Varsayım analizi için en az bir sayısal değişken gerek.")
    else:
        y = st.selectbox("Normallik için sayısal değişken seç:", numeric_cols, key="assump_y")
        res = shapiro_test(df[y])
        st.markdown(f"**Normallik testi:** {res['name']}")
        st.write(f"İstatistik: {res['stat']}")
        st.write(f"p-değeri: {res['p']}")
        st.caption(res['note'] or "—")

        # QQ Plot (theoretical vs observed)
        qs, qx = qq_points(df[y])
        if qs is not None and qx is not None:
            qq_fig = go.Figure()
            qq_fig.add_trace(go.Scatter(x=qs, y=qx, mode="markers", name="Gözlenen vs Teorik"))
            # 45° referans çizgisi
            lo, hi = np.nanmin([qs.min(), qx.min()]), np.nanmax([qs.max(), qx.max()])
            qq_fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="45° çizgi"))
            qq_fig.update_layout(title="QQ Plot (Normal Teori)")
            st.plotly_chart(qq_fig, use_container_width=True)

        # Levene için grup seçimi (opsiyonel)
        if cat_cols:
            gcol = st.selectbox(
                "Varyans homojenliği için kategorik grup sütunu (opsiyonel):",
                ["(Yok)"] + cat_cols,
                key="assump_group",
            )
            if gcol != "(Yok)":
                levels = pd.Series(df[gcol]).dropna().unique()
                groups = [df.loc[df[gcol] == lev, y].dropna().values for lev in levels]
                lev = levene_test(*groups)
                st.markdown("**Levene (medyan merkezli):**")
                st.write(f"İstatistik: {lev['stat']}")
                st.write(f"p-değeri: {lev['p']}")
                st.caption(lev['note'] or "—")
        else:
            st.info("Levene için kategorik bir sütun bulunamadı.")


# ============== TEST ÖNER + ÇALIŞTIR TAB ==============
with tab_test:
    st.subheader("🧭 Otomatik Test Önerici ve Çalıştırma")

    # Kullanıcı seçimleri
    target_col = st.selectbox("Hedef (test edilecek değişken)", df.columns, key="test_target")
    group_col_opt = st.selectbox("Grup / Karşılaştırma sütunu (opsiyonel)", ["(Yok)"] + df.columns.tolist(), key="test_group")
    paired = st.checkbox("Eşleştirilmiş (paired) karşılaştırma", value=False)

    group_col = None if group_col_opt == "(Yok)" else group_col_opt

    # Bilgilendirme: grup seviyeleri
    if group_col is not None:
        uniq = pd.Series(df[group_col]).dropna().unique()
        st.caption(f"`{group_col}` benzersiz seviye sayısı: {len(uniq)} — İlk seviyeler: {list(uniq)[:10]}")

    if st.button("Öneri getir ve testi çalıştır"):
        out = recommend_and_test(df, target=target_col, group=group_col, paired=paired)

        # Sonuçların gösterimi (formatlama)
        st.markdown("### 🔎 Sonuç")
        st.write("**Öneri:**", out.get("recommendation") or "—")
        st.write("**Çalıştırılan Test:**", out.get("test") or "—")

        stat_val = out.get("stat")
        p_val = out.get("p")
        eff_val = out.get("effect")
        note_txt = out.get("note")

        stat_fmt = f"{stat_val:.6f}" if isinstance(stat_val, (int, float)) and stat_val is not None else str(stat_val)
        p_fmt = f"{p_val:.6g}" if isinstance(p_val, (int, float)) and p_val is not None else str(p_val)

        if eff_val is None or (isinstance(eff_val, float) and np.isnan(eff_val)):
            eff_fmt = "—"
        else:
            eff_fmt = f"{eff_val:.4f}" if isinstance(eff_val, (int, float)) else str(eff_val)

        st.write("**İstatistik:**", stat_fmt)
        st.write("**p-değeri:**", p_fmt)
        st.write("**Etki Büyüklüğü:**", eff_fmt)

        if note_txt:
            st.info(note_txt)

        # Akıllı yorum (bonus): normallik bozuksa nonparametrik sonuçlara öncelik verildiğini hatırlat
        st.caption("Not: Normallik/varians koşulları sağlanmıyorsa parametrik sonuçlar bilgilendirici kabul edilir; karar nonparametrik test üzerinden verilmelidir.")
