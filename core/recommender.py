# ---- core/recommender.py ----
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from scipy import stats as ss
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================ Yardımcılar ============================

def _to_numeric(s: pd.Series) -> pd.Series:
    """Sayısal gibi görünen stringleri float'a çevirir; çevrilemeyenleri NaN yapar."""
    return pd.to_numeric(s, errors="coerce")


def _is_numeric(s: pd.Series) -> bool:
    return np.issubdtype(s.dropna().dtype, np.number)


def _unique_clean(s: pd.Series):
    return pd.Series(s).dropna().unique()


# --------------------------- Etki Büyüklükleri -----------------------

def cohens_d(x, y):
    x = pd.Series(x).dropna().astype(float).values
    y = pd.Series(y).dropna().astype(float).values
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    return float((x.mean() - y.mean()) / sp)


def cliffs_delta(x, y):
    x = pd.Series(x).dropna().astype(float).values
    y = pd.Series(y).dropna().astype(float).values
    if len(x) == 0 or len(y) == 0:
        return None
    greater = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    n = len(x) * len(y)
    return float((greater - less) / n)


def eta_squared_anova(groups):
    """One-way ANOVA için eta-squared (η²) ~ SS_between / SS_total"""
    # groups: list of 1D arrays
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((all_vals - grand_mean) ** 2).sum()
    if ss_total == 0:
        return None
    return float(ss_between / ss_total)


def epsilon_squared_kruskal(H, n, k):
    """Kruskal–Wallis için epsilon-squared (ε²)"""
    if n <= 1:
        return None
    return float((H - k + 1) / (n - k))


def cramer_v(table: pd.DataFrame):
    if not SCIPY_OK:
        return None
    chi2, p, dof, exp = ss.chi2_contingency(table, correction=False)
    n = table.values.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return float(v)


# ------------------------- Varsayım Kontrolleri ----------------------

def group_shapiro_norm_flags(groups):
    """
    Her grup için Shapiro uygular (n<=5000). n<3 ise 'undetermined'.
    Dönüş: list[dict] ve 'all_normal' özet bayrağı.
    """
    flags = []
    all_normal = True
    for g in groups:
        g = pd.Series(g).dropna().values
        if len(g) < 3 or not SCIPY_OK:
            flags.append({"n": len(g), "normal": None, "p": None})
            all_normal = False  # emin olamıyorsak parametrik önermekten kaçın
            continue
        stat, p = ss.shapiro(g) if len(g) <= 5000 else ss.normaltest(g)
        is_norm = (p >= 0.05)
        flags.append({"n": len(g), "normal": is_norm, "p": float(p)})
        if not is_norm:
            all_normal = False
    return flags, all_normal


def levene_equal_var(groups):
    """Levene (medyan merkezli) ile varyans homojenliği; SciPy yoksa None döner."""
    if not SCIPY_OK:
        return None, None
    clean = [pd.Series(g).dropna().values for g in groups if len(pd.Series(g).dropna()) > 0]
    if len(clean) < 2:
        return None, None
    st, p = ss.levene(*clean, center="median")
    return float(st), float(p)


# ============================== Ana API ==============================

def recommend_and_test(df: pd.DataFrame, target: str, group: str | None = None, paired: bool = False):
    """
    Girdi kombinasyonuna göre uygun istatistik testi önerir ve mümkünse çalıştırır.
    Dönen sözlük anahtarları:
      - recommendation, test, stat, p, effect, note
    """
    out = {"recommendation": "", "test": "", "stat": None, "p": None, "effect": None, "note": ""}

    # --- Ön kontroller ve tip düzeltmeleri
    if target not in df.columns:
        out["recommendation"] = "Hedef sütun bulunamadı."
        return out

    # Hedefi sayısallaştır (mümkünse)
    df = df.copy()
    df[target] = _to_numeric(df[target]) if not _is_numeric(df[target]) else df[target]

    # Grup yoksa: korelasyon/karşılaştırma yapılandırılamaz
    if group is None:
        out["recommendation"] = "Grup sütunu seçilirse karşılaştırma testleri yapılabilir."
        return out
    if group not in df.columns:
        out["recommendation"] = "Grup sütunu bulunamadı."
        return out

    gser = df[group]

    # Sayısal ama az unique değerli ise kategorik kabul et
    if _is_numeric(gser) and gser.nunique(dropna=True) < 15:
        gser = gser.astype("category")
    # Metinse zaten kategorik
    if gser.dtype.name not in ["category", "object"] and not _is_numeric(gser):
        # Güvenlik: bilinmeyen tipleri category yap
        gser = gser.astype("category")

    # ================== İki kategorik (Ki-kare / Fisher) ==================
    if (not _is_numeric(df[target])) and (gser.dtype.name in ["category", "object"]):
        tab = pd.crosstab(df[target], gser)
        if SCIPY_OK:
            if (tab.values < 5).sum() > 0 and tab.shape == (2, 2):
                stat, p = ss.fisher_exact(tab.values)
                out.update({
                    "recommendation": "İki kategorik değişken: Beklenen frekans düşük → Fisher’s exact.",
                    "test": "Fisher’s exact",
                    "stat": float(stat), "p": float(p),
                    "effect": None
                })
            else:
                chi2, p, dof, exp = ss.chi2_contingency(tab)
                out.update({
                    "recommendation": "İki kategorik değişken: Ki-kare uygun.",
                    "test": "Chi-square",
                    "stat": float(chi2), "p": float(p),
                    "effect": cramer_v(tab)
                })
        else:
            out["recommendation"] = "İki kategorik değişken: Ki-kare / Fisher (SciPy yok)."
        return out

    # ================== Sayısal hedef + Kategorik grup ====================
    if _is_numeric(df[target]) and (gser.dtype.name in ["category", "object"]):
        levels = _unique_clean(gser)
        groups = [df.loc[gser == lev, target].dropna().values for lev in levels]
        k = len(groups)

        if k < 2:
            out["recommendation"] = "Sadece bir grup var → karşılaştırma yapılamaz."
            return out

        # Varsayım kontrolleri
        shapiro_flags, all_normal = group_shapiro_norm_flags(groups)
        lev_stat, lev_p = levene_equal_var(groups)
        equal_var = (lev_p is not None and lev_p >= 0.05)

        # ---- İki grup
        if k == 2:
            if SCIPY_OK:
                # Parametrik: t-test (Levene'ye göre equal_var/ Welch)
                t_stat, t_p = ss.ttest_ind(groups[0], groups[1], equal_var=equal_var)
                d = cohens_d(groups[0], groups[1])
                # Nonparametrik: Mann–Whitney
                u_stat, u_p = ss.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                # Karar metni
                reason = []
                if not all_normal:
                    reason.append("Normallik bazı gruplarda p<0.05")
                if lev_p is not None and lev_p < 0.05:
                    reason.append("Varyans homojenliği sağlanmadı (Levene p<0.05)")
                reason_txt = "; ".join(reason) if reason else "Varsayımlar makul"

                # Normallik/Levene bozulduysa nonparametrik öncelik
                if (not all_normal) or (lev_p is not None and lev_p < 0.05):
                    out.update({
                        "recommendation": f"İki bağımsız grup: Nonparametrik önerilir (sebep: {reason_txt}). Alternatif bilgi için t-test (Welch/Standart) da verildi.",
                        "test": "Mann–Whitney U",
                        "stat": float(u_stat), "p": float(u_p),
                        "effect": cliffs_delta(groups[0], groups[1]),
                        "note": f"t-test stat={t_stat:.3f}, p={t_p:.3f}, Cohen's d={d if d is not None else 'NA'}"
                    })
                else:
                    welch = not equal_var
                    out.update({
                        "recommendation": f"İki bağımsız grup: Parametrik uygun (normallik & homojenlik). {'Welch' if welch else 'Eş varyanslı'} t-test uygulanır.",
                        "test": "t-test (independent, Welch)" if welch else "t-test (independent)",
                        "stat": float(t_stat), "p": float(t_p),
                        "effect": d,
                        "note": f"Mann–Whitney U={u_stat:.3f}, p={u_p:.3f}"
                    })
            else:
                out["recommendation"] = "İki bağımsız grup: t-test / Mann–Whitney (SciPy yok)."
            return out

        # ---- 3+ grup
        if k >= 3:
            if SCIPY_OK:
                # Parametrik ANOVA
                f_stat, f_p = ss.f_oneway(*groups)
                eta2 = eta_squared_anova(groups)
                # Nonparametrik Kruskal
                H, p_kw = ss.kruskal(*groups)
                eps2 = epsilon_squared_kruskal(H, n=sum(len(g) for g in groups), k=k)

                reason = []
                if not all_normal:
                    reason.append("Normallik bazı gruplarda p<0.05")
                if lev_p is not None and lev_p < 0.05:
                    reason.append("Varyans homojenliği sağlanmadı (Levene p<0.05)")
                reason_txt = "; ".join(reason) if reason else "Varsayımlar makul"

                if (not all_normal) or (lev_p is not None and lev_p < 0.05):
                    # Nonparametrik öncelik
                    out.update({
                        "recommendation": f"{k} bağımsız grup: Nonparametrik önerilir (sebep: {reason_txt}). ANOVA bilgisel amaçlı da hesaplandı.",
                        "test": "Kruskal–Wallis",
                        "stat": float(H), "p": float(p_kw),
                        "effect": eps2,
                        "note": f"ANOVA F={f_stat:.3f}, p={f_p:.3f}, eta²={eta2 if eta2 is not None else 'NA'}"
                    })
                else:
                    out.update({
                        "recommendation": f"{k} bağımsız grup: Parametrik koşullar makul → ANOVA önerilir.",
                        "test": "One-way ANOVA",
                        "stat": float(f_stat), "p": float(f_p),
                        "effect": eta2,
                        "note": f"Kruskal–Wallis H={H:.3f}, p={p_kw:.3f}, ε²={eps2 if eps2 is not None else 'NA'}"
                    })
            else:
                out["recommendation"] = f"{k} bağımsız grup: ANOVA / Kruskal–Wallis (SciPy yok)."
            return out

    # ================== Sayısal hedef + Sayısal grup (korelasyon) ==========
    if _is_numeric(df[target]) and _is_numeric(gser):
        if not SCIPY_OK:
            out["recommendation"] = "İki sayısal değişken: Pearson / Spearman (SciPy yok)."
            return out
        # Pearson & Spearman birlikte verelim
        x = df[target].dropna().values
        y = pd.Series(gser).dropna().values
        common_n = min(len(x), len(y))
        if common_n < 3:
            out["recommendation"] = "Korelasyon için yeterli gözlem yok (n<3)."
            return out
        r, p = ss.pearsonr(x[:common_n], y[:common_n])
        rho, p_rho = ss.spearmanr(x, y)
        out.update({
            "recommendation": "İki sayısal değişken: Pearson (parametrik) & Spearman (nonparametrik).",
            "test": "Pearson correlation",
            "stat": float(r), "p": float(p),
            "effect": None,
            "note": f"Spearman ρ={rho:.3f}, p={p_rho:.3f}"
        })
        return out

    # ================== Eşleştirilmiş durum (UI genişletmesi gerek) ========
    if paired and _is_numeric(df[target]):
        out["recommendation"] = "Eşleştirilmiş karşılaştırma: paired t-test / Wilcoxon (UI'da ikinci ölçüm seçilerek)."
        return out

    # ================== Varsayılan ==================
    out["recommendation"] = "Seçim kombinasyonuna göre uygun test belirlenemedi."
    return out
