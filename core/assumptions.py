# ---- core/assumptions.py ----
from __future__ import annotations
import numpy as np
import pandas as pd

# SciPy varsa kullan, yoksa "not_available" döndür.
try:
    from scipy import stats as ss
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def drop_na_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").dropna().values

def shapiro_test(s: pd.Series, max_n: int = 5000):
    """
    n<=5000 ise Shapiro; daha büyüklerde adil değil -> D'Agostino (K2) deneriz.
    SciPy yoksa not_available döndürürüz.
    """
    if not SCIPY_OK:
        return {"name": "Shapiro", "stat": None, "p": None, "note": "SciPy not available"}
    x = drop_na_numeric(s)
    if len(x) < 3:
        return {"name": "Shapiro", "stat": None, "p": None, "note": "n<3"}
    if len(x) <= max_n:
        st, p = ss.shapiro(x)
        return {"name": "Shapiro", "stat": float(st), "p": float(p), "note": ""}
    # büyük n: K2
    k2, p = ss.normaltest(x)
    return {"name": "D’Agostino K²", "stat": float(k2), "p": float(p), "note": "n>5000"}

def levene_test(*groups):
    """
    Varyans homojenliği: Levene (medyan merkezli). SciPy yoksa not_available.
    """
    if not SCIPY_OK:
        return {"name": "Levene", "stat": None, "p": None, "note": "SciPy not available"}
    clean = [drop_na_numeric(pd.Series(g)) for g in groups if len(pd.Series(g).dropna()) > 0]
    if len(clean) < 2:
        return {"name": "Levene", "stat": None, "p": None, "note": "group<2"}
    st, p = ss.levene(*clean, center='median')
    return {"name": "Levene", "stat": float(st), "p": float(p), "note": ""}

def qq_points(s: pd.Series, q_count: int = 1000):
    """
    QQ grafiği için teorik (normal) kuantiller ve gözlenen kuantilleri döndür.
    SciPy yoksa kendi normal kuantillerimizi üretiriz.
    """
    x = drop_na_numeric(s)
    if len(x) == 0:
        return None, None
    x = np.sort(x)
    probs = np.linspace(0.001, 0.999, min(q_count, len(x)))
    # teorik normal kuantiller
    if SCIPY_OK:
        q_theory = ss.norm.ppf(probs)
    else:
        # yaklaşık: inverse CDF yerine std normaldan geniş örnekleme ve quantile
        z = np.random.standard_normal(100000)
        q_theory = np.quantile(z, probs)
    q_sample = np.quantile(x, probs)
    return q_theory, q_sample
