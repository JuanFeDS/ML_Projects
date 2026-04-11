"""
Tests estadísticos de asociación entre variables y el target.
chi-cuadrado para categóricas, Mann-Whitney para numéricas.
"""
import pandas as pd
from scipy import stats


def compute_chi2_stats(df: pd.DataFrame, col: str, target: str) -> dict:
    """Test chi-cuadrado entre una variable categórica y el target.

    Returns:
        dict con chi2, p, dof y summary (frecuencias + tasa del target).
    """
    freq = df[col].value_counts(dropna=False).reset_index()
    freq.columns = [col, "count"]
    freq["pct"] = (freq["count"] / len(df) * 100).round(2)

    target_rate = (
        df.groupby(col, dropna=False)[target]
        .mean()
        .reset_index()
        .rename(columns={target: "tasa_transported"})
    )
    target_rate["tasa_transported"] = target_rate["tasa_transported"].round(4)
    summary = freq.merge(target_rate, on=col, how="left")

    contingency = pd.crosstab(df[col].fillna("NaN"), df[target])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

    return {"chi2": round(chi2, 2), "p": p_val, "dof": dof, "summary": summary}
