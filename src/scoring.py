import numpy as np
import pandas as pd

from comune import TAB
from features import colonne_feature, costruisci_panel
from modelli import crea_modelli

PESI_DEFAULT = {"D": 0.35, "C": 0.15, "T": 0.10, "K": 0.30, "G": 0.10}
EPS = 0.5          # evita divisioni per zero nella copertura
K_MAX = 3.0        # oltre 3 mesi di copertura lo stock è considerato "abbondante"
G_MAX = 180.0      # età dello stock oltre 180 giorni trattata come 180
SOGLIA_ALTA, SOGLIA_BASSA = 0.60, 0.35
COPERTURA_MIN, COPERTURA_MAX = 1.5, 2.5   # mesi (il lead time simulato è di 2-5 settimane)


def min_max(s: pd.Series) -> pd.Series:
    """Normalizzazione min-max; se tutti i valori sono uguali restituisce 0.5."""
    lo, hi = s.min(), s.max()
    if np.isclose(hi, lo):
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def calcola_score(righe_mese: pd.DataFrame, domanda_prevista: pd.Series, pesi=None) -> pd.DataFrame:
    """
    righe_mese: righe del panel relative a un solo mese t (una per modello)
    domanda_prevista: previsione delle vendite per t+1, stesso indice di righe_mese
    """
    pesi = pesi or PESI_DEFAULT
    assert np.isclose(sum(pesi.values()), 1.0), "i pesi devono sommare a 1"

    r = pd.DataFrame(index=righe_mese.index)
    r["chiave_modello"] = righe_mese["chiave_modello"]
    r["D"] = domanda_prevista
    r["C"] = righe_mese["tasso_conv_3m"]
    r["T"] = righe_mese["trend_config"]
    r["stock"] = righe_mese["stock_fine_mese"]
    r["K"] = (r["stock"] / np.maximum(r["D"], EPS)).clip(upper=K_MAX)
    r["G"] = righe_mese["eta_media_stock"].clip(upper=G_MAX)

    r["score"] = (pesi["D"] * min_max(r["D"]) + pesi["C"] * min_max(r["C"])
                  + pesi["T"] * min_max(r["T"]) + pesi["K"] * (1 - min_max(r["K"]))
                  + pesi["G"] * (1 - min_max(r["G"])))

    def regola(riga):
        if riga["score"] >= SOGLIA_ALTA and riga["K"] < COPERTURA_MIN:
            return "Riassortire"
        if riga["score"] < SOGLIA_BASSA or riga["K"] >= COPERTURA_MAX:
            return "Non acquistare"
        return "Monitorare"

    r["suggerimento"] = r.apply(regola, axis=1)
    return r.sort_values("score", ascending=False)


def main():
    panel = costruisci_panel()
    feat = colonne_feature(panel)
    train = panel.dropna(subset=feat + ["target_vendite_t1"])
    ultimo = panel[panel["indice_t"] == panel["indice_t"].max()].dropna(subset=feat)

    rf = crea_modelli()["RandomForest"].fit(train[feat], train["target_vendite_t1"])
    d_prev = pd.Series(np.clip(rf.predict(ultimo[feat]), 0, None), index=ultimo.index)

    classifica = calcola_score(ultimo, d_prev)
    mese_suggerito = pd.Period(ultimo["mese"].iloc[0], "M") + 1
    out = classifica[["chiave_modello", "D", "stock", "K", "G", "score", "suggerimento"]].round(2)
    out.insert(0, "posizione", range(1, len(out) + 1))
    out.to_csv(TAB / "classifica_riassortimento.csv", index=False)
    out.to_latex(TAB / "classifica_riassortimento.tex", index=False, float_format="%.2f")
    print(f"Suggerimenti di riassortimento per {mese_suggerito}:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()