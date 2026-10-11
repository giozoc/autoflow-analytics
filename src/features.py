from pathlib import Path

import numpy as np
import pandas as pd

from comune import (carica_ground_truth, carica_tabelle, configurazioni_con_modello,
                    proposte_con_modello, stock_a_fine_mese, vendite_dettaglio)

OUT = Path("data/processed")

FEATURE_NUMERICHE = [
    "vendite", "vendite_lag1", "vendite_lag2", "media_mobile_3",
    "configurazioni", "configurazioni_lag1", "proposte_inviate", "tasso_conv_3m",
    "stock_fine_mese", "eta_media_stock", "trend_config", "mese_sin", "mese_cos", "indice_t",
]


def costruisci_panel() -> pd.DataFrame:
    t = carica_tabelle()
    vend, prop, conf = vendite_dettaglio(t), proposte_con_modello(t), configurazioni_con_modello(t)
    stock = stock_a_fine_mese(t)

    modelli = sorted(t["veicoli"]["chiave_modello"].unique())
    mesi = sorted(stock["mese"].unique())
    griglia = pd.MultiIndex.from_product([modelli, mesi], names=["chiave_modello", "mese"])

    p = pd.DataFrame(index=griglia)
    p["vendite"] = vend.groupby(["chiave_modello", "mese"]).size()
    p["configurazioni"] = conf.groupby(["chiave_modello", "mese"]).size()
    inviate = prop[prop["stato"] != "BOZZA"]
    p["proposte_inviate"] = inviate.groupby(["chiave_modello", "mese"]).size()
    p = p.fillna(0).reset_index()
    p = p.merge(stock, on=["chiave_modello", "mese"], how="left")

    g = p.groupby("chiave_modello", group_keys=False)
    p["vendite_lag1"] = g["vendite"].shift(1)
    p["vendite_lag2"] = g["vendite"].shift(2)
    p["media_mobile_3"] = g["vendite"].transform(lambda s: s.rolling(3).mean())
    p["configurazioni_lag1"] = g["configurazioni"].shift(1)
    v3 = g["vendite"].transform(lambda s: s.rolling(3).sum())
    pr3 = g["proposte_inviate"].transform(lambda s: s.rolling(3).sum())
    p["tasso_conv_3m"] = (v3 / pr3.replace(0, np.nan)).clip(upper=1).fillna(0)
    c3 = g["configurazioni"].transform(lambda s: s.rolling(3).sum())
    c3_prec = g["configurazioni"].transform(lambda s: s.shift(3).rolling(3).sum())
    p["trend_config"] = (c3 - c3_prec) / (c3_prec + 1)

    # Codifica ciclica del mese: dicembre e gennaio risultano "vicini"
    num_mese = p["mese"].dt.month
    p["mese_sin"] = np.sin(2 * np.pi * num_mese / 12)
    p["mese_cos"] = np.cos(2 * np.pi * num_mese / 12)
    p["indice_t"] = p["mese"].rank(method="dense").astype(int) - 1

    # One-hot dell'alimentazione (ultima parola della chiave)
    p["alimentazione"] = p["chiave_modello"].str.split().str[-1]
    p = pd.get_dummies(p, columns=["alimentazione"], prefix="alim", dtype=int)

    # Target: vendite del mese successivo
    p["target_vendite_t1"] = g["vendite"].shift(-1)

    # Ground truth (solo per la valutazione, MAI come feature)
    gt = carica_ground_truth()[["chiave_modello", "mese", "domanda_reale"]]
    gt["mese"] = gt["mese"] - 1          # la domanda di t+1 viene allineata alla riga t
    p = p.merge(gt.rename(columns={"domanda_reale": "gt_domanda_t1"}),
                on=["chiave_modello", "mese"], how="left")
    return p


def colonne_feature(panel: pd.DataFrame) -> list:
    return FEATURE_NUMERICHE + [c for c in panel.columns if c.startswith("alim_")]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    panel = costruisci_panel()
    panel.to_csv(OUT / "panel.csv", index=False)
    print(panel.shape)
    print(panel.dropna().head().to_string())