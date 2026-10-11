from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comune import TAB, salva_figura
from features import colonne_feature, costruisci_panel

SEED = 42
MESI_TEST = 6
OUT = Path("data/processed")


def crea_modelli():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                              random_state=SEED, n_jobs=-1),
    }


def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))


def valutazione_expanding_window(panel: pd.DataFrame):
    feat = colonne_feature(panel)
    dati = panel.dropna(subset=feat + ["target_vendite_t1"]).copy()
    ultimo_t = int(dati["indice_t"].max())
    mesi_test = list(range(ultimo_t - MESI_TEST + 1, ultimo_t + 1))

    predizioni = []
    for t in mesi_test:
        train = dati[dati["indice_t"] <= t - 1]
        test = dati[dati["indice_t"] == t].copy()
        test["pred_Baseline_MA3"] = test["media_mobile_3"]
        for nome, modello in crea_modelli().items():
            modello.fit(train[feat], train["target_vendite_t1"])
            test[f"pred_{nome}"] = np.clip(modello.predict(test[feat]), 0, None)
        predizioni.append(test)
    return pd.concat(predizioni), feat


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = costruisci_panel()
    pred, feat = valutazione_expanding_window(panel)
    pred.to_csv(OUT / "predizioni_test.csv", index=False)

    righe = []
    for col in [c for c in pred.columns if c.startswith("pred_")]:
        nome = col.replace("pred_", "")
        righe.append({
            "modello": nome,
            "MAE_vendite": mean_absolute_error(pred["target_vendite_t1"], pred[col]),
            "RMSE_vendite": rmse(pred["target_vendite_t1"], pred[col]),
            "MAE_domanda_reale": mean_absolute_error(pred["gt_domanda_t1"], pred[col]),
            "RMSE_domanda_reale": rmse(pred["gt_domanda_t1"], pred[col]),
        })
    metriche = pd.DataFrame(righe).round(3)
    metriche.to_csv(TAB / "metriche_previsione.csv", index=False)
    metriche.to_latex(TAB / "metriche_previsione.tex", index=False, float_format="%.3f")
    print(f"Righe di test: {len(pred)} ({pred['mese'].nunique()} mesi x {pred['chiave_modello'].nunique()} modelli)")
    print(metriche.to_string(index=False))

    # Figura: vendite reali vs previste, aggregate per mese
    agg = pred.groupby("mese")[["target_vendite_t1", "pred_Baseline_MA3", "pred_Ridge",
                                "pred_RandomForest"]].sum()
    etichette = [str(pd.Period(m, "M") + 1) for m in agg.index]  # il target si riferisce a t+1
    plt.figure()
    plt.plot(etichette, agg["target_vendite_t1"], "k-o", label="Vendite reali")
    plt.plot(etichette, agg["pred_Baseline_MA3"], "--", label="Media mobile 3 mesi")
    plt.plot(etichette, agg["pred_Ridge"], "-.", label="Ridge")
    plt.plot(etichette, agg["pred_RandomForest"], "-s", label="Random Forest")
    plt.ylabel("Vendite totali")
    plt.title("Previsione delle vendite nei mesi di test")
    plt.legend()
    salva_figura("previsione_vs_reale")

    # Figura: importanza delle feature (Random Forest addestrata su tutti i dati disponibili)
    dati = panel.dropna(subset=feat + ["target_vendite_t1"])
    rf = crea_modelli()["RandomForest"].fit(dati[feat], dati["target_vendite_t1"])
    imp = pd.Series(rf.feature_importances_, index=feat).sort_values()
    plt.figure(figsize=(7, 4.5))
    plt.barh(imp.index, imp.values)
    plt.xlabel("Importanza (riduzione media dell'impurità)")
    plt.title("Importanza delle feature - Random Forest")
    salva_figura("importanza_feature")


if __name__ == "__main__":
    main()