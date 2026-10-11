import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from comune import TAB, salva_figura
from features import costruisci_panel
from modelli import valutazione_expanding_window
from scoring import PESI_DEFAULT, calcola_score

SEED = 42
K_TOP = 3
N_PERTURBAZIONI = 500


def precision_at_k(punteggi: pd.Series, rilevanti: pd.Series, k: int = K_TOP) -> float:
    top = punteggi.sort_values(ascending=False).index[:k]
    return float(rilevanti.loc[top].mean())


def main():
    panel = costruisci_panel()
    pred, _ = valutazione_expanding_window(panel)
    rng = np.random.default_rng(SEED)

    risultati, dettaglio_sens = [], []
    for mese, righe in pred.groupby("mese"):
        fabbisogno = righe["gt_domanda_t1"] - righe["stock_fine_mese"]
        rilevanti = (fabbisogno > 0).astype(int)

        strategie = {
            "Score completo (RF)": calcola_score(righe, righe["pred_RandomForest"])["score"],
            "Ablazione (D = MA3)": calcola_score(righe, righe["media_mobile_3"])["score"],
            "B1: vendite ultimi 3 mesi": righe["media_mobile_3"],
            "B2: stock più basso": -righe["stock_fine_mese"],
        }
        for nome, punteggio in strategie.items():
            punteggio = punteggio.reindex(righe.index)
            rho = spearmanr(punteggio, fabbisogno).statistic
            risultati.append({"mese": mese, "strategia": nome, "spearman": rho,
                              "precision_at_3": precision_at_k(punteggio, rilevanti)})

        completo = calcola_score(righe, righe["pred_RandomForest"])
        sugg = completo["suggerimento"] == "Riassortire"
        risultati.append({"mese": mese, "strategia": "_riassortire", "n_suggeriti": int(sugg.sum()),
                          "corretti": int(rilevanti.loc[completo.index][sugg].sum())})

        # Sensibilità: perturbazione moltiplicativa uniforme in [0.8, 1.2] e rinormalizzazione
        base = completo["score"]
        for _ in range(N_PERTURBAZIONI):
            w = {k: v * rng.uniform(0.8, 1.2) for k, v in PESI_DEFAULT.items()}
            tot = sum(w.values())
            w = {k: v / tot for k, v in w.items()}
            pert = calcola_score(righe, righe["pred_RandomForest"], pesi=w)["score"].reindex(base.index)
            dettaglio_sens.append({"mese": mese, "kendall_tau": kendalltau(base, pert).statistic,
                                   "top3_invariata": set(base.index[:K_TOP]) ==
                                                     set(pert.sort_values(ascending=False).index[:K_TOP])})

    ris = pd.DataFrame(risultati)
    tabella = (ris[ris["strategia"] != "_riassortire"]
               .groupby("strategia")[["spearman", "precision_at_3"]].mean()
               .sort_values("spearman", ascending=False).round(3))
    tabella.to_csv(TAB / "valutazione_scoring.csv")
    tabella.to_latex(TAB / "valutazione_scoring.tex", float_format="%.3f")
    print("Confronto strategie (media sui mesi di test):")
    print(tabella.to_string())

    r = ris[ris["strategia"] == "_riassortire"]
    n, c = int(r["n_suggeriti"].sum()), int(r["corretti"].sum())
    print(f"\nSuggerimenti 'Riassortire': {n}, corretti (fabbisogno > 0): {c}"
          f" -> precisione {c / n:.2f}" if n else "\nNessun suggerimento 'Riassortire'")

    sens = pd.DataFrame(dettaglio_sens)
    print(f"\nSensibilità dei pesi ({N_PERTURBAZIONI} perturbazioni per mese):")
    print(f"  tau di Kendall medio: {sens['kendall_tau'].mean():.3f}  (minimo {sens['kendall_tau'].min():.3f})")
    print(f"  top-3 invariata nel {100 * sens['top3_invariata'].mean():.1f}% dei casi")
    sens.to_csv(TAB / "sensibilita_pesi.csv", index=False)

    plt.figure()
    plt.hist(sens["kendall_tau"], bins=30)
    plt.xlabel("tau di Kendall rispetto alla classifica con pesi di default")
    plt.ylabel("Frequenza")
    plt.title("Stabilità della classifica rispetto ai pesi (±20%)")
    salva_figura("sensibilita_pesi")

    plt.figure()
    tabella["spearman"].sort_values().plot(kind="barh")
    plt.xlabel("Correlazione di Spearman media con il fabbisogno reale")
    plt.title("Confronto tra strategie di suggerimento")
    salva_figura("confronto_strategie")


if __name__ == "__main__":
    main()