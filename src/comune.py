from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAW = Path("data/raw")
GT = Path("data/ground_truth")
FIG = Path("output/figure")
TAB = Path("output/tabelle")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.figsize": (7, 3.6), "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def carica_tabelle() -> dict:
    nomi = ["clienti", "addetti_vendite", "veicoli", "optional_accessori", "configurazioni",
            "configurazioni_optional", "proposte", "fatture"]
    t = {n: pd.read_csv(RAW / f"{n}.csv") for n in nomi}
    v = t["veicoli"]
    # Chiave del "modello commerciale": un Veicolo è un'auto fisica, l'analisi aggrega per modello
    v["chiave_modello"] = v["marca"] + " " + v["modello"].astype(str) + " " + v["alimentazione"]
    v["data_inserimento"] = pd.to_datetime(v["data_inserimento"])
    t["configurazioni"]["data_creazione"] = pd.to_datetime(t["configurazioni"]["data_creazione"])
    t["proposte"]["data_creazione"] = pd.to_datetime(t["proposte"]["data_creazione"])
    t["fatture"]["data_emissione"] = pd.to_datetime(t["fatture"]["data_emissione"])
    return t


def vendite_dettaglio(t: dict) -> pd.DataFrame:
    """Join Fattura -> Proposta -> Configurazione -> Veicolo (una riga per vendita)."""
    df = (t["fatture"]
          .merge(t["proposte"][["id", "configurazione_id", "addetto_id", "data_creazione"]],
                 left_on="proposta_id", right_on="id", suffixes=("", "_proposta"))
          .merge(t["configurazioni"][["id", "veicolo_id", "prezzo_totale"]],
                 left_on="configurazione_id", right_on="id", suffixes=("", "_conf"))
          .merge(t["veicoli"][["id", "chiave_modello", "alimentazione", "data_inserimento"]],
                 left_on="veicolo_id", right_on="id", suffixes=("", "_veicolo")))
    df["giorni_giacenza"] = (df["data_emissione"] - df["data_inserimento"].dt.normalize()).dt.days
    df["mese"] = df["data_emissione"].dt.to_period("M")
    return df


def proposte_con_modello(t: dict) -> pd.DataFrame:
    df = (t["proposte"]
          .merge(t["configurazioni"][["id", "veicolo_id"]], left_on="configurazione_id",
                 right_on="id", suffixes=("", "_conf"))
          .merge(t["veicoli"][["id", "chiave_modello"]], left_on="veicolo_id",
                 right_on="id", suffixes=("", "_veicolo")))
    df["mese"] = df["data_creazione"].dt.to_period("M")
    return df


def configurazioni_con_modello(t: dict) -> pd.DataFrame:
    df = t["configurazioni"].merge(t["veicoli"][["id", "chiave_modello"]],
                                   left_on="veicolo_id", right_on="id", suffixes=("", "_veicolo"))
    df["mese"] = df["data_creazione"].dt.to_period("M")
    return df


def stock_a_fine_mese(t: dict) -> pd.DataFrame:
    """
    Ricostruzione temporale dello stock: un veicolo è in giacenza alla data d se
    data_inserimento <= d e (non è stato venduto oppure data_emissione_fattura > d).
    Restituisce, per ogni (modello, mese), il numero di veicoli e l'età media in giorni.
    """
    v = t["veicoli"][["id", "chiave_modello", "data_inserimento"]].copy()
    vend = vendite_dettaglio(t)[["veicolo_id", "data_emissione"]]
    v = v.merge(vend, left_on="id", right_on="veicolo_id", how="left")
    mesi = pd.period_range(t["fatture"]["data_emissione"].min().to_period("M"),
                           t["fatture"]["data_emissione"].max().to_period("M"), freq="M")
    righe = []
    for m in mesi:
        fine = m.to_timestamp(how="end").normalize()
        in_stock = v[(v["data_inserimento"].dt.normalize() <= fine)
                     & (v["data_emissione"].isna() | (v["data_emissione"] > fine))]
        eta = (fine - in_stock["data_inserimento"].dt.normalize()).dt.days
        g = in_stock.assign(eta=eta).groupby("chiave_modello")["eta"].agg(["count", "mean"])
        for chiave in v["chiave_modello"].unique():
            righe.append({"chiave_modello": chiave, "mese": m,
                          "stock_fine_mese": int(g["count"].get(chiave, 0)),
                          "eta_media_stock": float(g["mean"].get(chiave, 0.0))})
    return pd.DataFrame(righe)


def carica_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(GT / "domanda_latente.csv")
    gt["chiave_modello"] = gt["marca"] + " " + gt["modello"].astype(str) + " " + gt["alimentazione"]
    gt["mese"] = pd.PeriodIndex(gt["mese"], freq="M")
    return gt


def salva_figura(nome: str):
    plt.tight_layout()
    plt.savefig(FIG / f"{nome}.pdf")   # PDF vettoriale, da includere in LaTeX
    plt.close()