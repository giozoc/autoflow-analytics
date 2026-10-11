import matplotlib.pyplot as plt
import pandas as pd

from comune import (TAB, carica_tabelle, configurazioni_con_modello, proposte_con_modello,
                    salva_figura, stock_a_fine_mese, vendite_dettaglio)

t = carica_tabelle()
vend = vendite_dettaglio(t)
prop = proposte_con_modello(t)
conf = configurazioni_con_modello(t)
stock = stock_a_fine_mese(t)

# --- Figura: vendite mensili -----------------------------------------------------
serie = vend.groupby("mese").size()
plt.figure()
plt.plot(serie.index.to_timestamp(), serie.values, marker="o")
plt.ylabel("Veicoli venduti")
plt.title("Vendite mensili")
salva_figura("vendite_mensili")

# --- Figura: funnel configurazione -> proposta -> vendita ------------------------
inviate = prop[prop["stato"] != "BOZZA"]
funnel = pd.Series({"Configurazioni": len(conf), "Proposte inviate": len(inviate), "Vendite": len(vend)})
plt.figure()
plt.bar(funnel.index, funnel.values)
for i, val in enumerate(funnel.values):
    plt.text(i, val, str(val), ha="center", va="bottom")
plt.title("Funnel commerciale")
salva_figura("funnel")

# --- Figura: stato finale delle proposte ------------------------------------------
plt.figure()
prop["stato"].value_counts().plot(kind="bar")
plt.ylabel("Numero di proposte")
plt.title("Distribuzione degli stati delle proposte")
salva_figura("stati_proposte")

# --- Figura: optional più scelti -----------------------------------------------------
top_opt = (t["configurazioni_optional"]
           .merge(t["optional_accessori"], left_on="optional_id", right_on="id")
           .groupby("nome").size().sort_values())
plt.figure()
plt.barh(top_opt.index, top_opt.values)
plt.xlabel("Numero di configurazioni")
plt.title("Optional più selezionati")
salva_figura("optional_top")

# --- Figura: stock totale a fine mese -------------------------------------------------
st = stock.groupby("mese")["stock_fine_mese"].sum()
plt.figure()
plt.plot(st.index.to_timestamp(), st.values, marker="s")
plt.ylabel("Veicoli in giacenza")
plt.title("Stock a fine mese (ricostruito dalle date)")
salva_figura("stock_mensile")

# --- Tabella: KPI per modello -------------------------------------------------------
ultimo_mese = stock["mese"].max()
kpi = pd.DataFrame({
    "vendite": vend.groupby("chiave_modello").size(),
    "configurazioni": conf.groupby("chiave_modello").size(),
    "proposte_inviate": inviate.groupby("chiave_modello").size(),
    "giacenza_media_gg": vend.groupby("chiave_modello")["giorni_giacenza"].mean().round(1),
    "stock_attuale": stock[stock["mese"] == ultimo_mese].set_index("chiave_modello")["stock_fine_mese"],
}).fillna(0)
kpi["tasso_conversione"] = (kpi["vendite"] / kpi["proposte_inviate"]).round(3)
kpi = kpi.sort_values("vendite", ascending=False)
kpi.to_csv(TAB / "kpi_modelli.csv")
kpi.to_latex(TAB / "kpi_modelli.tex", float_format="%.2f")

# --- Tabella: performance addetti -----------------------------------------------------
inv_add = inviate.dropna(subset=["addetto_id"])
perf = pd.DataFrame({
    "proposte_inviate": inv_add.groupby("addetto_id").size(),
    "vendite": inv_add[inv_add["stato"] == "COMPLETATA"].groupby("addetto_id").size(),
})
perf["tasso_conversione"] = (perf["vendite"] / perf["proposte_inviate"]).round(3)
perf.to_csv(TAB / "performance_addetti.csv")
perf.to_latex(TAB / "performance_addetti.tex", float_format="%.3f")
senza = inviate[inviate["addetto_id"].isna()]
conv_senza = (senza["stato"] == "COMPLETATA").mean()

print(kpi.to_string())
print("\nPerformance addetti:\n", perf.to_string())
print(f"\nConversione proposte senza addetto: {conv_senza:.3f}")
print("Figure salvate in output/figure, tabelle in output/tabelle")