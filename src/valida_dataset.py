import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
risultati = []


def controlla(nome: str, condizione: bool, dettaglio: str = ""):
    risultati.append((nome, bool(condizione), dettaglio))


def carica(nome):
    return pd.read_csv(RAW / f"{nome}.csv")


cli, add, vei = carica("clienti"), carica("addetti_vendite"), carica("veicoli")
opt, conf, co = carica("optional_accessori"), carica("configurazioni"), carica("configurazioni_optional")
pro, fat = carica("proposte"), carica("fatture")

# 1. Chiavi primarie e vincoli di unicità ------------------------------------
for nome, df in [("clienti", cli), ("addetti_vendite", add), ("veicoli", vei), ("optional", opt),
                 ("configurazioni", conf), ("proposte", pro), ("fatture", fat)]:
    controlla(f"PK univoca {nome}", df["id"].is_unique)
controlla("targa univoca", vei["targa"].is_unique)
controlla("VIN univoco", vei["vin"].is_unique)
controlla("VIN lunghezza 17", (vei["vin"].str.len() == 17).all())
controlla("numero_fattura univoco", fat["numero_fattura"].is_unique)
controlla("codice optional univoco", opt["codice"].is_unique)

# 2. Integrità referenziale ----------------------------------------------------
controlla("FK configurazioni.cliente_id", conf["cliente_id"].isin(cli["id"]).all())
controlla("FK configurazioni.veicolo_id", conf["veicolo_id"].isin(vei["id"]).all())
controlla("FK config_optional.configurazione_id", co["configurazione_id"].isin(conf["id"]).all())
controlla("FK config_optional.optional_id", co["optional_id"].isin(opt["id"]).all())
controlla("FK proposte.configurazione_id", pro["configurazione_id"].isin(conf["id"]).all())
controlla("FK proposte.addetto_id (nullable)", pro["addetto_id"].dropna().isin(add["id"]).all())
controlla("FK fatture.proposta_id", fat["proposta_id"].isin(pro["id"]).all())
controlla("Fattura-Proposta 1:1", fat["proposta_id"].is_unique)

# 3. Coerenza degli stati --------------------------------------------------------
fp = fat.merge(pro, left_on="proposta_id", right_on="id", suffixes=("_f", "_p"))
controlla("ogni fattura -> proposta COMPLETATA", (fp["stato"] == "COMPLETATA").all())
controlla("ogni proposta COMPLETATA ha una fattura",
          set(pro.loc[pro["stato"] == "COMPLETATA", "id"]) == set(fat["proposta_id"]))
controlla("fattura.cliente_id = proposta.cliente_id", (fp["cliente_id_f"] == fp["cliente_id_p"]).all())
controlla("importo fattura = prezzo proposta", (fp["importo_totale"] == fp["prezzo_proposta"]).all())

fpc = fp.merge(conf, left_on="configurazione_id", right_on="id", suffixes=("", "_c"))
venduti = vei.loc[vei["stato"] == "VENDUTO", "id"]
controlla("veicolo VENDUTO <-> fattura", set(venduti) == set(fpc["veicolo_id"]))
controlla("nessun veicolo venduto due volte", fpc["veicolo_id"].is_unique)

acc = pro[pro["stato"] == "ACCETTATA"].merge(conf, left_on="configurazione_id", right_on="id")
controlla("veicolo OPZIONATO <-> proposta ACCETTATA",
          set(vei.loc[vei["stato"] == "OPZIONATO", "id"]) == set(acc["veicolo_id"]))
controlla("visibile_al_pubblico solo se DISPONIBILE",
          (vei["visibile_al_pubblico"] == (vei["stato"] == "DISPONIBILE")).all())

stati_validi = {"BOZZA", "INVIATA", "ACCETTATA", "RIFIUTATA", "SCADUTA", "ANNULLATA", "COMPLETATA"}
controlla("stati proposta nell'enumerativo", set(pro["stato"]) <= stati_validi)
controlla("stati veicolo nell'enumerativo",
          set(vei["stato"]) <= {"DISPONIBILE", "OPZIONATO", "VENDUTO", "NON_VISIBILE"})

# 4. Vincoli temporali -------------------------------------------------------------
d = lambda s: pd.to_datetime(s).dt.normalize()
cv = conf.merge(vei, left_on="veicolo_id", right_on="id", suffixes=("", "_v"))
controlla("veicolo inserito prima della (ultima modifica) configurazione",
          (d(cv["data_inserimento"]) <= d(cv["data_modifica"])).all())
pc = pro.merge(conf, left_on="configurazione_id", right_on="id", suffixes=("", "_c"))
controlla("proposta creata dopo la configurazione",
          (d(pc["data_creazione"]) >= d(pc["data_creazione_c"])).all())
controlla("scadenza = creazione + 30 giorni",
          ((pd.to_datetime(pro["data_scadenza"]) - d(pro["data_creazione"])).dt.days == 30).all())
controlla("fattura emessa dopo la proposta",
          (pd.to_datetime(fp["data_emissione"]) >= d(fp["data_creazione"])).all())
pag = fat.dropna(subset=["data_pagamento"])
controlla("pagamento dopo emissione",
          (pd.to_datetime(pag["data_pagamento"]) >= pd.to_datetime(pag["data_emissione"])).all())

# 5. Coerenza dei prezzi --------------------------------------------------------------
somma_opt = co.merge(opt, left_on="optional_id", right_on="id").groupby("configurazione_id")["prezzo"].sum()
atteso = conf["prezzo_base"] + conf["id"].map(somma_opt).fillna(0)
controlla("prezzo_totale = prezzo_base + optional", (conf["prezzo_totale"] == atteso).all())
controlla("prezzo_proposta <= prezzo_totale", (pc["prezzo_proposta"] <= pc["prezzo_totale"]).all())

# Report -------------------------------------------------------------------------------------
falliti = 0
for nome, ok, dett in risultati:
    print(f"[{'OK ' if ok else 'KO '}] {nome} {dett}")
    falliti += not ok
print(f"\n{len(risultati) - falliti}/{len(risultati)} controlli superati")
sys.exit(1 if falliti else 0)
