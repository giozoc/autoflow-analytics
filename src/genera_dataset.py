import argparse
import heapq
import itertools
import string
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from faker import Faker


# ----------------------------------------------------------------------------
# Utilità
# ----------------------------------------------------------------------------
def primo_del_mese(d: date) -> date:
    return d.replace(day=1)


def mese_successivo(d: date) -> date:
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def chiave_mese(d: date) -> str:
    return d.strftime("%Y-%m")


class Generatore:
    # Tipi di evento, con priorità (a parità di data si elabora prima il numero più basso)
    INIZIO_MESE, ARRIVO, VISIBILE, TENTATIVO_CONFIG, CREA_PROPOSTA, DECISIONE, SCADENZA, FATTURA = range(8)

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"])
        self.fake = Faker("it_IT")
        self.fake.seed_instance(cfg["seed"])

        self.inizio = date.fromisoformat(cfg["periodo"]["data_inizio"])
        self.fine = date.fromisoformat(cfg["periodo"]["data_fine"])

        self.catalogo = cfg["catalogo"]
        self.coda = []                        # coda di priorità degli eventi
        self.contatore = itertools.count()    # per rendere stabile l'ordinamento

        # "Tabelle" in memoria (liste di dizionari)
        self.clienti, self.addetti, self.veicoli, self.optional = [], [], [], []
        self.configurazioni, self.config_optional, self.proposte, self.fatture = [], [], [], []

        # Strutture di stato
        self.disponibili = defaultdict(list)   # modello -> id veicoli DISPONIBILI
        self.in_transito = defaultdict(int)    # modello -> veicoli ordinati non ancora arrivati
        self.gt = defaultdict(lambda: defaultdict(float))  # ground truth (modello, mese) -> metriche
        self.num_fattura_anno = defaultdict(int)
        self.targhe, self.vin = set(), set()

    # ------------------------------------------------------------------ eventi
    def pianifica(self, giorno: date, tipo: int, **dati):
        if giorno <= self.fine:  # gli eventi oltre la data di fine non avvengono
            heapq.heappush(self.coda, (giorno, tipo, next(self.contatore), dati))

    def timestamp(self, giorno: date) -> str:
        ora = int(self.rng.integers(9, 19))
        minuto = int(self.rng.integers(0, 60))
        return datetime(giorno.year, giorno.month, giorno.day, ora, minuto).strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------------------------------------- anagrafiche base
    def crea_addetti(self):
        for i in range(self.cfg["concessionaria"]["num_addetti"]):
            nome, cognome = self.fake.first_name(), self.fake.last_name()
            self.addetti.append({
                "id": i + 1, "nome": nome, "cognome": cognome,
                "matricola": f"AV{i + 1:04d}",
                "email": f"{nome}.{cognome}@autoflow-demo.it".lower().replace(" ", "").replace("'", ""),
                "telefono": self.fake.phone_number(),
                # attributo NASCOSTO: abilità commerciale (non esportato nel CSV)
                "_abilita": float(np.clip(self.rng.normal(1.0, 0.15), 0.7, 1.3)),
            })

    def crea_optional(self):
        for i, o in enumerate(self.cfg["optional"]):
            self.optional.append({"id": i + 1, "codice": o["codice"], "nome": o["nome"],
                                  "descrizione": f"Optional: {o['nome']}", "prezzo": o["prezzo"],
                                  "_prob": o["prob"], "_bonus_suv": o["bonus_suv"]})

    def nuovo_cliente(self, giorno: date) -> int:
        nome, cognome = self.fake.first_name(), self.fake.last_name()
        cid = len(self.clienti) + 1
        self.clienti.append({
            "id": cid, "nome": nome, "cognome": cognome,
            "email": f"{nome}.{cognome}.{cid}@example.it".lower().replace(" ", "").replace("'", ""),
            "telefono": self.fake.phone_number(),
            "indirizzo": self.fake.address().replace("\n", " "),
            "codice_fiscale": self.fake.ssn(),
            "data_nascita": self.fake.date_of_birth(minimum_age=19, maximum_age=80).isoformat(),
            "data_registrazione": self.timestamp(giorno - timedelta(days=int(self.rng.integers(0, 15)))),
        })
        return cid

    def scegli_cliente(self, giorno: date) -> int:
        p = self.cfg["concessionaria"]["prob_cliente_esistente"]
        if self.clienti and self.rng.random() < p:
            return int(self.rng.integers(1, len(self.clienti) + 1))
        return self.nuovo_cliente(giorno)

    def genera_targa(self) -> str:
        while True:
            t = self.fake.license_plate()
            if t not in self.targhe:
                self.targhe.add(t)
                return t

    def genera_vin(self, marca: str) -> str:
        caratteri = [c for c in string.ascii_uppercase + string.digits if c not in "IOQ"]  # standard VIN
        while True:
            wmi = (marca.replace(" ", "").upper() + "XXX")[:3]
            v = wmi + "".join(self.rng.choice(caratteri, 14))
            if v not in self.vin:
                self.vin.add(v)
                return v

    def crea_veicolo(self, m: int, giorno_arrivo: date) -> int:
        mod = self.catalogo[m]
        vid = len(self.veicoli) + 1
        self.veicoli.append({
            "id": vid, "marca": mod["marca"], "modello": mod["modello"], "anno": giorno_arrivo.year,
            "targa": self.genera_targa(), "vin": self.genera_vin(mod["marca"]),
            "prezzo_base": int(round(mod["prezzo"] * self.rng.uniform(0.98, 1.03), -2)),
            "chilometraggio": int(self.rng.integers(0, 30)),
            "alimentazione": mod["alimentazione"], "cambio": mod["cambio"],
            "colore_esterno": str(self.rng.choice(self.cfg["colori"])),
            "stato": "NON_VISIBILE", "visibile_al_pubblico": False,
            "data_inserimento": self.timestamp(giorno_arrivo),
            "data_ultimo_aggiornamento": self.timestamp(giorno_arrivo),
            "_m": m,
        })
        return vid

    # ----------------------------------------------------------- simulazione
    def lambda_config(self, m: int, giorno: date) -> float:
        """Intensità attesa di configurazioni per il modello m nel mese di 'giorno'."""
        mod = self.catalogo[m]
        t = (giorno.year - self.inizio.year) * 12 + (giorno.month - self.inizio.month)
        stag = self.cfg["stagionalita"][giorno.month - 1]
        return mod["config_mese"] * stag * (1 + mod["trend_mensile"]) ** t

    def vendite_ultimi_mesi(self, m: int, giorno: date, n: int = 3) -> float:
        """Media delle vendite (fatture) degli ultimi n mesi completi: è ciò che 'vede' il gestore."""
        mesi, cursore = [], primo_del_mese(giorno)
        for _ in range(n):
            cursore = (cursore - timedelta(days=1)).replace(day=1)
            if cursore >= self.inizio:
                mesi.append(chiave_mese(cursore))
        if not mesi:
            return None
        conteggio = sum(1 for f in self.fatture
                        if f["_m"] == m and f["data_emissione"][:7] in mesi)
        return conteggio / len(mesi)

    def stock_iniziale(self):
        r = self.cfg["riassortimento"]
        for m, mod in enumerate(self.catalogo):
            vendite_attese = self.lambda_config(m, self.inizio) * 0.6 * mod["conv"]
            n = max(2, int(round(vendite_attese * r["mesi_stock_iniziale"])))
            for _ in range(n):
                # giacenza iniziale: veicoli arrivati nei 60 giorni precedenti
                arrivo = self.inizio - timedelta(days=int(self.rng.integers(1, 60)))
                vid = self.crea_veicolo(m, arrivo)
                v = self.veicoli[vid - 1]
                v["stato"], v["visibile_al_pubblico"] = "DISPONIBILE", True
                self.disponibili[m].append(vid)

    def evento_inizio_mese(self, giorno: date):
        r = self.cfg["riassortimento"]
        fine_mese = mese_successivo(giorno) - timedelta(days=1)
        for m in range(len(self.catalogo)):
            # 1) riassortimento "umano" (se non è il primo mese)
            if giorno > self.inizio:
                media = self.vendite_ultimi_mesi(m, giorno) or 0.0
                obiettivo = r["moltiplicatore_scorta"] * media + r["scorta_sicurezza"]
                non_visibili = sum(1 for v in self.veicoli if v["_m"] == m and v["stato"] == "NON_VISIBILE")
                posizione = len(self.disponibili[m]) + non_visibili + self.in_transito[m]
                fabbisogno = max(0.0, obiettivo - posizione)
                qta = int(round(fabbisogno * self.rng.lognormal(0, r["rumore_ordine_sigma"])))
                if posizione == 0:
                    qta = max(qta, 2)
                for _ in range(qta):
                    lt = int(self.rng.integers(r["lead_time_min_giorni"], r["lead_time_max_giorni"] + 1))
                    self.in_transito[m] += 1
                    self.pianifica(giorno + timedelta(days=lt), self.ARRIVO, m=m)

            # 2) tentativi di configurazione del mese (processo di Poisson)
            lam = self.lambda_config(m, giorno)
            n = int(self.rng.poisson(lam))
            self.gt[(m, chiave_mese(giorno))]["lambda_config"] = lam
            self.gt[(m, chiave_mese(giorno))]["tentativi_config"] += n
            giorni_mese = (fine_mese - giorno).days + 1
            for _ in range(n):
                self.pianifica(giorno + timedelta(days=int(self.rng.integers(0, giorni_mese))),
                               self.TENTATIVO_CONFIG, m=m)
        self.pianifica(mese_successivo(giorno), self.INIZIO_MESE)

    def evento_arrivo(self, giorno: date, m: int):
        self.in_transito[m] -= 1
        vid = self.crea_veicolo(m, giorno)
        self.pianifica(giorno + timedelta(days=int(self.rng.integers(1, 4))), self.VISIBILE, vid=vid)

    def evento_visibile(self, giorno: date, vid: int):
        v = self.veicoli[vid - 1]
        v["stato"], v["visibile_al_pubblico"] = "DISPONIBILE", True
        v["data_ultimo_aggiornamento"] = self.timestamp(giorno)
        self.disponibili[v["_m"]].append(vid)

    def prob_accettazione(self, m: int, addetto_id):
        c = self.cfg["concessionaria"]
        p = self.catalogo[m]["conv"]
        if addetto_id is None:
            return p * c["fattore_senza_addetto"]
        return min(0.95, p * self.addetti[addetto_id - 1]["_abilita"])

    def evento_tentativo_config(self, giorno: date, m: int):
        c = self.cfg["concessionaria"]
        mese = chiave_mese(giorno)
        if not self.disponibili[m]:
            # Domanda persa: nessun veicolo in showroom. Non lascia traccia nel database,
            # ma la registriamo nella ground truth simulando se sarebbe diventata una vendita.
            self.gt[(m, mese)]["config_perse"] += 1
            if (self.rng.random() < c["prob_proposta_da_config"]
                    and self.rng.random() > c["prob_bozza"]
                    and self.rng.random() < self.catalogo[m]["conv"]):
                self.gt[(m, mese)]["vendite_perse"] += 1
            return

        vid = int(self.rng.choice(self.disponibili[m]))
        cliente = self.scegli_cliente(giorno)
        cfg_id = len(self.configurazioni) + 1
        suv = self.catalogo[m]["segmento"] == "SUV"
        prezzo_opt = 0
        for o in self.optional:
            p = min(0.95, o["_prob"] * (o["_bonus_suv"] if suv else 1.0))
            if self.rng.random() < p:
                self.config_optional.append({"configurazione_id": cfg_id, "optional_id": o["id"]})
                prezzo_opt += o["prezzo"]
        base = self.veicoli[vid - 1]["prezzo_base"]
        ts = self.timestamp(giorno)
        self.configurazioni.append({
            "id": cfg_id, "cliente_id": cliente, "veicolo_id": vid,
            "prezzo_base": base, "prezzo_totale": base + prezzo_opt, "note": "",
            "data_creazione": ts, "data_modifica": ts, "_m": m,
        })
        if self.rng.random() < c["prob_proposta_da_config"]:
            self.pianifica(giorno + timedelta(days=int(self.rng.integers(0, 4))),
                           self.CREA_PROPOSTA, cfg_id=cfg_id)

    def evento_crea_proposta(self, giorno: date, cfg_id: int):
        c = self.cfg["concessionaria"]
        conf = self.configurazioni[cfg_id - 1]
        pid = len(self.proposte) + 1
        addetto = (int(self.rng.integers(1, len(self.addetti) + 1))
                   if self.rng.random() < c["prob_addetto_assegnato"] else None)
        sconto = self.rng.uniform(0, c["sconto_max"])
        scadenza = giorno + timedelta(days=c["giorni_validita_proposta"])
        stato = "BOZZA" if self.rng.random() < c["prob_bozza"] else "INVIATA"
        self.proposte.append({
            "id": pid, "cliente_id": conf["cliente_id"], "addetto_id": addetto,
            "configurazione_id": cfg_id,
            "prezzo_proposta": int(round(conf["prezzo_totale"] * (1 - sconto), -1)),
            "note_cliente": "", "note_interne": "", "stato": stato,
            "data_creazione": self.timestamp(giorno), "data_scadenza": scadenza.isoformat(),
            "_m": conf["_m"],
        })
        if stato == "INVIATA":
            self.pianifica(giorno + timedelta(days=int(self.rng.integers(2, 15))), self.DECISIONE, pid=pid)

    def evento_decisione(self, giorno: date, pid: int):
        p = self.proposte[pid - 1]
        m = p["_m"]
        if self.rng.random() < self.prob_accettazione(m, p["addetto_id"]):
            conf = self.configurazioni[p["configurazione_id"] - 1]
            vid = conf["veicolo_id"]
            if vid not in self.disponibili[m]:
                # veicolo nel frattempo venduto/opzionato: si prova con un altro dello stesso modello
                if self.disponibili[m]:
                    vid = int(self.rng.choice(self.disponibili[m]))
                    conf["veicolo_id"] = vid
                    conf["data_modifica"] = self.timestamp(giorno)
                else:
                    p["stato"] = "ANNULLATA"          # vendita persa per mancanza di stock
                    self.gt[(m, chiave_mese(giorno))]["vendite_perse"] += 1
                    return
            self.disponibili[m].remove(vid)
            v = self.veicoli[vid - 1]
            v["stato"], v["visibile_al_pubblico"] = "OPZIONATO", False
            v["data_ultimo_aggiornamento"] = self.timestamp(giorno)
            p["stato"] = "ACCETTATA"
            self.pianifica(giorno + timedelta(days=int(self.rng.integers(3, 16))), self.FATTURA, pid=pid)
        elif self.rng.random() < 0.7:
            p["stato"] = "RIFIUTATA"
        else:
            self.pianifica(date.fromisoformat(p["data_scadenza"]), self.SCADENZA, pid=pid)

    def evento_scadenza(self, giorno: date, pid: int):
        self.proposte[pid - 1]["stato"] = "SCADUTA"

    def evento_fattura(self, giorno: date, pid: int):
        p = self.proposte[pid - 1]
        conf = self.configurazioni[p["configurazione_id"] - 1]
        v = self.veicoli[conf["veicolo_id"] - 1]
        self.num_fattura_anno[giorno.year] += 1
        pagamento = giorno + timedelta(days=int(self.rng.integers(0, 11)))
        self.fatture.append({
            "id": len(self.fatture) + 1,
            "numero_fattura": f"{giorno.year}/{self.num_fattura_anno[giorno.year]:06d}",
            "data_emissione": giorno.isoformat(), "cliente_id": p["cliente_id"], "proposta_id": pid,
            "importo_totale": p["prezzo_proposta"],
            "data_pagamento": pagamento.isoformat() if pagamento <= self.fine else None,
            "_m": p["_m"],
        })
        p["stato"] = "COMPLETATA"
        v["stato"], v["visibile_al_pubblico"] = "VENDUTO", False
        v["data_ultimo_aggiornamento"] = self.timestamp(giorno)

    # ------------------------------------------------------------------ main
    def esegui(self):
        self.crea_addetti()
        self.crea_optional()
        self.stock_iniziale()
        self.pianifica(self.inizio, self.INIZIO_MESE)
        gestori = {
            self.INIZIO_MESE: self.evento_inizio_mese, self.ARRIVO: self.evento_arrivo,
            self.VISIBILE: self.evento_visibile, self.TENTATIVO_CONFIG: self.evento_tentativo_config,
            self.CREA_PROPOSTA: self.evento_crea_proposta, self.DECISIONE: self.evento_decisione,
            self.SCADENZA: self.evento_scadenza, self.FATTURA: self.evento_fattura,
        }
        while self.coda:
            giorno, tipo, _, dati = heapq.heappop(self.coda)
            gestori[tipo](giorno, **dati)

    def esporta(self, cartella: Path):
        raw, gt = cartella / "raw", cartella / "ground_truth"
        raw.mkdir(parents=True, exist_ok=True)
        gt.mkdir(parents=True, exist_ok=True)

        def salva(righe, nome):
            df = pd.DataFrame(righe)
            df = df[[c for c in df.columns if not c.startswith("_")]]  # rimuove attributi nascosti
            for col in [c for c in df.columns if c.endswith("_id")]:
                df[col] = df[col].astype("Int64")   # interi "nullable": 3 e non 3.0, vuoto se NULL
            df.to_csv(raw / f"{nome}.csv", index=False)
            return len(df)

        tabelle = {
            "clienti": self.clienti, "addetti_vendite": self.addetti, "veicoli": self.veicoli,
            "optional_accessori": self.optional, "configurazioni": self.configurazioni,
            "configurazioni_optional": self.config_optional, "proposte": self.proposte,
            "fatture": self.fatture,
        }
        for nome, righe in tabelle.items():
            print(f"  {nome:<25} {salva(righe, nome):>6} righe")

        # Ground truth: domanda reale = vendite effettive + vendite perse
        vendite = defaultdict(int)
        for f in self.fatture:
            vendite[(f["_m"], f["data_emissione"][:7])] += 1
        righe, cursore = [], self.inizio
        while cursore <= self.fine:
            for m, mod in enumerate(self.catalogo):
                k = (m, chiave_mese(cursore))
                g = self.gt[k]
                righe.append({
                    "mese": k[1], "marca": mod["marca"], "modello": mod["modello"],
                    "alimentazione": mod["alimentazione"],
                    "lambda_config": round(g.get("lambda_config", 0.0), 3),
                    "tentativi_config": int(g.get("tentativi_config", 0)),
                    "config_perse": int(g.get("config_perse", 0)),
                    "vendite_effettive": vendite[k],
                    "vendite_perse": int(g.get("vendite_perse", 0)),
                    "domanda_reale": vendite[k] + int(g.get("vendite_perse", 0)),
                })
            cursore = mese_successivo(cursore)
        pd.DataFrame(righe).to_csv(gt / "domanda_latente.csv", index=False)
        print(f"  {'ground_truth':<25} {len(righe):>6} righe")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/parametri.yaml")
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gen = Generatore(cfg)
    print("Simulazione in corso...")
    gen.esegui()
    print("Esportazione:")
    gen.esporta(Path(args.output))


if __name__ == "__main__":
    main()