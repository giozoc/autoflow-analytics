# AutoFlow Analytics

[![Progetto](https://img.shields.io/badge/progetto-AutoFlow-blue)](https://github.com/giozoc/AutoFlow)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](#stack-tecnologico)
[![Architettura](https://img.shields.io/badge/Pipeline-Batch-blue)](#architettura)
[![Licenza](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](#licenza)

---

## Documentazione

AutoFlow Analytics raccoglie il codice, il dataset simulato e gli output sperimentali del **Capitolo 4** della tesi *AutoFlow* (Corso di Laurea Triennale in Informatica, Università degli Studi di Salerno): un sistema decisionale intelligente a supporto del riassortimento dello stock.

- **Progetto principale**
  La piattaforma gestionale da cui il modulo analitico trae il modello dati:
  👉 <https://github.com/giozoc/AutoFlow>

- **Contenuto del repository**:

  * Simulatore a eventi discreti per la generazione del dataset
  * Script di validazione dell'integrità (37 controlli)
  * Pipeline di analisi: KPI, feature engineering, modelli predittivi, scoring, valutazione
  * Figure (PDF) e tabelle (CSV e LaTeX) prodotte dagli esperimenti

---

## Panoramica del Progetto

AutoFlow Analytics estende AutoFlow con un **modulo decisionale** che utilizza in sola lettura i dati operativi (veicoli, configurazioni, proposte, fatture) per:

- calcolare **KPI operativi** (vendite mensili, funnel commerciale, tasso di conversione, giacenza media)
- ricostruire lo **stock storico** a partire dalle date, poiché il database memorizza solo lo stato corrente
- **prevedere la domanda** del mese successivo per ogni modello commerciale
- produrre una **classifica di priorità di riassortimento** interpretabile (Riassortire / Monitorare / Non acquistare)
- **valutare sperimentalmente** i risultati tramite baseline, ablazione e analisi di sensibilità dei pesi

Poiché AutoFlow non è ancora in esercizio presso una concessionaria reale, la sperimentazione usa un **dataset simulato**, strutturalmente identico ai dati che la piattaforma produrrebbe e dotato di *ground truth* (domanda reale, comprese le vendite perse per mancanza di stock) usata solo per la valutazione.

> Il dataset non intende riprodurre fedelmente il mercato automobilistico, ma fornire un banco di prova controllato e riproducibile.

---

## Stack Tecnologico

- **Linguaggio**: Python 3.11 / 3.12
- **Analisi dati**: pandas, NumPy, SciPy
- **Machine learning**: scikit-learn (Ridge, Random Forest)
- **Visualizzazione**: Matplotlib (figure in PDF vettoriale)
- **Configurazione**: YAML (PyYAML)
- **Dati anagrafici fittizi**: Faker (localizzazione `it_IT`)

---

## Architettura

Il modulo è una **pipeline batch** a fasi con input e output ben definiti (pattern *Pipes and Filters*):

1. **Generazione** – simulatore a eventi discreti con coda di priorità, seed fisso e parametri in `config/parametri.yaml`
2. **Validazione** – controlli di unicità, integrità referenziale, coerenza di stati, tempi e prezzi
3. **KPI e analisi esplorativa** – indicatori operativi, figure e tabelle
4. **Modelli predittivi** – baseline (media mobile), Ridge e Random Forest, con validazione *expanding window*
5. **Scoring** – punteggio multi-criterio in [0, 1] con regole decisionali esplicite
6. **Valutazione** – Spearman, Precision@3, ablazione e tau di Kendall sulle perturbazioni dei pesi

Nell'architettura complessiva la pipeline legge dal database di AutoFlow e scrive i risultati in tabelle dedicate, esposte poi dal backend Spring Boot tramite API REST riservate all'amministratore.

---

## Struttura del Repository

```text
autoflow-analytics/
├── config/
│   └── parametri.yaml          # parametri della simulazione
├── src/
│   ├── genera_dataset.py       # simulatore a eventi discreti
│   ├── valida_dataset.py       # 37 controlli di integrità
│   ├── comune.py               # funzioni condivise
│   ├── kpi.py                  # KPI e figure
│   ├── features.py             # feature engineering
│   ├── modelli.py              # modelli predittivi
│   ├── scoring.py              # algoritmo di suggerimento
│   └── valutazione.py          # valutazione sperimentale
├── data/
│   ├── raw/                    # CSV delle tabelle
│   ├── ground_truth/           # domanda reale (solo valutazione)
│   └── processed/              # panel e predizioni
├── output/
│   ├── figure/                 # PDF
│   └── tabelle/                # CSV e .tex
├── esegui_tutto.sh             # pipeline completa (macOS/Linux)
├── esegui_tutto.ps1            # pipeline completa (Windows)
└── requirements.txt
```

---

## Avvio

1. Clona il repository e crea l'ambiente virtuale:

```bash
   git clone https://github.com/giozoc/autoflow-analytics.git
   cd autoflow-analytics
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
```

2. Esegui l'intera pipeline:

```bash
   ./esegui_tutto.sh                # Windows: .\esegui_tutto.ps1
```

   In alternativa, passo per passo (dalla radice del progetto, con `PYTHONPATH=src`):

```bash
   python src/genera_dataset.py --config config/parametri.yaml
   python src/valida_dataset.py
   python src/kpi.py
   python src/modelli.py
   python src/scoring.py
   python src/valutazione.py
```

L'esecuzione richiede meno di un minuto. Con lo stesso seed (`42`) e le stesse versioni delle librerie i dati generati sono identici.

---

## Licenza

Questo progetto è distribuito sotto licenza **GNU GPL v3.0**.
