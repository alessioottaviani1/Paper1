# CFA Quant Awards 2026 — Checklist (v2, aggiornata dopo la review del build)

## FATTO (v2)
- [x] 16 placeholder in appendice → risolti: nomi reali recuperati da
  paper1_appendix.tex / paper1_tables_final.tex (incl. figure GIRF e DCC in
  results/rq3_duffie/figures/). Al prossimo build sulla tua macchina spariscono.
- [x] 2 riferimenti "??" → risolti: aggiunta la sottosezione B con
  \label{app:oos_rolling} + oos_alpha_appendix_monthly, e incluso factor_list
  (che definisce tab:factor_list).
- [x] Tre errori fattuali nella MIA bozza corretti dopo ri-verifica sul paper:
  (1) moltiplicatore di regime: è 3–7× sui NOVE incroci strategia–benchmark,
  non "2–7× per ogni control set" (iTraxx best-subset è ~1.4×); (2) i selettori
  alternativi sono CINQUE incluso il knockoffs (non 5+knockoffs); (3) abstract
  "doubles to triples" → "up to sevenfold under the benchmark controls".
- [x] Aggiunte al corpo (condensazioni dal paper, ~330 parole): meccanica dei
  trade (soglie, % target-exit 92.3/90.7/97.1, durate ~13/8/7 mesi), stabilità
  sub-period + rolling 36m, esposizioni ricorrenti + double-selection, medie
  DCC (−0.29/−0.10), decadimento GIRF (2–3 mesi) + robustezza a 2–3 lag.
- [x] Warning "duplicate identifier page.1" → cover a pagina 0.
- [x] Tutte le celle di Tabelle 1–3 ri-verificate cifra per cifra sul compilato
  del paper (Table I, II, XII, VI-C, IX, XIV, XIX, A.I, A.II).

## DA FARE — contenuto (in ordine di valore)
- [ ] **Sezione 5 (Implications for Investors)**: unica sezione vuota, ~550
  parole TUE. È il motivo principale per cui il corpo è corto.
- [ ] **Fine campione vs n**: May 2025 dichiarato, ma n=163/207/243 ⇒ Ott/Nov
  2025 (May ⇒ 158/201/237) e le figure stampano Oct/Nov 2025. Decidere e
  allineare TUTTO (prosa, Tabella 1, tabelle pipeline, figure).
- [ ] **Figura 1 (equity curves)**: rigenerare con la data giusta, poi
  decommentare il blocco in qa_sec2 (vale ~0.5 pp di corpo).
- [ ] Rigenerare rolling_alpha_composite se mai usata ("CDS--Bond" letterale).

## Budget lunghezza (dopo audit dei vincitori)
- Corpo attuale: ~3.070 parole ≈ 4.7 pp → SOTTO la fascia (regole 5–7;
  vincitori 2.900–4.800, primi premi 3.800–4.800).
- Con Sez. 5 (+550) + Figura 1 (+0.5 pp) → ~4.950 parole ≈ 6.8 pp. Target.

## Compliance (invariata)
- [ ] Registrazione: email a quantawards@cfafrance.org (scaduta il 31/5,
  case-by-case). Deadline consegna 31 agosto.
- Cover solo titolo · metadata anonimi (build_qa.sh verifica) · niente link al
  repo (cognome nell'URL) · filename esatto col nome.
- Regola AI: structuring/editorial only → Sez. 5 e i punti ALEX-WRITE tuoi.

## Nota per il paper JF (non per il concorso)
- §I.B "SW dominates EW on both Sharpe and MPPM for all three": falso per lo
  Sharpe iTraxx (EW 1.30 > SW 1.25). Riformulare su MPPM prima della submission.
