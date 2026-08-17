"""
02 - Download completo da Bloomberg. Nessun argomento: si lancia e basta, si puo' rilanciare.

METODO: identico a quello collaudato in inflation_linked/bbg.py, che risolve tre problemi
che il codice ingenuo non vede (dettagli in bbg.py):
  - richieste per BLOOMBERG ID + " Corp", non per ISIN;
  - conversione del formato narwhals LUNGO che la versione installata di xbbg restituisce
    (leggere .iloc[:,0] su quell'oggetto darebbe la colonna dei ticker, in silenzio);
  - blocchi con throttle e fallback per singolo ticker.

CACHE E CHECKPOINT. Ogni titolo scaricato va in cache come parquet e il CSV di blocco viene
riscritto ogni 50 titoli: se il limite giornaliero scatta a meta', si riparte dai soli
mancanti. Il limite (-4002 WORKFLOW_REVIEW_NEEDED) viene riconosciuto, non confuso con un
errore di ticker: lo script si ferma, salva e lo dichiara.

Output: PROC/px_{bot,btp,cct}.csv, PROC/curves_market.csv, PROC/static_bbg.csv
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

try:
    import bbg
    from bbg import BbgLimitReached
    HAVE_BBG, IMPORT_ERR = True, None
except Exception as e:
    HAVE_BBG, IMPORT_ERR = False, str(e)

START, END = pd.Timestamp(START_EXTENDED), pd.Timestamp(END_SAMPLE)
S_STR, E_STR = START.strftime("%Y-%m-%d"), END.strftime("%Y-%m-%d")
STATIC_FIELDS = ["CPN", "CPN_FREQ", "MATURITY", "FIRST_SETTLE_DT", "AMT_OUTSTANDING",
                 "DAY_CNT_DES", "FLT_SPREAD", "RESET_IDX", "CPN_TYP"]
CURVES = {**index_tickers(), **swap_tickers(), **ois_tickers()}

def pull_one(bb_id, isin, field=PRICE_FIELD):
    f = CACHE / f"{isin}_{field}.parquet"
    if f.exists():
        try: return pd.read_parquet(f).iloc[:, 0], "cache"
        except Exception: f.unlink()
    try:
        d = bbg.bdh([bbg.ticker(bb_id)], field, S_STR, E_STR, verbose=False)
        if d is None or d.empty: return None, "empty"
        s = pd.to_numeric(d.iloc[:, 0], errors="coerce").dropna()
        if s.empty: return None, "empty"
        s.name = isin; s.to_frame().to_parquet(f)
        return s, "ok"
    except BbgLimitReached: return None, "limit"
    except Exception: return None, "error"

def pull_block(pairs, label, P):
    out, cnt = {}, {"cache": 0, "ok": 0, "empty": 0, "limit": 0, "error": 0}
    for i, (bb_id, isin) in enumerate(pairs, 1):
        s, st = pull_one(bb_id, isin); cnt[st] += 1
        if s is not None and not s.empty: out[isin] = s
        if st == "limit":
            P(f"  [LIMITE BLOOMBERG] dopo {i}/{len(pairs)} titoli di {label}.")
            P(f"  {cnt['cache']+cnt['ok']} gia' in cache: rilanciare domani, riparte da li'.")
            break
        if i % 50 == 0:
            print(f"  {label}: {i}/{len(pairs)} (nuovi {cnt['ok']}, cache {cnt['cache']})")
            if out: pd.DataFrame(out).sort_index().to_csv(PROC / f"px_{label}.csv")
    if out: pd.DataFrame(out).sort_index().to_csv(PROC / f"px_{label}.csv")
    P(f"  {label:4s}: {len(out)}/{len(pairs)} serie | nuovi {cnt['ok']}, cache {cnt['cache']}, "
      f"vuoti {cnt['empty']}, errori {cnt['error']}")
    return cnt

if __name__ == "__main__":
    print("== 02 pull prices ==")
    L = []; P = L.append
    P("=== 02 DOWNLOAD PREZZI E CURVE ===")
    if not HAVE_BBG:
        P(f"[STOP] modulo bbg/xbbg non disponibile: {IMPORT_ERR}")
        P("  xbbg e' gia' nei requirements (>=0.7.7): verificare l'installazione del venv.")
        P("  Il terminale Bloomberg deve essere APERTO E LOGGATO sulla stessa macchina.")
        save_txt("02_pull.txt", L); print("\n".join(L)); raise SystemExit
    P(f"identificatore: Bloomberg ID + ' Corp' | campo {PRICE_FIELD} | {START.date()} -> {END.date()}")
    P(f"blocchi bdh {bbg.CHUNK_BDH} / bdp {bbg.CHUNK_BDP} | throttle {bbg.THROTTLE_SEC}s")

    hit = False
    for lab, f in [("bot", "static_bot.csv"), ("btp", "static_btp.csv"), ("cct", "static_cct.csv")]:
        d = pd.read_csv(PROC / f, parse_dates=["maturity"])
        d = d[d.maturity > START].dropna(subset=["bb_id", "isin"])
        pairs = list(zip(d["bb_id"].astype(str), d["isin"].astype(str)))
        P(f"\n{lab.upper()} ({len(pairs)} titoli):")
        if pull_block(pairs, lab, P)["limit"]: hit = True; break

    if not hit:
        P(f"\nCURVE DI MERCATO ({len(CURVES)} serie: {len(YEARS_SWAP)} IRS + {len(YEARS_SWAP)} OIS + indici):")
        cu = {}
        for tk, nm in CURVES.items():
            try:
                d = bbg.bdh([tk], "PX_LAST", S_STR, E_STR, verbose=False)
                if d is not None and not d.empty:
                    cu[nm] = pd.to_numeric(d.iloc[:, 0], errors="coerce").dropna()
                    P(f"  {nm:10s} ok ({len(cu[nm])})")
                else: P(f"  {nm:10s} VUOTO")
            except BbgLimitReached: P(f"  {nm:10s} LIMITE"); hit = True; break
            except Exception as e: P(f"  {nm:10s} FAIL {str(e)[:60]}")
        if cu: pd.DataFrame(cu).sort_index().to_csv(PROC / "curves_market.csv")

    if not hit:
        P("\nANAGRAFICA BLOOMBERG (serve FLT_SPREAD per i CCTeu):")
        idmap = pd.concat([pd.read_csv(PROC / f)[["bb_id", "isin"]] for f in
                           ["static_cct.csv", "static_btp.csv", "static_bot.csv"]]).dropna()
        try:
            S = bbg.bdp([bbg.ticker(b) for b in idmap["bb_id"].astype(str)], STATIC_FIELDS)
            if not S.empty:
                back = {bbg.ticker(b): i for b, i in zip(idmap["bb_id"].astype(str), idmap["isin"])}
                S.index = [back.get(str(x), str(x)) for x in S.index]
                S.to_csv(PROC / "static_bbg.csv"); P(f"  {len(S)} righe -> [saved] static_bbg.csv")
        except BbgLimitReached:
            P("  [LIMITE] anagrafica non completata: rilanciare domani."); hit = True

    P("\n" + ("RILANCIARE domani lo stesso comando: la cache riprende da dove si e' fermato."
             if hit else "Download completo. Procedere con 03_bot_auction.py"))
    save_txt("02_pull.txt", L); print("\n".join(L))
