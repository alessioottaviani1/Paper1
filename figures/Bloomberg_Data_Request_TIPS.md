# Bloomberg Data Request — TIPS–Treasury Limits-to-Arbitrage Project

**For:** Alessio Ottaviani (EDHEC PhD). **Purpose:** close the data gaps that currently limit the structural TIPS paper (daily stress identification, transaction-cost/repo gate, term structure of the basis, European extension).

**General settings for every pull (Bloomberg Excel add-in, BDH):**
- **Frequency:** DAILY (use `Per=cd` calendar daily, fill `previous`).
- **History:** from **1998-01-01** (or earliest available) to today. Pull the longest history; trim later.
- **Fields:** `PX_LAST` always. Where bid/ask is relevant also pull `PX_BID`, `PX_ASK` (bonds: `YLD_YTM_MID`, `YLD_YTM_BID`, `YLD_YTM_ASK`). For equities used as factors: `TOT_RETURN_INDEX_GROSS_DVDS`.
- Save each block as a separate sheet/file with the ticker in the header row (same layout as the existing `TIPS_Data.xlsx`).
- Over-pull: where I list tenors 1–30Y, grab them all even if we use a subset.

Priority tags: **[P1]** essential to fix the current limitations · **[P2]** strengthens identification/robustness · **[P3]** European extension.

---

## BLOCK A — USD inflation-swap curve & breakevens  [P1]
*Needed to re-derive the basis independently and to build the term structure of λ (currently we only have the 5y5y forward).*

- **Zero-coupon USD CPI inflation swaps, full curve:** `USSWIT1 Curncy`, `USSWIT2`, `USSWIT3`, `USSWIT4`, `USSWIT5`, `USSWIT6`, `USSWIT7`, `USSWIT8`, `USSWIT9`, `USSWIT10`, `USSWIT12`, `USSWIT15`, `USSWIT20`, `USSWIT25`, `USSWIT30 Curncy`. (If `USSWIT` doesn't resolve on your terminal, build the same tenors from `SWPM`/`ILBE <GO>` → USD CPI zero-coupon swaps.)
- **Forwards (already have 5y5y `FWISUS55`):** also `FWISUS11`, `FWISUS22`, `FWISUS1010` if available.
- **US breakevens, full curve:** `USGGBE02 Index`, `USGGBE05` (have), `USGGBE10`, `USGGBE20`, `USGGBE30 Index`.

## BLOCK B — TIPS real curve & STRIPS  [P1]
*The third leg of the FLL replication (coupon matching) and the maturity-resolved basis.*

- **TIPS generic real yields:** `USGGT02Y Index`, `USGGT05Y`, `USGGT07Y`, `USGGT10Y`, `USGGT20Y`, `USGGT30Y Index`.
- **Gürkaynak–Sack–Wright fitted curves (gold standard):** the TIPS curve and nominal curve parameters — these are the Fed files `feds200628` (nominal) and `feds200805` (TIPS), already in the repo, but pull the **latest vintage** from the Fed (they update; see "Non-Bloomberg" below).
- **Nominal Treasury STRIPS:** principal STRIPS prices/yields across maturities (`S <govt> <GO>` → STRIPS), or the nominal zero curve `I025 <GO>`. Pull the zero curve 1–30Y if individual STRIPS are awkward.
- **Individual benchmark TIPS (optional, [P2]):** the on-the-run + recent off-the-run TIPS by CUSIP with `PX_BID/PX_ASK` and `YLD_YTM_*` — lets us match the existing 107-CUSIP panel and get bid-ask per security.

## BLOCK C — Nominal Treasury curve (fill the gaps)  [P1]
*We have 1M, 2Y, 5Y, 10Y, 30Y. Complete the curve.*

- `USGG3M Index`, `USGG6M`, `USGG12M` (or `USGG1YR`), `USGG3YR`, `USGG7YR`, `USGG20YR Index`.
- Bills for the repo–bill spread: `USB1M`, `USB3M`, `USB6M Index` (or use `USGG3M`).

## BLOCK D — Funding & repo (DAILY)  [P1]  ← the single most important new block
*This is what recovers March-2020 in the factors and gives the stress identification beyond 2008-09.*

- **Repo (general collateral):** `BGCR Index` (Broad GC Rate), `TGCR Index` (Tri-party GC), `SOFRRATE Index` (SOFR). For the **pre-2018 history**: DTCC GCF Treasury repo and the primary-dealer overnight Treasury repo — find via `RRRA <GO>` / `MMR <GO>`; pull the longest GC Treasury repo series available.
- **Term SOFR / OIS:** `USOSFR3 Curncy` (3M SOFR OIS) or the OIS curve `USSO1`, `USSO3`, `USSO6`, `USSO12 Curncy`.
- **LIBOR (historical, for LIBOR-OIS):** `US0001M Index`, `US0003M Index`, `US0006M Index` (3M is the key one).
- **TED & spreads (or construct):** `.TEDSP` if available, else we build US0003M − USGG3M.
- **TIPS-specific repo / specialness [P1, hard]:** any TIPS repo rates and special-vs-GC spreads your terminal exposes (`RRRA <GO>`, dealer repo runs). Pull whatever exists, even partial/short history — this is the gate-1 input and currently entirely missing. Also note: **haircuts are not on Bloomberg** (we'll use literature/clearing values).
- **Bid-ask on the basis legs [P1]:** `PX_BID`/`PX_ASK` for the benchmark TIPS, the maturity-matched nominal Treasury, and the inflation swaps (Block A) — for the transaction-cost band.

## BLOCK E — Intermediary-capital & dealer-stress factors (DAILY)  [P1/P2]
*Daily proxies for the He–Kelly–Manela channel (the monthly HKM factor stays as the benchmark; these give daily identifying variation).*

- **Primary-dealer parent equities (build a daily dealer-equity factor):** total-return index for `GS US Equity`, `MS US Equity`, `JPM US Equity`, `BAC US Equity`, `C US Equity`, `WFC US Equity`, `BK US Equity`, `STT US Equity`, `BCS LN Equity`, `BARC LN Equity`, `DBK GR Equity`, `UBSG SW Equity`, `BNP FP Equity`, `ACA FP Equity`, `RY CN Equity`, `TD CN Equity`. (Field `TOT_RETURN_INDEX_GROSS_DVDS`.)
- **Dealer 5Y senior CDS:** the 5Y senior CDS of the same dealers — via `CDSW <GO>` or `<ticker> CDS USD SR 5Y Corp` (e.g. Goldman, Morgan Stanley, JPM, BofA, Citi, BNP, Deutsche, Barclays, UBS).
- **Financial-conditions indices:** `BFCIUS Index` (have), plus `NFCIINDX Index` (Chicago Fed NFCI, weekly) if exposed.

## BLOCK F — Volatility, credit & liquidity (DAILY)  [P2]
- **Vol (have, re-pull to extend):** `VIX Index`, `MOVE Index`. Add `SRVIX Index` (swaption vol) and `CVIX Index`.
- **Credit indices:** CDX.NA.IG 5Y on-the-run and CDX.NA.HY — via `CDX <GO>` (generic on-the-run, e.g. `IBOXUMAE Index` for IG if it resolves); iTraxx Main & Crossover 5Y — `ITRXEBE Index`, `ITRXEXE Index`.
- **Government-bond liquidity:** `GVLQUSD Index` (the Bloomberg USD govt liquidity index already in the HPW file — re-pull daily, full history). This is our daily noise/illiquidity proxy if the constructed Hu-Pan-Wang isn't feasible.
- **Excess bond premium [P2, non-Bloomberg]:** Gilchrist–Zakrajšek EBP — monthly, from the Fed (see below).

## BLOCK G — European inflation-linked extension  [P3]
*Cross-country + multi-episode variation that directly fixes the one-episode identification problem.*

- **Euro nominal curves:** Germany `GDBR2/5/10/30 Index`; France `GTFRF2/5/10/30 Index` (or `GFRN`); Italy `GBTPGR2/5/10/30 Index`. Add the bill/short rates: `GETB1`, German `GDBR1`.
- **Euro zero-coupon inflation swaps (HICP ex-tobacco):** `EUSWI1 Curncy` … `EUSWI30 Curncy` (tenors 1,2,3,4,5,6,7,8,9,10,12,15,20,25,30). 5y5y forward: `EUSWI5F5` (or via `SWPM`). If `EUSWI` doesn't resolve, use `ILBE <GO>` → EUR HICPxT zero-coupon swaps.
- **Euro real / inflation-linked bonds (real yields):**
  - **Germany (Bundei/OBLei):** generic German ILB real yields, or pull the individual linkers by ISIN (`ILBE <GO>` → Germany).
  - **France (OAT€i = HICP-linked, OATi = French-CPI-linked):** real yields / individual linkers by ISIN.
  - **Italy (BTP€i = HICP-linked):** real yields / individual linkers by ISIN. *(BTP Italia from Paper 1 is retail/Italian-CPI and is a different instrument — keep it separate.)*
  - For each: `PX_BID/PX_ASK`, `YLD_YTM_*`, maturity, coupon (so we can replicate the basis per FLL).
- **Euro funding/repo:** `EUR003M Index` (3M Euribor), `ESTRON Index` (€STR), euro OIS `EUSWE3`/`EESWE` curve; euro GC repo / RepoFunds Rate (`RRRA <GO>` → EUR), Bund repo specialness if exposed.
- **Euro stress:** `V2X Index` (VSTOXX), German-Italian 10Y spread `.GERIT10` (or construct `GBTPGR10 − GDBR10`), `ITRXEBE`/`ITRXEXE` (already in F).

---

## Non-Bloomberg sources (download separately)
These are not on Bloomberg; grab them so the panel is complete:
1. **He–Kelly–Manela intermediary capital ratio & risk factor** — Zhiguo He's site (zhiguohe.net → "Intermediary Asset Pricing" data) and Asaf Manela's site. Monthly. We have the factor; also get the **raw capital-ratio level** and the **latest vintage**.
2. **Gilchrist–Zakrajšek Excess Bond Premium** — Federal Reserve (Favara et al. updated EBP series). Monthly CSV.
3. **GSW fitted curves** — Fed: `feds200628` (nominal Treasury) and `feds200805` (TIPS) — latest vintages.
4. **Primary-dealer positions/financing** — NY Fed primary dealer statistics (weekly), for a dealer-balance-sheet proxy.

---

## If you want a strict priority order (download P1 first)
1. **Block D** (funding & repo daily, incl. TIPS specialness & leg bid-ask) — fixes gate-1 and gives March-2020.
2. **Block A + B + C** (full inflation-swap curve, TIPS real curve, STRIPS, nominal gaps) — re-derive λ and its term structure.
3. **Block E** (daily dealer-equity & CDS factors) — daily intermediary-capital identification.
4. **Block F** (vol/credit/liquidity daily) — completes the factor panel.
5. **Block G** (European linkers) — the cross-country extension; the biggest single boost to stress identification.

Everything daily, max history, with bid/ask where I flagged it. Pull generously — extra tenors and extra dealers cost nothing and we can trim.
