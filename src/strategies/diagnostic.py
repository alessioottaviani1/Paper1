import pandas as pd
for s in ['btp_italia','cds_bond_basis','itraxx_combined']:
    df = pd.read_csv(f'results/{s}/index_daily.csv', index_col=0, parse_dates=True)
    iv = df['index_value'].dropna()      # equity curve GIÀ pronta dal pipeline
    rm = iv.cummax()
    dd = iv/rm - 1.0
    trough = dd.idxmin()
    peak_value = rm.loc[trough]
    after = iv.loc[trough:]
    recovered = after[after >= peak_value]
    if len(recovered):
        rec = recovered.index[0]
        print(f'{s}: maxDD {dd.min()*100:.1f}% @ {trough.date()}, recovered {rec.date()} ({(rec-trough).days} giorni)')
    else:
        print(f'{s}: maxDD {dd.min()*100:.1f}% @ {trough.date()}, NOT recovered (final {iv.iloc[-1]:.3f} vs peak {peak_value:.3f})')