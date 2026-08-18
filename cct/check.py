import pandas as pd
S = pd.read_csv(r".\data\processed\cct\coupon_schedule.csv", parse_dates=["fixing_date"])
def stat(df, label):
    tot = len(df); pop = int(df["param_ann"].notna().sum())
    print(f"{label}: {tot} cedole | popolate {pop} ({round(100*pop/tot) if tot else 0}%)")
stat(S[(S["rule"]=="A")&(S["fixing_date"]>="1995-01-01")&(S["fixing_date"]<="1998-12-31")], "regola A 1995-98 (esteso)")
stat(S[(S["rule"]=="B")&(S["fixing_date"]>="1995-01-01")&(S["fixing_date"]<"2002-01-01")], "regola B 1995-2001")
print("flag A:", S[S["rule"]=="A"]["param_series"].value_counts().to_dict() if "param_series" in S.columns else "assente")