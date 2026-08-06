# ============================================================
# import_data.py
# Import asset pricing data from Sina.
# ============================================================

import akshare as ak
import pandas as pd

# US Indices
assets = {
    ".INX": ("SP500", "index"),
    ".IXIC": ("NASDAQ", "index"),
    ".DJI": ("DOW", "index"),
}

# Chinese ETFs
early_etfs = {
    "sh510050": ("SSE50", "etf"),     
    "sh510880": ("DIVIDEND", "etf"),    
}

# ETFs starting 2011-2013
mid_etfs = {
    "sz159915": ("CHINEXT", "etf"),    
    "sz159934": ("GOLD", "etf"),        
}

# ETFs starting ~2015
late_etfs = {
    "sh513600": ("FTSE100", "etf"),     
    "sz159941": ("NASDAQ2", "etf"),     
    "sh513800": ("JAPAN", "etf"),       
}

all_data = {}

# Fetch indices
for sym, (name, src) in assets.items():
    df = ak.index_us_stock_sina(symbol=sym)
    all_data[name] = df.set_index(pd.to_datetime(df["date"]))["close"]
    print(f"✓ {name}: {df.iloc[0]['date']} to {df.iloc[-1]['date']} ({len(df)} rows)")

# Fetch early ETFs
for code, (name, src) in early_etfs.items():
    df = ak.fund_etf_hist_sina(symbol=code)
    all_data[name] = df.set_index(pd.to_datetime(df["date"]))["close"]
    print(f"✓ {name}: {df.iloc[0]['date']} to {df.iloc[-1]['date']} ({len(df)} rows)")

# Fetch mid ETFs
for code, (name, src) in mid_etfs.items():
    df = ak.fund_etf_hist_sina(symbol=code)
    all_data[name] = df.set_index(pd.to_datetime(df["date"]))["close"]
    print(f"✓ {name}: {df.iloc[0]['date']} to {df.iloc[-1]['date']} ({len(df)} rows)")

# Fetch late ETFs
for code, (name, src) in late_etfs.items():
    df = ak.fund_etf_hist_sina(symbol=code)
    all_data[name] = df.set_index(pd.to_datetime(df["date"]))["close"]
    print(f"✓ {name}: {df.iloc[0]['date']} to {df.iloc[-1]['date']} ({len(df)} rows)")
    

# Option A: Longest history, moderate diversification (2007-2026)
portfolio_a = ["SP500", "NASDAQ", "DOW", "SSE50", "DIVIDEND"]
df_a = pd.concat({k: all_data[k] for k in portfolio_a}, axis=1).dropna()
print(f"Portfolio A (5 assets, from 2007): {len(df_a)} rows, {df_a.columns.tolist()}")
print(f"  Date range: {df_a.index[0].date()} to {df_a.index[-1].date()}")

# Option B: Add CHINEXT and GOLD (2013-2026)
portfolio_b = portfolio_a + ["CHINEXT", "GOLD"]
df_b = pd.concat({k: all_data[k] for k in portfolio_b}, axis=1).dropna()
print(f"Portfolio B (7 assets, from 2013): {len(df_b)} rows, {df_b.columns.tolist()}")
print(f"  Date range: {df_b.index[0].date()} to {df_b.index[-1].date()}")

# Option C: Full global (2015-2026)
portfolio_c = portfolio_b + ["FTSE100"]
df_c = pd.concat({k: all_data[k] for k in portfolio_c}, axis=1).dropna()
print(f"Portfolio C (8 assets, from 2015): {len(df_c)} rows, {df_c.columns.tolist()}")
print(f"  Date range: {df_c.index[0].date()} to {df_c.index[-1].date()}")

# Save all three options
df_a.to_csv("portfolio_A_5assets_2007.csv")
df_b.to_csv("portfolio_B_7assets_2013.csv")
df_c.to_csv("portfolio_C_8assets_2015.csv")