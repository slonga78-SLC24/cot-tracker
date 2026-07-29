import os
import json
import datetime
import numpy as np
import pandas as pd
import cot_reports as cot
import gspread
from google.oauth2.service_account import Credentials

# Unified contract search keywords (case-insensitive search)
# This prevents broken matches if CFTC changes minor spacing or wording
ASSET_KEYWORDS = {
    "S&P 500": "S&P 500",
    "NASDAQ-100": "Nasdaq 100",
    "RUSSELL E-MINI": "Russell 2000",
    "VIX FUTURES": "VIX",
    "MSCI EMERGING": "MSCI EM",
    "30-DAY FEDERAL": "30d Fed Funds",
    "2-YEAR TREASURY": "2y Treasury",
    "5-YEAR TREASURY": "5y Treasury",
    "10-YEAR TREASURY": "10y Treasury",
    "U.S. TREASURY BONDS": "Treasury Bonds",
    "U.S. DOLLAR INDEX": "USD",
    "EURO FX": "EUR",
    "BRITISH POUND": "GBP",
    "JAPANESE YEN": "JPY",
    "AUSTRALIAN DOLLAR": "AUS",
    "CANADIAN DOLLAR": "CAD",
    "BRAZILIAN REAL": "BRL",
    "BITCOIN": "Bitcoin",
    "GOLD": "Gold",
    "SILVER": "Silver",
    "COPPER": "Copper",
    "CRUDE OIL": "Crude Oil",
    "WHEAT": "Wheat"
}

def compute_z_score(latest, mean, std):
    if std == 0 or pd.isna(std):
        return 0.0
    return round((latest - mean) / std, 2)

def main():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")

    if not creds_json or not sheet_id:
        raise ValueError("Missing environment variables for Google authentication.")

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    print("Fetching historical CFTC data...")
    current_year = datetime.datetime.now().year
    years = list(range(2018, current_year + 1))
    
    # Use Legacy Futures report as primary source (covers Financials + Commodities)
    df_list = []
    for y in years:
        try:
            df_year = cot.cot_year(year=y, cot_report_type='legacy_fut')
            df_list.append(df_year)
        except Exception as e:
            print(f"Warning year {y}: {e}")

    df_raw = pd.concat(df_list, ignore_index=True)
    df_raw['As_of_Date_In_Form_YYMMDD'] = pd.to_datetime(df_raw['As_of_Date_In_Form_YYMMDD'])
    df_raw = df_raw.sort_values('As_of_Date_In_Form_YYMMDD')

    summary_rows = []

    for keyword, clean_name in ASSET_KEYWORDS.items():
        # Fuzzy match contract names
        mask = df_raw['Market_and_Exchange_Names'].str.contains(keyword, case=False, na=False)
        asset_df = df_raw[mask].copy()

        if asset_df.empty:
            continue

        # Non-Commercial (Speculator) Net Position calculation
        if 'NonComm_Positions_Long_All' in asset_df.columns and 'NonComm_Positions_Short_All' in asset_df.columns:
            asset_df['Net_Pos'] = asset_df['NonComm_Positions_Long_All'] - asset_df['NonComm_Positions_Short_All']
        else:
            continue

        net_series = asset_df['Net_Pos'].dropna()
        if len(net_series) == 0:
            continue

        latest_val = net_series.iloc[-1]
        prev_val = net_series.iloc[-2] if len(net_series) > 1 else np.nan
        ww_change = latest_val - prev_val

        avg_3m = net_series.tail(12).mean()

        # 1Y Metrics
        net_1y = net_series.tail(52)
        avg_1y, min_1y, max_1y, std_1y = net_1y.mean(), net_1y.min(), net_1y.max(), net_1y.std()
        z_1y = compute_z_score(latest_val, avg_1y, std_1y)

        # Historical Metrics (Since 2018)
        avg_hist, min_hist, max_hist, std_hist = net_series.mean(), net_series.min(), net_series.max(), net_series.std()
        z_hist = compute_z_score(latest_val, avg_hist, std_hist)

        summary_rows.append([
            clean_name,
            int(latest_val),
            int(ww_change) if not pd.isna(ww_change) else 0,
            int(avg_3m),
            int(avg_1y), int(min_1y), int(max_1y), z_1y,
            int(avg_hist), int(min_hist), int(max_hist), z_hist
        ])

    headers = [
        "Non Commercials Net Long", "Latest", "W/W Chg", "3M Avg",
        "1Y Avg", "1Y Min", "1Y Max", "1Y Z-Score",
        "Since 2018 Avg", "Since 2018 Min", "Since 2018 Max", "Since 2018 Z-Score"
    ]

    sh = gc.open_by_key(sheet_id)
    worksheet = sh.sheet1
    worksheet.clear()
    
    worksheet.update('A1', [headers] + summary_rows)
    print("Google Sheet updated successfully with data!")

if __name__ == "__main__":
    main()
