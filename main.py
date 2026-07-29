import os
import json
import datetime
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import cot_reports as cot
import gspread
from google.oauth2.service_account import Credentials

EXACT_ASSET_MAP = {
    "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "S&P 500",
    "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "Nasdaq 100",
    "RUSSELL 2000 MINI INDEX FUTURE - CHICAGO MERCANTILE EXCHANGE": "Russell 2000",
    "VIX FUTURES - CBOE FUTURES EXCHANGE": "VIX",
    "MSCI EMERGING MARKETS INDEX - CHICAGO MERCANTILE EXCHANGE": "MSCI EM",
    "30-DAY FEDERAL FUNDS - CHICAGO BOARD OF TRADE": "30d Fed Funds",
    "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE": "Treasury Bonds",
    "U.S. DOLLAR INDEX - ICE FUTURES U.S.": "USD",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "EUR",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "JPY",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "AUS",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "CAD",
    "BRAZILIAN REAL - CHICAGO MERCANTILE EXCHANGE": "BRL",
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "Bitcoin",
    "GOLD - COMMODITY EXCHANGE INC.": "Gold",
    "SILVER - COMMODITY EXCHANGE INC.": "Silver",
    "COPPER - COMMODITY EXCHANGE INC.": "Copper",
    "CRUDE OIL LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE": "Crude Oil",
    "WHEAT - CHICAGO BOARD OF TRADE": "Wheat"
}

def compute_z_score(latest, mean, std):
    if std == 0 or pd.isna(std):
        return 0.0
    return round((latest - mean) / std, 2)

def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        print("Telegram credentials missing. Skipping alert.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print("Telegram alert sent successfully!")
            else:
                print(f"Failed to send Telegram alert: Status {response.status}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def main():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not creds_json or not sheet_id:
        raise ValueError("Missing environment variables for Google authentication.")

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    print("Fetching historical CFTC data...")
    current_year = datetime.datetime.now().year
    years = list(range(2018, current_year + 1))
    
    df_list = []
    for y in years:
        try:
            df_year = cot.cot_year(year=y, cot_report_type='legacy_fut')
            df_list.append(df_year)
        except Exception as e:
            print(f"Warning year {y}: {e}")

    df_raw = pd.concat(df_list, ignore_index=True)

    date_col = next((c for c in df_raw.columns if 'date' in c.lower() or 'yymmdd' in c.lower()), None)
    name_col = next((c for c in df_raw.columns if 'market' in c.lower() or 'name' in c.lower()), None)

    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    df_raw = df_raw.sort_values(date_col)

    long_col = next((c for c in df_raw.columns if 'noncomm' in c.lower() and 'long' in c.lower()), None)
    short_col = next((c for c in df_raw.columns if 'noncomm' in c.lower() and 'short' in c.lower()), None)

    summary_rows = []
    alert_assets = []

    for exact_cftc_name, clean_name in EXACT_ASSET_MAP.items():
        asset_df = df_raw[df_raw[name_col].astype(str).str.strip() == exact_cftc_name].copy()

        if asset_df.empty:
            continue

        asset_df['Net_Pos'] = asset_df[long_col] - asset_df[short_col]
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

        # Flag if 1Y OR Historical Z-Score is >= 1.0 or <= -1.0
        if abs(z_1y) >= 1.0 or abs(z_hist) >= 1.0:
            alert_assets.append({
                "name": clean_name,
                "z_1y": z_1y,
                "z_hist": z_hist
            })

    headers = [
        "Non Commercials Net Long", "Latest", "W/W Chg", "3M Avg",
        "1Y Avg", "1Y Min", "1Y Max", "1Y Z-Score",
        "Since 2018 Avg", "Since 2018 Min", "Since 2018 Max", "Since 2018 Z-Score"
    ]

    sh = gc.open_by_key(sheet_id)
    worksheet = sh.sheet1
    worksheet.clear()
    
    worksheet.update('A1', [headers] + summary_rows)
    print("Sheet updated!")

    # Format Telegram Message
    if alert_assets and bot_token and chat_id:
        msg_lines = ["📊 *COT Positioning Alert (|Z| ≥ 1.0)*\n"]
        for item in alert_assets:
            msg_lines.append(
                f"• *{item['name']}*\n"
                f"   └ 1Y Z-Score: `{item['z_1y']:+.2f}`\n"
                f"   └ Since 2018 Z-Score: `{item['z_hist']:+.2f}`"
            )
        
        full_message = "\n".join(msg_lines)
        send_telegram_alert(bot_token, chat_id, full_message)
    elif not alert_assets:
        print("No assets triggered the Z-score threshold (|Z| >= 1.0).")

if __name__ == "__main__":
    main()
