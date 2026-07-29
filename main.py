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

# Asset mapping covering dynamic CFTC contract naming variations
ASSET_PATTERNS = {
    "S&P 500": ["E-MINI S&P 500", "S&P 500 STOCK INDEX"],
    "Nasdaq 100": ["NASDAQ-100 STOCK INDEX", "E-MINI NASDAQ-100"],
    "Russell 2000": ["RUSSELL 2000 MINI INDEX", "E-MINI RUSSELL 2000"],
    "VIX": ["VIX FUTURES"],
    "MSCI EM": ["MSCI EMERGING MARKETS INDEX"],
    "30d Fed Funds": ["30-DAY FEDERAL FUNDS"],
    "Treasury Bonds": ["U.S. TREASURY BONDS"],
    "USD": ["U.S. DOLLAR INDEX", "USD INDEX"],
    "EUR": ["EURO FX"],
    "GBP": ["BRITISH POUND"],
    "JPY": ["JAPANESE YEN"],
    "AUS": ["AUSTRALIAN DOLLAR"],
    "CAD": ["CANADIAN DOLLAR"],
    "BRL": ["BRAZILIAN REAL"],
    "Bitcoin": ["BITCOIN"],
    "Gold": ["GOLD - COMMODITY EXCHANGE"],
    "Silver": ["SILVER - COMMODITY EXCHANGE"],
    "Copper": ["COPPER - COMMODITY EXCHANGE"],
    "Crude Oil": ["CRUDE OIL, LIGHT SWEET", "CRUDE OIL LIGHT SWEET"],
    "Wheat": ["WHEAT - CHICAGO BOARD OF TRADE"]
}

def parse_cot_dates(series):
    """Safely converts dynamic CFTC date columns (YYMMDD vs YYYY-MM-DD) to datetime."""
    str_series = series.astype(str).str.split('.').str[0].str.strip()
    sample_val = str_series.dropna().iloc[0] if not str_series.dropna().empty else ""
    if len(sample_val) == 6 and sample_val.isdigit():
        return pd.to_datetime(str_series, format='%y%m%d', errors='coerce')
    else:
        return pd.to_datetime(str_series, errors='coerce')

def compute_z_score(latest, mean, std):
    if std == 0 or pd.isna(std):
        return 0.0
    return round((latest - mean) / std, 2)

def get_trading_permissions(z_1y):
    """
    Determines directional trading permissions based on 1Y Z-Score Regimes:
    - Z <= -2.0 : Bullish Capitulation (LONG-ONLY)
    - Z >= +2.0 : Overbought / Squeeze Risk (SHORT-ONLY / NO LONGS)
    - +1.0 <= Z < +2.0 : Bullish Trend (LONG-ONLY / DIP BUYING)
    - Else : Neutral Zone (UNRESTRICTED)
    """
    if z_1y <= -2.0:
        return "BULLISH CAPITULATION", "🟢 LONG-ONLY (High Conviction)"
    elif z_1y >= 2.0:
        return "BEARISH OVERBOUGHT", "🔴 SHORT-ONLY / NO LONGS"
    elif z_1y >= 1.0:
        return "BULLISH TREND", "🟢 LONG-ONLY (Dip Buying)"
    else:
        return "NEUTRAL ZONE", "⚪ UNRESTRICTED (Both Long & Short)"

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
    long_col = next((c for c in df_raw.columns if 'noncomm' in c.lower() and 'long' in c.lower()), None)
    short_col = next((c for c in df_raw.columns if 'noncomm' in c.lower() and 'short' in c.lower()), None)

    # Clean dates and net positions
    df_raw['Clean_Date'] = parse_cot_dates(df_raw[date_col])
    df_raw['Market_Upper'] = df_raw[name_col].astype(str).str.strip().str.upper()
    df_raw['Net_Pos'] = df_raw[long_col] - df_raw[short_col]

    def map_asset(market_str):
        for clean_name, patterns in ASSET_PATTERNS.items():
            if any(p in market_str for p in patterns):
                return clean_name
        return None

    df_raw['Clean_Asset'] = df_raw['Market_Upper'].apply(map_asset)
    df_filtered = df_raw.dropna(subset=['Clean_Asset', 'Clean_Date', 'Net_Pos']).copy()

    pivot_net = df_filtered.pivot_table(
        index='Clean_Date', 
        columns='Clean_Asset', 
        values='Net_Pos', 
        aggfunc='max'
    ).sort_index()

    summary_rows = []
    alert_assets = []
    permission_rows = []

    for clean_name in ASSET_PATTERNS.keys():
        if clean_name not in pivot_net.columns:
            continue

        net_series = pivot_net[clean_name].dropna()
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

        # Get Regime and Directional Permission
        regime, permission = get_trading_permissions(z_1y)

        summary_rows.append([
            clean_name,
            int(latest_val),
            int(ww_change) if not pd.isna(ww_change) else 0,
            int(avg_3m),
            int(avg_1y), int(min_1y), int(max_1y), z_1y,
            int(avg_hist), int(min_hist), int(max_hist), z_hist,
            permission
        ])

        # Store permission status for key trading assets in Telegram summary
        if clean_name in ["S&P 500", "Nasdaq 100", "Russell 2000", "USD", "Gold"]:
            permission_rows.append({
                "name": clean_name,
                "z_1y": z_1y,
                "regime": regime,
                "permission": permission
            })

        # Flag threshold alerts (|Z| >= 1.0)
        if abs(z_1y) >= 1.0 or abs(z_hist) >= 1.0:
            alert_assets.append({
                "name": clean_name,
                "z_1y": z_1y,
                "z_hist": z_hist
            })

    headers = [
        "Non Commercials Net Long", "Latest", "W/W Chg", "3M Avg",
        "1Y Avg", "1Y Min", "1Y Max", "1Y Z-Score",
        "Since 2018 Avg", "Since 2018 Min", "Since 2018 Max", "Since 2018 Z-Score",
        "System Directional Permission"
    ]

    sh = gc.open_by_key(sheet_id)
    worksheet = sh.sheet1
    worksheet.clear()
    
    worksheet.update('A1', [headers] + summary_rows)
    print("Sheet updated!")

    # Format Telegram Message
    if bot_token and chat_id:
        msg_lines = ["📊 *COT POSITIONING & SYSTEM PERMISSIONS*\n"]
        
        # SECTION 1: Directional System Permissions
        msg_lines.append("🎯 *System Trading Permissions (SPY/QQQ/Macro):*")
        for p in permission_rows:
            msg_lines.append(
                f"• *{p['name']}* (`1Y Z: {p['z_1y']:+.2f}`)\n"
                f"   └ Rule: *{p['permission']}*"
            )
        
        # SECTION 2: Active Threshold Alerts
        if alert_assets:
            msg_lines.append("\n🚨 *Active Z-Score Alerts (|Z| ≥ 1.0):*")
            for item in alert_assets:
                icon_1y = "🔴" if item['z_1y'] > 0 else "🟢"
                icon_hist = "🔴" if item['z_hist'] > 0 else "🟢"
                msg_lines.append(
                    f"• *{item['name']}*\n"
                    f"   └ 1Y Z-Score: {icon_1y} `{item['z_1y']:+.2f}` | Since 2018: {icon_hist} `{item['z_hist']:+.2f}`"
                )

        full_message = "\n".join(msg_lines)
        send_telegram_alert(bot_token, chat_id, full_message)

if __name__ == "__main__":
    main()
