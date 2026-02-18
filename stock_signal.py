#!/usr/bin/env python3
"""
Indian Stock Market Signal System
Combines technical analysis (RSI, MACD, EMA, Bollinger Bands, Volume)
with fundamental analysis (P/E, P/B, D/E, growth, promoter holding)
to generate BUY/SELL signals sent via Telegram.

Data: Zerodha Kite API (falls back to yfinance if token unavailable)
Schedule: GitHub Actions — 9:30 AM IST and 3:00 PM IST, weekdays only
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, date, timedelta

import pytz
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST_FILE   = "watchlist.json"
STATE_FILE       = "signals_state.json"
LOG_FILE         = "signals_log.json"
IST              = pytz.timezone("Asia/Kolkata")

LOOKBACK_DAYS    = 120      # days of OHLCV history to fetch
MIN_DATA_POINTS  = 60       # minimum candles needed to compute all indicators

RSI_OVERSOLD     = 30
RSI_OVERBOUGHT   = 70
VOLUME_SPIKE_MULT = 2.0     # volume > 2x 20-day avg = spike

# Fundamental scoring weights (must sum to 100)
FUND_WEIGHTS = {
    "pe_score":        20,
    "pb_score":        15,
    "de_score":        15,
    "revenue_growth":  15,
    "profit_growth":   15,
    "promoter_holding": 20,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

def get_price_data_kite(symbol, api_key, access_token):
    """
    Fetch daily OHLCV from Zerodha Kite API.
    Returns DataFrame[date, open, high, low, close, volume].
    Raises on any failure so caller can fall back to yfinance.
    """
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Resolve instrument token
    instruments = kite.instruments("NSE")
    token = None
    for inst in instruments:
        if inst["tradingsymbol"] == symbol:
            token = inst["instrument_token"]
            break
    if token is None:
        raise ValueError(f"Symbol '{symbol}' not found in NSE instruments list")

    to_date   = date.today()
    from_date = to_date - timedelta(days=LOOKBACK_DAYS)

    candles = kite.historical_data(token, from_date, to_date, "day")
    if not candles:
        raise ValueError(f"Kite returned no data for {symbol}")

    df = pd.DataFrame(candles)
    df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()

    if len(df) < MIN_DATA_POINTS:
        raise ValueError(f"Kite: only {len(df)} data points for {symbol} (need {MIN_DATA_POINTS})")

    return df


def get_price_data_yfinance(symbol):
    """
    Fetch daily OHLCV from Yahoo Finance (NSE via .NS suffix).
    Returns DataFrame[date, open, high, low, close, volume].
    Raises on failure.
    """
    yf_symbol = symbol + ".NS"
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period="6mo", interval="1d")

    if df.empty:
        raise ValueError(f"yfinance returned no data for {yf_symbol}")

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "date"})

    # Strip timezone
    if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["close"])

    if len(df) < MIN_DATA_POINTS:
        raise ValueError(f"yfinance: only {len(df)} data points for {symbol} (need {MIN_DATA_POINTS})")

    return df


def get_price_data(symbol, api_key, access_token):
    """
    Dispatcher: try Kite first, fall back to yfinance.
    Returns (DataFrame, data_source_string).
    """
    if api_key and access_token:
        try:
            df = get_price_data_kite(symbol, api_key, access_token)
            log.info(f"{symbol}: data from Kite API ({len(df)} candles)")
            return df, "Kite API"
        except Exception as e:
            log.warning(f"{symbol}: Kite failed ({e}) — falling back to yfinance")

    try:
        df = get_price_data_yfinance(symbol)
        log.info(f"{symbol}: data from yfinance ({len(df)} candles)")
        return df, "yfinance"
    except Exception as e:
        log.error(f"{symbol}: yfinance also failed — {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(df):
    """RSI(14): < 30 = BUY, > 70 = SELL."""
    try:
        series = ta.rsi(df["close"], length=14)
        if series is None or series.dropna().empty:
            return {"signal": "NEUTRAL", "value": None, "reason": "RSI: insufficient data"}

        val = series.dropna().iloc[-1]
        if val < RSI_OVERSOLD:
            return {"signal": "BUY",     "value": val, "reason": f"RSI={val:.1f} (oversold <{RSI_OVERSOLD})"}
        elif val > RSI_OVERBOUGHT:
            return {"signal": "SELL",    "value": val, "reason": f"RSI={val:.1f} (overbought >{RSI_OVERBOUGHT})"}
        else:
            return {"signal": "NEUTRAL", "value": val, "reason": f"RSI={val:.1f}"}
    except Exception as e:
        return {"signal": "NEUTRAL", "value": None, "reason": f"RSI error: {e}"}


def compute_macd(df):
    """MACD(12,26,9): BUY on bullish crossover, SELL on bearish crossover."""
    try:
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_df is None or len(macd_df.dropna()) < 2:
            return {"signal": "NEUTRAL", "macd": None, "signal_line": None,
                    "reason": "MACD: insufficient data"}

        macd_col   = [c for c in macd_df.columns if c.startswith("MACD_")][0]
        signal_col = [c for c in macd_df.columns if c.startswith("MACDs_")][0]

        macd_line   = macd_df[macd_col].dropna()
        signal_line = macd_df[signal_col].dropna()

        # Align
        common_idx  = macd_line.index.intersection(signal_line.index)
        macd_line   = macd_line.loc[common_idx]
        signal_line = signal_line.loc[common_idx]

        if len(macd_line) < 2:
            return {"signal": "NEUTRAL", "macd": None, "signal_line": None,
                    "reason": "MACD: not enough aligned data"}

        prev_diff    = macd_line.iloc[-2] - signal_line.iloc[-2]
        current_diff = macd_line.iloc[-1] - signal_line.iloc[-1]

        m_val = macd_line.iloc[-1]
        s_val = signal_line.iloc[-1]

        if prev_diff < 0 and current_diff > 0:
            return {"signal": "BUY",  "macd": m_val, "signal_line": s_val,
                    "reason": "MACD crossed above signal (bullish)"}
        elif prev_diff > 0 and current_diff < 0:
            return {"signal": "SELL", "macd": m_val, "signal_line": s_val,
                    "reason": "MACD crossed below signal (bearish)"}
        else:
            sig = "BUY" if current_diff > 0 else "SELL"
            direction = "above" if current_diff > 0 else "below"
            return {"signal": sig, "macd": m_val, "signal_line": s_val,
                    "reason": f"MACD={m_val:.2f} {direction} signal={s_val:.2f}"}
    except Exception as e:
        return {"signal": "NEUTRAL", "macd": None, "signal_line": None,
                "reason": f"MACD error: {e}"}


def compute_ema_crossover(df):
    """EMA9 vs EMA21 crossover: BUY on golden cross, SELL on death cross."""
    try:
        ema9  = ta.ema(df["close"], length=9)
        ema21 = ta.ema(df["close"], length=21)

        if ema9 is None or ema21 is None:
            return {"signal": "NEUTRAL", "ema9": None, "ema21": None,
                    "reason": "EMA: insufficient data"}

        ema9  = ema9.dropna()
        ema21 = ema21.dropna()
        common = ema9.index.intersection(ema21.index)
        ema9  = ema9.loc[common]
        ema21 = ema21.loc[common]

        if len(ema9) < 2:
            return {"signal": "NEUTRAL", "ema9": None, "ema21": None,
                    "reason": "EMA: not enough aligned data"}

        prev_diff    = ema9.iloc[-2] - ema21.iloc[-2]
        current_diff = ema9.iloc[-1] - ema21.iloc[-1]
        e9 = ema9.iloc[-1]
        e21 = ema21.iloc[-1]

        if prev_diff < 0 and current_diff > 0:
            return {"signal": "BUY",  "ema9": e9, "ema21": e21,
                    "reason": f"EMA9 crossed above EMA21 (golden cross)"}
        elif prev_diff > 0 and current_diff < 0:
            return {"signal": "SELL", "ema9": e9, "ema21": e21,
                    "reason": f"EMA9 crossed below EMA21 (death cross)"}
        elif current_diff > 0:
            return {"signal": "BUY",  "ema9": e9, "ema21": e21,
                    "reason": f"EMA9({e9:.1f}) > EMA21({e21:.1f}) — uptrend"}
        else:
            return {"signal": "SELL", "ema9": e9, "ema21": e21,
                    "reason": f"EMA9({e9:.1f}) < EMA21({e21:.1f}) — downtrend"}
    except Exception as e:
        return {"signal": "NEUTRAL", "ema9": None, "ema21": None,
                "reason": f"EMA error: {e}"}


def compute_bollinger_bands(df):
    """
    Bollinger Bands(20, 2): BUY near lower band, SELL near upper band.
    lower/upper are used as support/resistance levels in alerts.
    """
    try:
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is None or bb.dropna().empty:
            return {"signal": "NEUTRAL", "lower": None, "middle": None,
                    "upper": None, "price": None, "reason": "BB: insufficient data"}

        lower_col  = [c for c in bb.columns if c.startswith("BBL_")][0]
        middle_col = [c for c in bb.columns if c.startswith("BBM_")][0]
        upper_col  = [c for c in bb.columns if c.startswith("BBU_")][0]

        lower  = bb[lower_col].dropna().iloc[-1]
        middle = bb[middle_col].dropna().iloc[-1]
        upper  = bb[upper_col].dropna().iloc[-1]
        price  = df["close"].iloc[-1]

        tolerance = (upper - lower) * 0.01  # within 1% of band = "touching"

        if price <= lower + tolerance:
            return {"signal": "BUY",  "lower": lower, "middle": middle,
                    "upper": upper, "price": price,
                    "reason": f"Price(₹{price:.1f}) at lower BB(₹{lower:.1f})"}
        elif price >= upper - tolerance:
            return {"signal": "SELL", "lower": lower, "middle": middle,
                    "upper": upper, "price": price,
                    "reason": f"Price(₹{price:.1f}) at upper BB(₹{upper:.1f})"}
        else:
            return {"signal": "NEUTRAL", "lower": lower, "middle": middle,
                    "upper": upper, "price": price,
                    "reason": f"Price in BB range [₹{lower:.1f}–₹{upper:.1f}]"}
    except Exception as e:
        return {"signal": "NEUTRAL", "lower": None, "middle": None,
                "upper": None, "price": None, "reason": f"BB error: {e}"}


def compute_volume_spike(df):
    """
    Volume > 2× 20-day average = spike.
    This is an amplifier — does NOT vote on BUY/SELL direction.
    """
    try:
        if len(df) < 21:
            return {"spike": False, "ratio": 0.0, "reason": "Volume: insufficient data"}

        avg_vol    = df["volume"].iloc[-21:-1].mean()
        today_vol  = df["volume"].iloc[-1]

        if avg_vol <= 0:
            return {"spike": False, "ratio": 0.0, "reason": "Volume: avg is zero"}

        ratio = today_vol / avg_vol
        spike = ratio >= VOLUME_SPIKE_MULT
        return {
            "spike": spike,
            "ratio": round(ratio, 2),
            "today_volume": int(today_vol),
            "avg_volume":   int(avg_vol),
            "reason": f"Volume {ratio:.1f}x 20-day avg {'⚠️ SPIKE' if spike else ''}"
        }
    except Exception as e:
        return {"spike": False, "ratio": 0.0, "reason": f"Volume error: {e}"}


def evaluate_technical_signals(df):
    """
    Run all indicators and aggregate into a direction.
    4 voting indicators: RSI, MACD, EMA, BB (need ≥ 3/4 to agree).
    Volume is non-voting — only adds a spike note.
    """
    rsi  = compute_rsi(df)
    macd = compute_macd(df)
    ema  = compute_ema_crossover(df)
    bb   = compute_bollinger_bands(df)
    vol  = compute_volume_spike(df)

    voters = [rsi, macd, ema, bb]
    buy_count  = sum(1 for i in voters if i["signal"] == "BUY")
    sell_count = sum(1 for i in voters if i["signal"] == "SELL")

    if buy_count >= 3:
        direction = "BUY"
    elif sell_count >= 3:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    triggered = [i["reason"] for i in voters if i["signal"] == direction and direction != "NEUTRAL"]

    return {
        "direction":   direction,
        "buy_count":   buy_count,
        "sell_count":  sell_count,
        "triggered":   triggered,
        "rsi":         rsi,
        "macd":        macd,
        "ema":         ema,
        "bb":          bb,
        "volume":      vol,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FUNDAMENTAL ANALYSIS (always via yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def get_fundamentals_yfinance(symbol):
    """
    Fetch fundamental data for an NSE symbol via yfinance.
    Returns a dict with all fields; missing values are None.
    Note: promoter holding is approximated from major_holders (insider %).
    """
    result = {
        "pe": None, "pb": None, "de": None,
        "revenue_qoq": None, "revenue_yoy": None,
        "profit_qoq": None, "profit_yoy": None,
        "promoter_pct": None,
        "fetch_error": None
    }
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info   = ticker.info or {}

        result["pe"] = info.get("trailingPE") or info.get("forwardPE")
        result["pb"] = info.get("priceToBook")
        result["de"] = info.get("debtToEquity")

        # Quarterly financials → QoQ growth
        try:
            qf = ticker.quarterly_financials
            if qf is not None and not qf.empty and len(qf.columns) >= 2:
                for row_label, key_qoq in [("Total Revenue", "revenue_qoq"),
                                            ("Net Income",    "profit_qoq")]:
                    if row_label in qf.index:
                        prev = qf.loc[row_label].iloc[1]
                        curr = qf.loc[row_label].iloc[0]
                        if prev and prev != 0:
                            result[key_qoq] = round((curr / prev - 1) * 100, 1)
        except Exception:
            pass

        # Annual financials → YoY growth
        try:
            af = ticker.financials
            if af is not None and not af.empty and len(af.columns) >= 2:
                for row_label, key_yoy in [("Total Revenue", "revenue_yoy"),
                                            ("Net Income",    "profit_yoy")]:
                    if row_label in af.index:
                        prev = af.loc[row_label].iloc[1]
                        curr = af.loc[row_label].iloc[0]
                        if prev and prev != 0:
                            result[key_yoy] = round((curr / prev - 1) * 100, 1)
        except Exception:
            pass

        # Promoter holding (approximate from major_holders)
        try:
            holders = ticker.major_holders
            if holders is not None and not holders.empty:
                raw = str(holders.iloc[0, 0]).replace("%", "").strip()
                result["promoter_pct"] = float(raw)
        except Exception:
            pass

    except Exception as e:
        result["fetch_error"] = str(e)
        log.warning(f"Fundamentals fetch error for {symbol}: {e}")

    return result


def _score_pe(pe, sector_pe):
    if pe is None: return 50
    if pe <= 0:    return 20
    if pe < sector_pe * 0.8:  return 90
    if pe < sector_pe:        return 70
    if pe < sector_pe * 1.2:  return 50
    if pe < sector_pe * 1.5:  return 30
    return 10


def _score_pb(pb):
    if pb is None: return 50
    if pb <= 1:    return 90
    if pb <= 3:    return 70
    if pb <= 5:    return 50
    if pb <= 8:    return 30
    return 10


def _score_de(de):
    if de is None: return 50
    if de < 0:     return 30
    if de == 0:    return 95
    if de < 0.5:   return 85
    if de < 1.0:   return 70
    if de < 2.0:   return 50
    if de < 3.0:   return 30
    return 10


def _score_growth(val):
    if val is None: return 50
    if val > 25:    return 95
    if val > 15:    return 80
    if val > 5:     return 65
    if val >= 0:    return 50
    if val > -10:   return 30
    return 10


def _score_promoter(pct):
    if pct is None: return 50
    if pct >= 65:   return 90
    if pct >= 50:   return 75
    if pct >= 35:   return 60
    if pct >= 20:   return 40
    return 20


def compute_fundamental_score(fundamentals, sector_pe):
    """
    Convert raw fundamental data into a 0-100 weighted score.
    Returns {total, breakdown} where breakdown shows raw values and sub-scores.
    """
    f = fundamentals

    revenue_val = f["revenue_yoy"] if f["revenue_yoy"] is not None else f["revenue_qoq"]
    profit_val  = f["profit_yoy"]  if f["profit_yoy"]  is not None else f["profit_qoq"]

    sub_scores = {
        "pe_score":        _score_pe(f["pe"], sector_pe),
        "pb_score":        _score_pb(f["pb"]),
        "de_score":        _score_de(f["de"]),
        "revenue_growth":  _score_growth(revenue_val),
        "profit_growth":   _score_growth(profit_val),
        "promoter_holding": _score_promoter(f["promoter_pct"]),
    }

    total = sum(sub_scores[k] * FUND_WEIGHTS[k] / 100 for k in FUND_WEIGHTS)

    breakdown = {
        "pe_score":        {"raw": f["pe"],            "score": sub_scores["pe_score"]},
        "pb_score":        {"raw": f["pb"],            "score": sub_scores["pb_score"]},
        "de_score":        {"raw": f["de"],            "score": sub_scores["de_score"]},
        "revenue_growth":  {"raw": revenue_val,        "score": sub_scores["revenue_growth"]},
        "profit_growth":   {"raw": profit_val,         "score": sub_scores["profit_growth"]},
        "promoter_holding":{"raw": f["promoter_pct"],  "score": sub_scores["promoter_holding"]},
    }

    return {"total": round(total, 1), "breakdown": breakdown}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(tech, fund_score):
    """
    Combine technical direction with fundamental score into a 5-tier signal.

    STRONG BUY  → BUY tech + fund ≥ 75
    BUY         → BUY tech + fund ≥ 60
    HOLD        → BUY tech + fund < 60  OR  SELL tech + fund ≥ 75  OR  NEUTRAL
    SELL        → SELL tech + fund ≥ 60
    STRONG SELL → SELL tech + fund < 60
    """
    direction   = tech["direction"]
    fund_strong = fund_score["total"] >= 75
    fund_ok     = fund_score["total"] >= 60

    if direction == "BUY":
        if fund_strong: return "STRONG BUY"
        if fund_ok:     return "BUY"
        return "HOLD"
    elif direction == "SELL":
        if fund_strong: return "HOLD"    # tech says sell but fundamentals excellent → hold
        if fund_ok:     return "SELL"
        return "STRONG SELL"
    else:
        return "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_signals_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_signals_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def has_signal_changed(symbol, new_signal, state):
    prev = state.get(symbol, {}).get("signal")
    return prev != new_signal


def append_to_signal_log(entry):
    existing = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TELEGRAM ALERTS
# ─────────────────────────────────────────────────────────────────────────────

def send_telegram_alert(bot_token, chat_id, text):
    """Send a Telegram message (HTML parse mode). Retries once on failure."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    for attempt in range(2):
        try:
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            return True
        except Exception as e:
            if attempt == 0:
                log.warning(f"Telegram send failed (attempt 1): {e} — retrying in 5s")
                time.sleep(5)
            else:
                log.error(f"Telegram send failed (attempt 2): {e}")
    return False


def _signal_emoji(signal):
    return {
        "STRONG BUY":  "🟢",
        "BUY":         "🟡",
        "HOLD":        "⚪",
        "SELL":        "🟠",
        "STRONG SELL": "🔴",
    }.get(signal, "⚪")


def _indicator_line(label, indicator, direction):
    if indicator.get("signal") == direction and direction != "NEUTRAL":
        mark = "✅"
    elif indicator.get("signal") == "NEUTRAL":
        mark = "⬜"
    else:
        mark = "❌"
    return f"  {mark} {indicator.get('reason', label)}"


def format_alert_message(symbol, price, signal, tech, fund_score,
                         data_source, run_time, prev_signal):
    """Build the full Telegram HTML message for a signal alert."""
    emoji = _signal_emoji(signal)
    direction = tech["direction"]

    # Header
    lines = [
        "━━━━━━━━━━━━━━━━━━━━",
        f"{emoji} <b>{signal} — {symbol}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💰 Price: <code>₹{price:,.2f}</code>  |  Source: {data_source}",
        f"📅 {run_time.strftime('%d %b %Y, %H:%M IST')}",
        "",
    ]

    # Technical section
    agree = tech["buy_count"] if direction == "BUY" else tech["sell_count"]
    lines.append(f"📊 <b>Technical ({agree}/4 indicators agree)</b>:")
    for ind_name, ind_data in [("RSI", tech["rsi"]), ("MACD", tech["macd"]),
                                ("EMA", tech["ema"]), ("BB",  tech["bb"])]:
        lines.append(_indicator_line(ind_name, ind_data, direction))

    vol = tech["volume"]
    if vol.get("spike"):
        lines.append(f"  ⚠️  {vol['reason']}")

    lines.append("")

    # Fundamentals section
    fs = fund_score
    bd = fs["breakdown"]
    lines.append(f"📈 <b>Fundamental Score: {fs['total']}/100</b>")

    pe_raw  = bd["pe_score"]["raw"]
    pb_raw  = bd["pb_score"]["raw"]
    de_raw  = bd["de_score"]["raw"]
    rev_raw = bd["revenue_growth"]["raw"]
    prf_raw = bd["profit_growth"]["raw"]
    prom_raw= bd["promoter_holding"]["raw"]

    lines.append(f"  PE: {f'{pe_raw:.1f}' if pe_raw else 'N/A'}  → {bd['pe_score']['score']}pts")
    lines.append(f"  PB: {f'{pb_raw:.1f}' if pb_raw else 'N/A'}  → {bd['pb_score']['score']}pts")
    lines.append(f"  D/E: {f'{de_raw:.2f}' if de_raw else 'N/A'}  → {bd['de_score']['score']}pts")
    lines.append(f"  Revenue growth: {f'{rev_raw:+.1f}%' if rev_raw is not None else 'N/A'}  → {bd['revenue_growth']['score']}pts")
    lines.append(f"  Profit growth: {f'{prf_raw:+.1f}%' if prf_raw is not None else 'N/A'}  → {bd['profit_growth']['score']}pts")
    lines.append(f"  Promoter holding: {f'{prom_raw:.1f}%' if prom_raw is not None else 'N/A'}  → {bd['promoter_holding']['score']}pts")

    lines.append("")

    # Support / Resistance from Bollinger Bands
    bb = tech["bb"]
    if bb.get("lower") and bb.get("upper"):
        lines.append(f"🎯 Support: ₹{bb['lower']:,.1f}  |  Resistance: ₹{bb['upper']:,.1f}")
        lines.append(f"   (Bollinger Bands 20-day)")

    lines.append(f"📌 Previous signal: {prev_signal or 'None (first run)'}")
    lines.append("━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load configuration
    api_key      = os.environ.get("KITE_API_KEY", "")
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "")
    bot_token    = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id      = os.environ.get("TELEGRAM_CHAT_ID", "")
    run_time     = datetime.now(IST)

    if not bot_token or not chat_id:
        log.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars are required")
        sys.exit(1)

    # 2. Load watchlist
    if not os.path.exists(WATCHLIST_FILE):
        log.error(f"{WATCHLIST_FILE} not found")
        sys.exit(1)

    with open(WATCHLIST_FILE) as f:
        watchlist = json.load(f)["stocks"]

    # 3. Load previous signal state
    state = load_signals_state()

    # 4. Send scan-start notification
    kite_status = "connected" if (api_key and access_token) else "not configured (using yfinance)"
    send_telegram_alert(
        bot_token, chat_id,
        f"📡 <b>Signal scan started</b>\n"
        f"🕐 {run_time.strftime('%d %b %Y, %H:%M IST')}\n"
        f"📋 Scanning {len(watchlist)} stocks\n"
        f"🔌 Kite API: {kite_status}"
    )

    # 5. Scan each stock
    alerts_sent = 0
    errors      = 0
    signal_counts = {"STRONG BUY": 0, "BUY": 0, "HOLD": 0, "SELL": 0, "STRONG SELL": 0}

    for stock_meta in watchlist:
        symbol    = stock_meta["symbol"]
        sector_pe = stock_meta["sector_pe"]

        log.info(f"--- Processing {symbol} ---")

        # a. Fetch price data
        df, data_source = get_price_data(symbol, api_key, access_token)
        if df is None:
            log.warning(f"Skipping {symbol}: no price data")
            errors += 1
            time.sleep(1)
            continue

        # b. Technical analysis
        tech = evaluate_technical_signals(df)

        # c. Fundamental analysis
        fundamentals = get_fundamentals_yfinance(symbol)
        fund_score   = compute_fundamental_score(fundamentals, sector_pe)

        # d. Generate signal
        signal = generate_signal(tech, fund_score)
        price  = float(df["close"].iloc[-1])

        signal_counts[signal] = signal_counts.get(signal, 0) + 1
        changed = has_signal_changed(symbol, signal, state)

        log.info(f"{symbol}: {signal} | tech={tech['direction']}({tech['buy_count']}B/{tech['sell_count']}S) | fund={fund_score['total']}/100 | changed={changed}")

        # e. Log to file
        append_to_signal_log({
            "timestamp":            run_time.isoformat(),
            "symbol":               symbol,
            "signal":               signal,
            "price":                price,
            "tech_direction":       tech["direction"],
            "buy_count":            tech["buy_count"],
            "sell_count":           tech["sell_count"],
            "fund_score":           fund_score["total"],
            "data_source":          data_source,
            "indicators_triggered": tech["triggered"],
            "previous_signal":      state.get(symbol, {}).get("signal"),
            "signal_changed":       changed,
            "outcome_price":        None,
            "outcome_date":         None,
            "outcome_pct":          None,
        })

        # f. Update state
        state[symbol] = {
            "signal":    signal,
            "timestamp": run_time.isoformat(),
            "price":     price,
        }

        # g. Send alert only if signal changed
        if changed:
            prev_signal = state.get(symbol, {}).get("signal")
            msg = format_alert_message(
                symbol, price, signal, tech, fund_score,
                data_source, run_time, prev_signal
            )
            if send_telegram_alert(bot_token, chat_id, msg):
                alerts_sent += 1

        time.sleep(1)  # rate limit yfinance

    # 6. Save updated state (GitHub Actions commits this)
    save_signals_state(state)

    # 7. Send summary
    summary = (
        f"✅ <b>Scan complete</b> — {run_time.strftime('%d %b %Y, %H:%M IST')}\n\n"
        f"📊 Results ({len(watchlist)} stocks):\n"
        f"  🟢 STRONG BUY: {signal_counts.get('STRONG BUY', 0)}\n"
        f"  🟡 BUY:        {signal_counts.get('BUY', 0)}\n"
        f"  ⚪ HOLD:       {signal_counts.get('HOLD', 0)}\n"
        f"  🟠 SELL:       {signal_counts.get('SELL', 0)}\n"
        f"  🔴 STRONG SELL:{signal_counts.get('STRONG SELL', 0)}\n\n"
        f"📨 Alerts sent (signal changes): {alerts_sent}\n"
        f"⚠️  Data errors: {errors}"
    )
    send_telegram_alert(bot_token, chat_id, summary)

    log.info("Scan complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "")
        if bot_token and chat_id:
            send_telegram_alert(bot_token, chat_id, f"❌ <b>System error</b>\n{e}")
        sys.exit(1)
