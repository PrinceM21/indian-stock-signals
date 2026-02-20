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
import requests
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# PURE NUMPY/PANDAS INDICATOR HELPERS (no extra library needed)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(series, length=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def _macd(series, fast=12, slow=26, signal=9):
    ema_fast   = _ema(series, fast)
    ema_slow   = _ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line= _ema(macd_line, signal)
    return macd_line, signal_line

def _bbands(series, length=20, std=2):
    middle = series.rolling(length, min_periods=length).mean()
    stddev = series.rolling(length, min_periods=length).std()
    upper  = middle + std * stddev
    lower  = middle - std * stddev
    return lower, middle, upper

def _atr(df, length=14):
    """
    Average True Range (ATR) using Wilder's smoothing.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    high       = df["high"]
    low        = df["low"]
    close      = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, min_periods=length).mean()

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
        series = _rsi(df["close"], length=14)
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
        macd_line, signal_line = _macd(df["close"], fast=12, slow=26, signal=9)

        # Align and drop NaN
        combined    = pd.concat([macd_line, signal_line], axis=1).dropna()
        if len(combined) < 2:
            return {"signal": "NEUTRAL", "macd": None, "signal_line": None,
                    "reason": "MACD: insufficient data"}

        macd_line   = combined.iloc[:, 0]
        signal_line = combined.iloc[:, 1]

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
        ema9  = _ema(df["close"], length=9)
        ema21 = _ema(df["close"], length=21)

        combined = pd.concat([ema9, ema21], axis=1).dropna()
        ema9  = combined.iloc[:, 0]
        ema21 = combined.iloc[:, 1]

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
        lower_s, middle_s, upper_s = _bbands(df["close"], length=20, std=2)
        combined = pd.concat([lower_s, middle_s, upper_s], axis=1).dropna()
        if combined.empty:
            return {"signal": "NEUTRAL", "lower": None, "middle": None,
                    "upper": None, "price": None, "reason": "BB: insufficient data"}

        lower  = combined.iloc[-1, 0]
        middle = combined.iloc[-1, 1]
        upper  = combined.iloc[-1, 2]
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

    # ATR(14) for stop-loss / target calculation
    try:
        atr_series = _atr(df, length=14)
        atr_value  = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else None
    except Exception:
        atr_value = None

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
        "atr":         atr_value,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FUNDAMENTAL ANALYSIS (always via yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def get_fundamentals_yfinance(symbol):
    """
    Fetch fundamental data for an NSE symbol via yfinance.
    Returns a dict with all fields needed for the 10-point checklist.
    Missing values are None.
    """
    result = {
        # Checklist fields
        "opm":              None,   # Operating Profit Margin (criterion 1)
        "eps":              None,   # Trailing EPS            (criterion 2)
        "de":               None,   # Debt-to-Equity          (criterion 3)
        "roe":              None,   # Return on Equity        (criterion 4)
        # ROCE (criterion 5) not available for NSE via yfinance → auto-skip
        "net_income":       None,   # Net income (for interest coverage, crit 6)
        "interest_expense": None,   # Interest expense        (criterion 6)
        "ebitda":           None,   # EBITDA (fallback proxy for crit 6)
        "promoter_pct":     None,   # Promoter / insider %    (criterion 7)
        "operating_cashflow": None, # Operating cash flow     (criterion 8)
        "revenue_yoy":      None,   # Revenue YoY growth %    (criterion 9 proxy)
        "profit_yoy":       None,   # Profit YoY growth %     (criterion 10 proxy)
        # Growth fallbacks
        "revenue_qoq":      None,
        "profit_qoq":       None,
        # Extra context fields (for alert message)
        "pe":               None,
        "week52_high":      None,
        "week52_low":       None,
        "market_cap":       None,
        "current_price":    None,
        "fetch_error":      None,
    }
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info   = ticker.info or {}

        # --- Core checklist fields from ticker.info ---
        result["opm"]             = info.get("operatingMargins")       # e.g. 0.24 = 24%
        result["eps"]             = info.get("trailingEps")
        result["de"]              = info.get("debtToEquity")
        result["roe"]             = info.get("returnOnEquity")         # e.g. 0.18 = 18%
        result["operating_cashflow"] = info.get("operatingCashflow")
        result["ebitda"]          = info.get("ebitda")

        # --- Extra context ---
        result["pe"]              = info.get("trailingPE") or info.get("forwardPE")
        result["week52_high"]     = info.get("fiftyTwoWeekHigh")
        result["week52_low"]      = info.get("fiftyTwoWeekLow")
        result["market_cap"]      = info.get("marketCap")
        result["current_price"]   = info.get("currentPrice") or info.get("regularMarketPrice")

        # --- Annual financials → YoY growth + interest expense ---
        try:
            af = ticker.financials
            if af is not None and not af.empty and len(af.columns) >= 2:
                # Revenue YoY
                for label in ("Total Revenue", "Revenue"):
                    if label in af.index:
                        prev = af.loc[label].iloc[1]
                        curr = af.loc[label].iloc[0]
                        if prev and prev != 0:
                            result["revenue_yoy"] = round((curr / prev - 1) * 100, 1)
                        break
                # Net income YoY
                for label in ("Net Income", "Net Income Common Stockholders"):
                    if label in af.index:
                        prev = af.loc[label].iloc[1]
                        curr = af.loc[label].iloc[0]
                        if prev and prev != 0:
                            result["profit_yoy"] = round((curr / prev - 1) * 100, 1)
                        # Store latest net income for interest coverage
                        result["net_income"] = af.loc[label].iloc[0]
                        break
                # Interest expense
                for label in ("Interest Expense", "Interest Expense Non Operating"):
                    if label in af.index:
                        result["interest_expense"] = abs(af.loc[label].iloc[0])
                        break
        except Exception:
            pass

        # --- Quarterly fallback for growth ---
        try:
            qf = ticker.quarterly_financials
            if qf is not None and not qf.empty and len(qf.columns) >= 2:
                for label in ("Total Revenue", "Revenue"):
                    if label in qf.index:
                        prev = qf.loc[label].iloc[1]
                        curr = qf.loc[label].iloc[0]
                        if prev and prev != 0:
                            result["revenue_qoq"] = round((curr / prev - 1) * 100, 1)
                        break
                for label in ("Net Income", "Net Income Common Stockholders"):
                    if label in qf.index:
                        prev = qf.loc[label].iloc[1]
                        curr = qf.loc[label].iloc[0]
                        if prev and prev != 0:
                            result["profit_qoq"] = round((curr / prev - 1) * 100, 1)
                        break
        except Exception:
            pass

        # --- Promoter holding (approximated from major_holders insider %) ---
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


def compute_fundamental_score(fundamentals, sector_pe=None):
    """
    Strict pass/fail checklist matching the user's 10-point criteria.

    Criteria:
      1. OPM > 20%
      2. EPS > 0 (positive / stable)
      3. D/E < 1
      4. ROE > 15%
      5. ROCE > 15%  (SKIPPED — not available via yfinance for NSE)
      6. Net Profit ≥ 2× Interest Expense  (or EBITDA proxy)
      7. Promoter holding ≥ 50%
      8. Operating Cash Flow > 0
      9. Revenue Growth > 10% YoY  (proxy for "Balance Sheet growing")
     10. Profit Growth > 10% YoY   (proxy for "10-year growth > 10%")

    Returns:
      {
        "passed": int (0-10, ROCE always skipped → max useful = 9),
        "total":  int (same as passed, for backward compat),
        "checklist": [ {name, value, passed, note}, ... ]
      }
    """
    f = fundamentals
    revenue_val = f["revenue_yoy"] if f["revenue_yoy"] is not None else f["revenue_qoq"]
    profit_val  = f["profit_yoy"]  if f["profit_yoy"]  is not None else f["profit_qoq"]

    # Interest coverage: net_income ≥ 2× interest_expense
    # Fallback: if interest_expense is 0 or None, treat as PASS (no significant debt interest)
    def _interest_coverage_pass():
        ie = f.get("interest_expense")
        ni = f.get("net_income")
        ebitda = f.get("ebitda")
        if ie is None or ie == 0:
            # No interest burden → automatically passes
            return True, "No significant interest expense"
        # Try net income
        if ni is not None:
            ratio = ni / ie
            if ratio >= 2.0:
                return True, f"Net profit {ratio:.1f}× interest"
            else:
                return False, f"Net profit only {ratio:.1f}× interest (<2×)"
        # Fallback: EBITDA / interest
        if ebitda is not None:
            ratio = ebitda / ie
            if ratio >= 2.0:
                return True, f"EBITDA {ratio:.1f}× interest (proxy)"
            else:
                return False, f"EBITDA only {ratio:.1f}× interest (<2×)"
        return None, "Interest coverage: data unavailable"

    ic_pass, ic_note = _interest_coverage_pass()

    def _fmt_pct(v):
        return f"{v*100:.1f}%" if v is not None else None

    checklist = [
        {
            "name":   "OPM > 20%",
            "value":  _fmt_pct(f["opm"]),
            "passed": (f["opm"] > 0.20) if f["opm"] is not None else None,
            "note":   f"{f['opm']*100:.1f}% operating margin" if f["opm"] is not None else "Data unavailable",
        },
        {
            "name":   "EPS Positive",
            "value":  f"{f['eps']:.2f}" if f["eps"] is not None else None,
            "passed": (f["eps"] > 0) if f["eps"] is not None else None,
            "note":   f"EPS ₹{f['eps']:.2f}" if f["eps"] is not None else "Data unavailable",
        },
        {
            "name":   "D/E < 1",
            "value":  f"{f['de']:.2f}" if f["de"] is not None else None,
            "passed": (f["de"] < 100) if f["de"] is not None else None,
            # yfinance returns D/E in %, e.g. 45.2 means 0.452 — threshold 100 = 1.0
            "note":   f"D/E {f['de']/100:.2f}" if f["de"] is not None else "Data unavailable",
        },
        {
            "name":   "ROE > 15%",
            "value":  _fmt_pct(f["roe"]),
            "passed": (f["roe"] > 0.15) if f["roe"] is not None else None,
            "note":   f"ROE {f['roe']*100:.1f}%" if f["roe"] is not None else "Data unavailable",
        },
        {
            "name":   "ROCE > 15%",
            "value":  None,
            "passed": None,   # Always skipped — not available via yfinance for NSE
            "note":   "Data unavailable (NSE ROCE not in yfinance)",
        },
        {
            "name":   "Net Profit ≥ 2× Interest",
            "value":  ic_note,
            "passed": ic_pass,
            "note":   ic_note,
        },
        {
            "name":   "Promoter Holding ≥ 50%",
            "value":  f"{f['promoter_pct']:.1f}%" if f["promoter_pct"] is not None else None,
            "passed": (f["promoter_pct"] >= 50) if f["promoter_pct"] is not None else None,
            "note":   f"Promoter {f['promoter_pct']:.1f}%" if f["promoter_pct"] is not None else "Data unavailable",
        },
        {
            "name":   "Cash Flow Positive",
            "value":  "Positive" if (f["operating_cashflow"] is not None and f["operating_cashflow"] > 0) else
                      ("Negative" if f["operating_cashflow"] is not None else None),
            "passed": (f["operating_cashflow"] > 0) if f["operating_cashflow"] is not None else None,
            "note":   f"Op. cash flow ₹{f['operating_cashflow']/1e7:.1f} Cr" if f["operating_cashflow"] is not None else "Data unavailable",
        },
        {
            "name":   "Revenue Growth > 10%",
            "value":  f"{revenue_val:+.1f}%" if revenue_val is not None else None,
            "passed": (revenue_val > 10) if revenue_val is not None else None,
            "note":   f"Revenue YoY {revenue_val:+.1f}%" if revenue_val is not None else "Data unavailable",
        },
        {
            "name":   "Profit Growth > 10%",
            "value":  f"{profit_val:+.1f}%" if profit_val is not None else None,
            "passed": (profit_val > 10) if profit_val is not None else None,
            "note":   f"Profit YoY {profit_val:+.1f}%" if profit_val is not None else "Data unavailable",
        },
    ]

    # Count passed (True) — None (data unavailable) does NOT count as pass
    passed_count = sum(1 for c in checklist if c["passed"] is True)

    return {
        "passed":    passed_count,
        "total":     passed_count,   # backward compat with generate_signal / log
        "checklist": checklist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(tech, fund_score):
    """
    Combine technical direction with fundamental checklist score into a 5-tier signal.

    fund_score["passed"] is 0-10 (checklist criteria passed).

    STRONG BUY  → BUY tech  + ≥ 8/10 passed
    BUY         → BUY tech  + ≥ 6/10 passed
    HOLD        → BUY tech  + < 6     OR  SELL tech + ≥ 8  OR  NEUTRAL
    SELL        → SELL tech + ≥ 6/10
    STRONG SELL → SELL tech + < 6/10
    """
    direction  = tech["direction"]
    passed     = fund_score["passed"]
    fund_strong = passed >= 8
    fund_ok     = passed >= 6

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


def _market_cap_label(market_cap):
    """Classify market cap in Indian context (in Crore)."""
    if market_cap is None:
        return "N/A"
    cr = market_cap / 1e7   # 1 Crore = 1e7
    if cr >= 20_000:
        return f"Large Cap (₹{cr/100:.0f}K Cr)"
    elif cr >= 5_000:
        return f"Mid Cap (₹{cr:.0f} Cr)"
    else:
        return f"Small Cap (₹{cr:.0f} Cr)"


def _52w_context(price, high, low):
    """Show where current price sits in 52-week range."""
    if high is None or low is None or high == low:
        return None
    pct_from_high = (high - price) / high * 100
    pct_from_low  = (price - low)  / (high - low) * 100
    return (
        f"₹{low:,.1f} ─── {pct_from_low:.0f}% ─── ₹{price:,.2f} ─── "
        f"{pct_from_high:.0f}% below high ₹{high:,.1f}"
    )


def format_alert_message(symbol, price, signal, tech, fund_score,
                         data_source, run_time, prev_signal,
                         fundamentals=None, sector=None):
    """Build the full Telegram HTML message for a signal alert."""
    emoji = _signal_emoji(signal)
    direction = tech["direction"]
    f = fundamentals or {}

    # ── Header ────────────────────────────────────────────────
    lines = [
        "━━━━━━━━━━━━━━━━━━━━",
        f"{emoji} <b>{signal} — {symbol}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    # Sector + Market Cap
    if sector:
        lines.append(f"🏭 Sector: {sector}  |  {_market_cap_label(f.get('market_cap'))}")

    lines += [
        f"💰 Price: <code>₹{price:,.2f}</code>  |  Source: {data_source}",
        f"📅 {run_time.strftime('%d %b %Y, %H:%M IST')}",
        "",
    ]

    # ── 52-Week Range ─────────────────────────────────────────
    ctx_52w = _52w_context(price, f.get("week52_high"), f.get("week52_low"))
    if ctx_52w:
        lines.append(f"📉 52W: {ctx_52w}")
        lines.append("")

    # ── Technical Section ─────────────────────────────────────
    agree = tech["buy_count"] if direction == "BUY" else tech["sell_count"]
    lines.append(f"📊 <b>Technical ({agree}/4 indicators agree)</b>:")
    for ind_name, ind_data in [("RSI",  tech["rsi"]),
                                ("MACD", tech["macd"]),
                                ("EMA",  tech["ema"]),
                                ("BB",   tech["bb"])]:
        lines.append(_indicator_line(ind_name, ind_data, direction))

    vol = tech["volume"]
    if vol.get("spike"):
        lines.append(f"  ⚠️  {vol['reason']}")

    lines.append("")

    # ── Fundamental Checklist ─────────────────────────────────
    passed  = fund_score["passed"]
    total_c = len(fund_score["checklist"])
    lines.append(f"📋 <b>Fundamental Checklist: {passed}/{total_c} passed</b>")

    for c in fund_score["checklist"]:
        if c["passed"] is None:
            mark = "⚠️ "
        elif c["passed"]:
            mark = "✅"
        else:
            mark = "❌"
        lines.append(f"  {mark} {c['name']}: {c['note']}")

    lines.append("")

    # ── Support / Resistance ──────────────────────────────────
    bb = tech["bb"]
    if bb.get("lower") and bb.get("upper"):
        lines.append(f"🎯 Support: ₹{bb['lower']:,.1f}  |  Resistance: ₹{bb['upper']:,.1f}")
        lines.append(f"   (Bollinger Bands 20-day)")

    # ── ATR-Based Risk Management ─────────────────────────────
    if signal in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
        atr = tech.get("atr")
        if atr and atr > 0:
            if signal in ("BUY", "STRONG BUY"):
                stop_loss = price - 1.5 * atr
                target    = price + 2.0 * atr
                sl_pct    = f"-{(price - stop_loss) / price * 100:.1f}%"
                tgt_pct   = f"+{(target - price) / price * 100:.1f}%"
            else:
                stop_loss = price + 1.5 * atr
                target    = price - 2.0 * atr
                sl_pct    = f"+{(stop_loss - price) / price * 100:.1f}%"
                tgt_pct   = f"-{(price - target) / price * 100:.1f}%"
            risk   = abs(price - stop_loss)
            reward = abs(target - price)
            rr     = reward / risk if risk > 0 else 0
            lines.append(f"🛡️ <b>Risk Management (ATR={atr:.1f})</b>:")
            lines.append(f"  🎯 Target:    ₹{target:,.2f}  ({tgt_pct})")
            lines.append(f"  🛑 Stop-Loss: ₹{stop_loss:,.2f}  ({sl_pct})")
            lines.append(f"  📐 Risk:Reward = 1:{rr:.2f}")
            lines.append("")

    lines.append(f"📌 Previous signal: {prev_signal or 'None (first run)'}")
    lines.append("━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def format_weekly_summary(state, run_time):
    """
    Build the Friday weekly summary Telegram message.
    Reads current signals from state dict (loaded from signals_state.json).
    Shows signal snapshot, top 3 BUY candidates, and any strong signals.
    """
    signal_counts  = {"STRONG BUY": 0, "BUY": 0, "HOLD": 0, "SELL": 0, "STRONG SELL": 0}
    buy_candidates = []   # (fund_passed, symbol, price, signal)
    strong_signals = []   # STRONG BUY / STRONG SELL entries

    for symbol, data in state.items():
        sig = data.get("signal", "HOLD")
        signal_counts[sig] = signal_counts.get(sig, 0) + 1
        fund_passed = data.get("fund_passed", 0) or 0
        price       = data.get("price", 0) or 0
        if sig in ("BUY", "STRONG BUY"):
            buy_candidates.append((fund_passed, symbol, price, sig))
        if sig in ("STRONG BUY", "STRONG SELL"):
            strong_signals.append((symbol, sig, price))

    signal_rank = {"STRONG BUY": 2, "BUY": 1}
    buy_candidates.sort(key=lambda x: (x[0], signal_rank.get(x[3], 0)), reverse=True)
    top3  = buy_candidates[:3]
    total = sum(signal_counts.values())

    lines = [
        "━━━━━━━━━━━━━━━━━━━━",
        "📅 <b>WEEKLY MARKET SUMMARY</b>",
        f"Week ending: {run_time.strftime('%d %b %Y')}",
        "━━━━━━━━━━━━━━━━━━━━",
        "",
        f"📊 <b>Signal Snapshot ({total} stocks)</b>:",
        f"  🟢 STRONG BUY:  {signal_counts.get('STRONG BUY', 0)}",
        f"  🟡 BUY:         {signal_counts.get('BUY', 0)}",
        f"  ⚪ HOLD:        {signal_counts.get('HOLD', 0)}",
        f"  🟠 SELL:        {signal_counts.get('SELL', 0)}",
        f"  🔴 STRONG SELL: {signal_counts.get('STRONG SELL', 0)}",
        "",
    ]

    if top3:
        lines.append("🏆 <b>Top BUY Candidates This Week</b>:")
        for i, (fp, sym, px, sig) in enumerate(top3, 1):
            lines.append(f"  {i}. {_signal_emoji(sig)} <b>{sym}</b> — ₹{px:,.2f}  |  Fundamentals: {fp}/10")
        lines.append("")

    if strong_signals:
        lines.append("⚡ <b>Strong Signals Alert</b>:")
        for sym, sig, px in strong_signals:
            lines.append(f"  {_signal_emoji(sig)} {sym}: {sig} @ ₹{px:,.2f}")
    else:
        lines.append("  (No STRONG BUY / STRONG SELL signals this week)")
    lines.append("")
    lines.append(f"🕐 {run_time.strftime('%d %b %Y, %H:%M IST')}")
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

    # Friday 10:00 UTC = 3:30 PM IST → weekly summary only run (no full scan)
    is_friday_summary = (run_time.weekday() == 4 and run_time.hour == 15 and run_time.minute >= 30)
    if is_friday_summary:
        log.info("Friday 3:30 PM IST — sending weekly summary")
        state = load_signals_state()
        if state:
            weekly_msg = format_weekly_summary(state, run_time)
            send_telegram_alert(bot_token, chat_id, weekly_msg)
            log.info("Weekly summary sent.")
        else:
            log.warning("Weekly summary: signals_state.json is empty — skipping")
            send_telegram_alert(bot_token, chat_id,
                "📅 <b>Weekly Summary</b>\n⚠️ No signal data found yet. Run the scanner first!")
        return

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

        log.info(f"{symbol}: {signal} | tech={tech['direction']}({tech['buy_count']}B/{tech['sell_count']}S) | fund={fund_score['passed']}/10 checklist | changed={changed}")

        # e. Log to file
        _atr_val = tech.get("atr")
        append_to_signal_log({
            "timestamp":            run_time.isoformat(),
            "symbol":               symbol,
            "signal":               signal,
            "price":                price,
            "tech_direction":       tech["direction"],
            "buy_count":            tech["buy_count"],
            "sell_count":           tech["sell_count"],
            "fund_passed":          fund_score["passed"],
            "fund_checklist":       [{"name": c["name"], "passed": c["passed"]} for c in fund_score["checklist"]],
            "data_source":          data_source,
            "indicators_triggered": tech["triggered"],
            "previous_signal":      state.get(symbol, {}).get("signal"),
            "signal_changed":       changed,
            "atr":                  _atr_val,
            "stop_loss":            round(price - 1.5 * _atr_val, 2) if _atr_val and signal in ("BUY","STRONG BUY") else
                                    round(price + 1.5 * _atr_val, 2) if _atr_val and signal in ("SELL","STRONG SELL") else None,
            "target":               round(price + 2.0 * _atr_val, 2) if _atr_val and signal in ("BUY","STRONG BUY") else
                                    round(price - 2.0 * _atr_val, 2) if _atr_val and signal in ("SELL","STRONG SELL") else None,
            "outcome_price":        None,
            "outcome_date":         None,
            "outcome_pct":          None,
        })

        # f. Update state
        state[symbol] = {
            "signal":      signal,
            "timestamp":   run_time.isoformat(),
            "price":       price,
            "atr":         tech.get("atr"),
            "fund_passed": fund_score["passed"],
        }

        # g. Send alert only if signal changed
        if changed:
            prev_signal = state.get(symbol, {}).get("signal")
            msg = format_alert_message(
                symbol, price, signal, tech, fund_score,
                data_source, run_time, prev_signal,
                fundamentals=fundamentals,
                sector=stock_meta.get("sector"),
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
        f"📋 Fundamentals scored via 10-point checklist\n"
        f"   (≥8 = STRONG · ≥6 = OK · &lt;6 = WEAK)\n\n"
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
