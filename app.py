"""
מנתח מניות — שיטת 5 שלבים  (v2)
- Stock data caching (5 min TTL)
- Self keep-alive ping every 10 min (Render free tier)
- /health endpoint for UptimeRobot monitoring
"""

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import requests as http_requests
import urllib.parse
import urllib.request
import threading
import time
import os


app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ─────────────────────────────────────────────
#  Simple in-memory cache (TTL = 5 minutes)
# ─────────────────────────────────────────────
_cache: dict = {}
_cache_lock = threading.Lock()
CACHE_TTL = 300  # seconds


def cache_get(key: str):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]
    return None


def cache_set(key: str, data):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}


# ─────────────────────────────────────────────
#  Keep-alive (prevents Render free tier sleep)
# ─────────────────────────────────────────────
def _ping_self():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if url:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=10)
        except Exception:
            pass
    threading.Timer(600, _ping_self).start()   # every 10 minutes


if os.environ.get("RENDER"):
    threading.Timer(60, _ping_self).start()    # first ping after 1 minute


# ─────────────────────────────────────────────
#  Translation helper (MyMemory free API)
# ─────────────────────────────────────────────

def translate_to_hebrew(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        chunks = []
        words = text.split()
        current = []
        for word in words:
            current.append(word)
            if len(' '.join(current)) > 480:
                chunks.append(' '.join(current[:-1]))
                current = [word]
        if current:
            chunks.append(' '.join(current))

        translated_parts = []
        for chunk in chunks:
            encoded = urllib.parse.quote(chunk)
            url = f"https://api.mymemory.translated.net/get?q={encoded}&langpair=en|he"
            resp = http_requests.get(url, timeout=8)
            if resp.ok:
                data = resp.json()
                translated_parts.append(data["responseData"]["translatedText"])
            else:
                translated_parts.append(chunk)

        return ' '.join(translated_parts)
    except Exception:
        return text


# ─────────────────────────────────────────────
#  Israeli ticker mappings
# ─────────────────────────────────────────────
ISRAELI_MAP = {
    "ARYT": "ARYT.TA",
    "INRM": "INRM.TA",
    "SPEN": "SPEN.TA",
    "TSEM": "TSEM.TA",
    "NICE": "NICE.TA",
    "TEVA": "TEVA.TA",
    "CHKP": "CHKP.TA",
}


# ─────────────────────────────────────────────
#  Data fetching with cache + retries
# ─────────────────────────────────────────────

def fetch_history(ticker_symbol: str, period: str = "1y", retries: int = 3) -> pd.DataFrame:
    cache_key = f"hist:{ticker_symbol}:{period}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker_symbol).history(period=period)
            if not hist.empty:
                cache_set(cache_key, hist)
                return hist
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
    return pd.DataFrame()


# ─────────────────────────────────────────────
#  Technical Indicators
# ─────────────────────────────────────────────

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_supports(hist: pd.DataFrame, n: int = 3, lookback: int = 90) -> list:
    data = hist.tail(lookback)
    lows = data["Low"].values
    supports = []
    for i in range(n, len(lows) - n):
        if all(lows[i] <= lows[i - j] for j in range(1, n + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, n + 1)):
            supports.append(float(lows[i]))
    return sorted(set(round(s, 2) for s in supports))


# ─────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────

def score_eq(price: float, sma150: float, supports: list) -> int:
    if price < sma150 * 0.96:
        return 2
    valids = [s for s in supports if s < price]
    if not valids:
        d = (price - sma150) / sma150
        if   d < 0.03: return 9
        elif d < 0.07: return 7
        elif d < 0.15: return 5
        elif d < 0.25: return 3
        return 2
    nearest = max(valids)
    d = (price - nearest) / nearest
    if   d < 0.02: return 10
    elif d < 0.05: return 9
    elif d < 0.09: return 7
    elif d < 0.14: return 6
    elif d < 0.22: return 4
    return 3


def score_ts(rsi: float, price: float, sma50: float, sma150: float,
             vol: float, avg_vol: float, hist: pd.DataFrame) -> int:
    pts = 0
    if   50 <= rsi <= 65: pts += 3
    elif 45 <= rsi <= 70: pts += 2
    elif 35 <= rsi <= 75: pts += 1
    if   price > sma150:        pts += 2
    elif price > sma150 * 0.97: pts += 1
    if price > sma50:           pts += 1
    vr = vol / avg_vol if avg_vol > 0 else 1.0
    if   vr > 1.5: pts += 2
    elif vr > 1.0: pts += 1
    if len(hist) >= 6 and price > float(hist["Close"].iloc[-6]):
        pts += 1
    return max(1, min(10, round(pts / 9 * 10)))


def score_analyst(stock) -> int:
    try:
        info = stock.info
        mean = info.get("recommendationMean")
        if mean and isinstance(mean, (int, float)):
            return max(1, min(10, round(12 - mean * 2)))
        recs = stock.recommendations
        if recs is not None and not recs.empty:
            grade_map = {
                "strong buy": 10, "buy": 8, "overweight": 8, "outperform": 7,
                "neutral": 5, "hold": 5, "market perform": 5,
                "underweight": 3, "underperform": 3,
                "sell": 2, "strong sell": 1,
            }
            scores = []
            for _, row in recs.tail(12).iterrows():
                g = str(row.get("To Grade", "")).lower()
                for k, v in grade_map.items():
                    if k in g:
                        scores.append(v)
                        break
            if scores:
                return max(1, min(10, round(sum(scores) / len(scores))))
    except Exception:
        pass
    return 5


def get_macro_env() -> dict:
    cached = cache_get("macro")
    if cached:
        return cached
    try:
        vix  = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
        oil  = float(yf.Ticker("CL=F").history(period="5d")["Close"].iloc[-1])
        spy  = yf.Ticker("^GSPC").history(period="35d")["Close"]
        sp_chg = float((spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0] * 100)
        result = {
            "vix": round(vix, 1), "oil": round(oil, 1),
            "sp500_30d": round(sp_chg, 1),
            "fear": vix > 25, "high_oil": oil > 85, "market_down": sp_chg < -3,
        }
        cache_set("macro", result)
        return result
    except Exception:
        return {"vix": 18, "oil": 75, "sp500_30d": 0,
                "fear": False, "high_oil": False, "market_down": False}


def macro_score_for_sector(sector: str, ticker: str, env: dict) -> int:
    sector = (sector or "").lower()
    score = 0
    defensive = ["defense", "aerospace", "gold", "silver", "precious", "energy", "oil", "uranium"]
    risky     = ["technology", "crypto", "bitcoin", "solar", "communication", "growth", "software"]
    if env["fear"] or env["market_down"]:
        if any(d in sector for d in defensive): score += 1
        if any(r in sector for r in risky):     score -= 1
    if env["high_oil"]:
        if "energy" in sector or "oil" in sector: score += 1
    return max(-2, min(2, score))


def build_grade(eq, ts, als, macro_norm, mode):
    if mode == "macro":
        raw = (eq * 3 + ts * 2.5 + als * 2.5 + macro_norm * 2) / 10
    else:
        raw = (eq * 4 + ts * 3 + als * 3) / 10
    if   raw >= 7.5: grade = "A"
    elif raw >= 6.0: grade = "B"
    elif raw >= 4.5: grade = "C"
    else:            grade = "D"
    return grade, round(raw, 2)


def build_pros_cons(price, sma150, sma50, rsi, atr, eq, ts, als,
                    info, vol_ratio, macro_raw, mode, macro_env):
    pros, cons = [], []
    if price > sma150:
        pct = (price - sma150) / sma150 * 100
        pros.append(f"מגמה ארוכת טווח עולה — מחיר {pct:.1f}% מעל ממוצע 150")
    else:
        pct = (sma150 - price) / sma150 * 100
        cons.append(f"מגמה ארוכת טווח יורדת — מחיר {pct:.1f}% מתחת לממוצע 150")
    if price > sma50:
        pros.append("מגמה קצרת טווח חיובית — מעל ממוצע 50")
    else:
        cons.append("מגמה קצרת טווח שלילית — מתחת לממוצע 50")
    if 50 <= rsi <= 65:
        pros.append(f"RSI {rsi:.0f} — מומנטום חיובי ומאוזן (אזור כניסה אידיאלי)")
    elif rsi > 70:
        cons.append(f"RSI {rsi:.0f} — קנייה יתר, סיכון לתיקון")
    elif rsi < 35:
        cons.append(f"RSI {rsi:.0f} — מכירת יתר / חולשה טכנית קיצונית")
    elif rsi < 45:
        cons.append(f"RSI {rsi:.0f} — מומנטום חלש")
    else:
        pros.append(f"RSI {rsi:.0f} — ניטרלי-חיובי")
    if vol_ratio > 1.5:
        pros.append(f"נפח גבוה פי {vol_ratio:.1f} מהממוצע — אישור מהלך")
    elif vol_ratio < 0.7:
        cons.append(f"נפח נמוך ({vol_ratio:.1f}x ממוצע) — מהלך לא מאושר בנפח")
    atr_pct = atr / price * 100
    if atr_pct > 6:
        cons.append(f"תנודתיות גבוהה (ATR {atr_pct:.1f}%) — דורש סטופ רחב יחסית")
    else:
        pros.append(f"תנודתיות סבירה (ATR {atr_pct:.1f}%) — ניהול סיכון נוח")
    if als >= 8:
        pros.append("קונצנזוס אנליסטים: Buy / Strong Buy")
    elif als >= 6:
        pros.append("קונצנזוס אנליסטים: רוב ממליצים לקנות")
    elif als <= 4:
        cons.append("קונצנזוס אנליסטים חלש — Hold / Underperform")
    if eq >= 8:
        pros.append("מחיר בדיוק באזור כניסה לפי שיטת 5 השלבים")
    elif eq <= 4:
        cons.append("מחיר רחוק מאזור הכניסה האידיאלי — כדאי לחכות לפולבק")
    pe = info.get("trailingPE") or info.get("forwardPE")
    if isinstance(pe, (int, float)) and pe > 0:
        if pe < 20:
            pros.append(f"P/E {pe:.1f} — הערכה נמוכה יחסית לשוק")
        elif pe > 60:
            cons.append(f"P/E {pe:.1f} — הערכת יתר גבוהה")
    if mode == "macro" and macro_env:
        vix_val = macro_env.get("vix", 0)
        oil_val = macro_env.get("oil", 0)
        if macro_raw >= 2:
            pros.append("סביבת מקרו מעולה — המגזר מרוויח ישירות מהתנאים הנוכחיים")
        elif macro_raw >= 1:
            pros.append("סביבת מקרו תומכת — המגזר מוגן יחסית")
        elif macro_raw <= -2:
            cons.append("סביבת מקרו שלילית מאוד — הימנע ממגזר זה כעת")
        elif macro_raw <= -1:
            cons.append("סביבת מקרו שלילית — לחץ על המגזר")
        if macro_env.get("fear"):
            cons.append(f"VIX {vix_val} — פחד גבוה בשוק, תנודתיות מוגברת")
        if macro_env.get("high_oil") and oil_val > 0:
            if "energy" in (info.get("sector") or "").lower():
                pros.append(f"נפט ב-${oil_val:.0f}/חבית — תומך ישירות במגזר")
            else:
                cons.append(f"נפט ב-${oil_val:.0f}/חבית — לחץ אינפלציוני על השוק")
    return pros[:6], cons[:6]


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculator")
def calculator():
    return render_template("calculator.html")


@app.route("/api/stock/<ticker>")
def stock_api(ticker):
    ticker = ticker.strip().upper()
    if ticker in ISRAELI_MAP:
        yticker, is_il = ISRAELI_MAP[ticker], True
    elif ticker.endswith(".TA"):
        yticker, is_il = ticker, True
    else:
        yticker, is_il = ticker, False

    try:
        hist = fetch_history(yticker, period="30d")
        if hist.empty or len(hist) < 2:
            return jsonify({"error": f"לא נמצאו נתונים עבור {ticker} — נסה שוב"}), 404

        price = float(hist["Close"].iloc[-1])
        atr14 = float(calc_atr(hist["High"], hist["Low"], hist["Close"]).iloc[-1])

        if is_il:
            currency, usdils = "ILS", None
        else:
            currency = "USD"
            try:
                fx_hist = fetch_history("ILS=X", period="5d")
                usdils  = float(fx_hist["Close"].iloc[-1]) if not fx_hist.empty else 3.7
            except Exception:
                usdils = 3.7

        return jsonify({
            "price":    round(price, 2),
            "atr14":    round(atr14, 2) if not np.isnan(atr14) else None,
            "currency": currency,
            "usdils":   round(usdils, 3) if usdils else None,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    body          = request.get_json(force=True)
    raw_ticker    = body.get("ticker", "").strip().upper()
    mode          = body.get("mode", "regular")
    portfolio_nis = float(body.get("portfolio_size", 120_000))

    if not raw_ticker:
        return jsonify({"error": "נא הזן סימול מניה"}), 400

    if raw_ticker in ISRAELI_MAP:
        ticker, is_il = ISRAELI_MAP[raw_ticker], True
    elif raw_ticker.endswith(".TA"):
        ticker, is_il = raw_ticker, True
    else:
        ticker, is_il = raw_ticker, False

    try:
        hist = fetch_history(ticker, period="1y")
        stk  = yf.Ticker(ticker)

        if hist.empty or len(hist) < 20:
            return jsonify({"error": f"לא נמצאו נתונים עבור {raw_ticker}. בדוק שהסימול נכון."}), 404

        info     = stk.info
        close    = hist["Close"].dropna()
        highs    = hist["High"].dropna()
        lows     = hist["Low"].dropna()
        vols     = hist["Volume"].dropna()

        rsi_s    = calc_rsi(close)
        atr_s    = calc_atr(highs, lows, close)
        sma50_s  = close.rolling(50).mean()
        sma150_s = close.rolling(150).mean()

        price    = float(close.iloc[-1])
        rsi      = float(rsi_s.iloc[-1])   if not np.isnan(rsi_s.iloc[-1])   else 50.0
        atr      = float(atr_s.iloc[-1])   if not np.isnan(atr_s.iloc[-1])   else price * 0.03
        sma50    = float(sma50_s.iloc[-1]) if len(close) >= 50  and not np.isnan(sma50_s.iloc[-1])  else price
        sma150   = float(sma150_s.iloc[-1]) if len(close) >= 150 and not np.isnan(sma150_s.iloc[-1]) else price * 0.9
        vol      = float(vols.iloc[-1])
        avg_vol  = float(vols.iloc[-20:].mean())
        vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0

        supports = find_supports(hist)
        eq  = score_eq(price, sma150, supports)
        ts  = score_ts(rsi, price, sma50, sma150, vol, avg_vol, hist)
        als = score_analyst(stk)

        macro_env = None
        macro_raw = 0
        if mode == "macro":
            macro_env = get_macro_env()
            macro_raw = macro_score_for_sector(
                info.get("sector") or info.get("industry") or "", ticker, macro_env
            )

        macro_norm = 6 + macro_raw * 2
        grade, score_val = build_grade(eq, ts, als, macro_norm, mode)

        below = [s for s in supports if s < price]
        if below:
            nearest_sup = max(below)
            ez_lo = round(nearest_sup * 0.99, 2)
            ez_hi = round(nearest_sup * 1.04, 2)
        else:
            ez_lo = round(sma150 * 0.98, 2)
            ez_hi = round(sma150 * 1.03, 2)

        stop_loss = round(max(ez_lo * 0.95, price - 2.0 * atr), 2)
        target1   = round(price + 3 * atr, 2)
        target2   = round(price + 5 * atr, 2)
        risk_dist = price - stop_loss
        rr_ratio  = round(3 * atr / max(risk_dist, 0.01), 2)

        if is_il:
            currency, fx = "₪", 1.0
        else:
            currency = "$"
            try:
                fx_hist = fetch_history("ILS=X", period="5d")
                fx = float(fx_hist["Close"].iloc[-1]) if not fx_hist.empty else 3.7
            except Exception:
                fx = 3.7

        stock_budget     = portfolio_nis * 0.40
        risk_per_trade   = portfolio_nis * 0.01
        risk_in_currency = risk_per_trade if is_il else risk_per_trade / fx
        num_shares       = int(risk_in_currency / (1.5 * atr)) if atr > 0 else 0
        pos_val_currency = round(num_shares * price, 0)
        pos_val_nis      = round(pos_val_currency * fx if not is_il else pos_val_currency, 0)
        pct_budget       = round(pos_val_nis / stock_budget * 100, 1) if stock_budget > 0 else 0

        pros, cons = build_pros_cons(
            price, sma150, sma50, rsi, atr, eq, ts, als,
            info, vol_ratio, macro_raw, mode, macro_env
        )

        company  = info.get("longName") or info.get("shortName") or raw_ticker
        sector   = info.get("sector")  or info.get("industry")   or "לא ידוע"
        raw_desc = (info.get("longBusinessSummary") or "")[:600]
        desc     = translate_to_hebrew(raw_desc)

        mc = info.get("marketCap") or 0
        if   mc >= 1e12: mc_str = f"${mc/1e12:.2f}T"
        elif mc >= 1e9:  mc_str = f"${mc/1e9:.1f}B"
        elif mc >= 1e6:  mc_str = f"${mc/1e6:.0f}M"
        elif mc > 0:     mc_str = f"${mc:,.0f}"
        else:            mc_str = "N/A"

        pe = info.get("trailingPE") or info.get("forwardPE") or None
        pe_str = f"{pe:.1f}" if isinstance(pe, (int, float)) and pe > 0 else "N/A"

        return jsonify({
            "success": True, "ticker": raw_ticker, "company": company,
            "sector": sector, "description": desc, "currency": currency,
            "is_il": is_il, "fx": round(fx, 3), "price": round(price, 2),
            "week52_hi": round(info.get("fiftyTwoWeekHigh") or price, 2),
            "week52_lo": round(info.get("fiftyTwoWeekLow")  or price, 2),
            "market_cap": mc_str, "pe": pe_str,
            "grade": grade, "score": score_val, "mode": mode,
            "eq": eq, "ts": ts, "als": als, "macro_raw": macro_raw,
            "rsi": round(rsi, 1), "atr": round(atr, 2),
            "atr_pct": round(atr / price * 100, 2),
            "sma50": round(sma50, 2), "sma150": round(sma150, 2),
            "above150": price > sma150, "above50": price > sma50,
            "vol_ratio": round(vol_ratio, 2),
            "ez_lo": ez_lo, "ez_hi": ez_hi, "stop_loss": stop_loss,
            "target1": target1, "target2": target2, "rr_ratio": rr_ratio,
            "portfolio_nis": portfolio_nis, "stock_budget": stock_budget,
            "risk_nis": round(risk_per_trade, 0), "num_shares": num_shares,
            "pos_val": pos_val_currency, "pos_val_nis": pos_val_nis,
            "pct_budget": pct_budget, "pros": pros, "cons": cons,
            "macro_env": macro_env,
        })

    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "detail": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, use_reloader=False)
