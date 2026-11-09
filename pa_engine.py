"""
pa_engine.py  -  "A+ Retest" scanner + real backtester for PA Scanner.
Setup (from your Nifty chart):  resistance/support zone tested 2+ times
 -> strong breakout candle on HIGH volume -> pullback on LOW volume that retests the zone
 -> confirmation candle closes back beyond the zone  => signal (long). Short = exact mirror.
Works on any symbol / any timeframe. Forex has no real volume on Yahoo -> volume rules become neutral.
Put this file next to app.py.  Needs only: numpy, pandas, yfinance (already in your app).
"""
import numpy as np
import pandas as pd
from datetime import timezone, timedelta, datetime

IST = timezone(timedelta(hours=5, minutes=30))

# interval -> (yahoo interval, max days Yahoo allows, resample rule, days used for live scan)
TF_SPEC = {
    "1m": ("1m", 6, None, 4), "3m": ("1m", 6, "3min", 4), "5m": ("5m", 58, None, 10),
    "15m": ("15m", 58, None, 20), "30m": ("30m", 58, None, 40), "1h": ("1h", 700, None, 90),
    "4h": ("1h", 700, "4h", 300), "1d": ("1d", 3000, None, 500),
}
DEFAULT_PARAMS = dict(min_touches=2, min_vol_ratio=1.3, max_retest_bars=10)


# ---------------------------------------------------------------- data
def fetch(symbol, interval="15m", days=None):
    yi, maxd, rs, _ = TF_SPEC.get(interval, (interval, 58, None, 20))
    days = int(min(days or maxd, maxd))
    df = None
    try:                                     # Nifty / BankNifty -> real FUTURES data (true volume)
        import futures_data
        if futures_data.is_fut_symbol(symbol) and futures_data.configured():
            df = futures_data.fetch_futures(symbol, yi, days)
    except Exception:
        df = None
    if df is None:
        import yfinance as yf
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = yf.Ticker(symbol).history(start=start, interval=yi)
        if df is None or len(df) == 0:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(IST)
    if rs:
        df = df.resample(rs).agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"}).dropna(subset=["Open"])
    return df if len(df) >= 120 else None


# ---------------------------------------------------------------- helpers
def _atr(H, L, C, n=14):
    pc = np.r_[C[0], C[:-1]]
    tr = np.maximum(H - L, np.maximum(np.abs(H - pc), np.abs(L - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().values


def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def _swing_highs(H, lo, hi, k=2):
    out = []
    for j in range(max(lo, k), hi - k):
        if H[j] >= H[j - k:j + k + 1].max():
            out.append(j)
    return out


def volume_profile(H, L, C, V, bins=24):
    """Returns POC / VAH / VAL (70% value area) of the given candles."""
    lo, hi = float(L.min()), float(H.max())
    if hi <= lo:
        return None
    w = V if V.sum() > 0 else np.ones(len(C))
    edges = np.linspace(lo, hi, bins + 1)
    tp = (H + L + C) / 3.0
    idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    prof = np.bincount(idx, weights=w, minlength=bins)
    poc = int(prof.argmax())
    total, acc, l, h = prof.sum(), prof[poc], poc, poc
    while acc < 0.7 * total and (l > 0 or h < bins - 1):
        up = prof[h + 1] if h < bins - 1 else -1
        dn = prof[l - 1] if l > 0 else -1
        if up >= dn:
            h += 1; acc += prof[h]
        else:
            l -= 1; acc += prof[l]
    mid = (edges[:-1] + edges[1:]) / 2
    return {"poc": float(mid[poc]), "vah": float(edges[h + 1]), "val": float(edges[l])}


def _is_india(symbol):
    s = symbol.upper()
    return s.endswith(".NS") or s.endswith(".BO") or s.startswith("^NSE") or s.startswith("^BSE")


def _in_session(ts, symbol, interval):
    """India intraday: ignore first 15 min and last 15 min (noise)."""
    if interval == "1d" or not _is_india(symbol):
        return True
    m = ts.hour * 60 + ts.minute
    return 9 * 60 + 30 <= m <= 15 * 60 + 15


# ---------------------------------------------------------------- core detector (long orientation)
def _detect_long(O, H, L, C, V, atr, e20, e50, i, p):
    if i < 70:
        return None
    rng_i = H[i] - L[i]
    if rng_i <= 0 or C[i] <= O[i]:
        return None
    if not (C[i] > H[i - 1] or (min(O[i], C[i]) - L[i]) / rng_i >= 0.35):
        return None                                   # no confirmation candle
    has_vol = p["has_vol"]
    best = None
    lo_b = max(i - p["max_retest_bars"], 65)
    for b in range(i - 1, lo_b - 1, -1):
        ab, rb = atr[b], H[b] - L[b]
        if rb <= 0 or C[b] <= O[b]:
            continue
        body = (C[b] - O[b]) / rb
        if body < 0.5 or (C[b] - L[b]) / rb < 0.65:
            continue                                  # breakout candle must be strong
        vr = 1.0
        if has_vol:
            base = V[b - 20:b].mean()
            vr = V[b] / base if base > 0 else 1.0
            if vr < p["min_vol_ratio"]:
                continue                              # breakout needs volume
        sw = _swing_highs(H, b - 60, b, 2)
        if len(sw) < 2:
            continue
        hs = np.array([H[j] for j in sw])
        zone = None
        for lv in hs:
            mem = np.where(np.abs(hs - lv) <= 0.4 * ab)[0]
            t = len(mem)
            if t < p["min_touches"] or b - sw[mem[0]] < 4:
                continue
            top = hs[mem].max()
            if not (C[b] > top + 0.1 * ab and O[b] <= top + 0.5 * ab and C[b - 1] <= top + 0.3 * ab):
                continue
            if zone is None or t > zone[0]:
                zone = (t, float(top))
        if zone is None:
            continue
        touches, top = zone
        bot = top - 0.5 * ab
        seg_L, seg_C = L[b + 1:i + 1], C[b + 1:i + 1]
        if len(seg_L) == 0 or C[i] <= top:
            continue
        if seg_C.min() < bot - 0.1 * ab:              # breakout failed (closed back inside)
            continue
        if seg_L.min() > top + 0.1 * ab:              # never retested the zone
            continue
        r = b + 1 + int(np.argmin(seg_L))
        if i - r > 2:                                 # retest too old -> signal is late
            continue
        pb = V[b + 1:i] if has_vol else np.array([])
        dry = None
        if has_vol and len(pb) > 0:
            dry = bool(pb.mean() < 0.8 * V[b])
        entry = float(C[i])
        stop = float(min(seg_L.min(), bot) - 0.15 * ab)
        risk = entry - stop
        if risk < 0.3 * atr[i] or risk > 3.0 * atr[i]:
            continue
        W = min(120, b)
        vp = volume_profile(H[b - W:b], L[b - W:b], C[b - W:b], V[b - W:b], 24)
        conf = bool(vp and any(abs(top - vp[k]) <= 0.5 * ab for k in ("poc", "vah", "val")))
        sw2 = _swing_highs(H, max(5, b - 120), b, 2)
        above = [H[j] for j in sw2 if H[j] > entry]
        room = (min(above) - entry) / risk if above else 9.9
        wick = (min(O[r], C[r]) - L[r]) / max(H[r] - L[r], 1e-12)
        trend = bool(e20[i] > e50[i] and C[i] > e50[i])

        sc = 20 * min(touches, 4) / 4
        sc += 10 * min(1, (body - 0.5) / 0.4) + 10 * min(1, (C[b] - top) / (0.8 * ab))
        sc += (20 * min(1, max(0, (vr - 1.0) / 1.5))) if has_vol else 10
        sc += 15 if dry else (7 if dry is None else 0)
        sc += 10 + (5 if wick >= 0.4 else 0)
        sc += 10 if trend else 0
        sc += 10 if conf else 0
        sc -= 15 if room < 1.0 else (10 if room < 1.5 else 0)
        sc = int(max(0, min(100, round(sc))))
        cand = dict(score=sc, entry=entry, stop=stop, zone_top=top, zone_bot=float(bot), touches=touches,
                    vol_ratio=round(float(vr), 2), dry=dry, trend_ok=trend, profile_conf=conf,
                    room_r=round(float(room), 1), b=b, atr=float(atr[i]))
        if best is None or sc > best["score"]:
            best = cand
    return best


def _prep(df):
    O, H, L, C = (df[c].values.astype(float) for c in ("Open", "High", "Low", "Close"))
    V = df["Volume"].values.astype(float)
    has_vol = bool(V.sum() > 0 and (V > 0).mean() >= 0.8)
    atr = _atr(H, L, C)
    e20, e50 = _ema(C, 20), _ema(C, 50)
    return dict(O=O, H=H, L=L, C=C, V=V, atr=atr, e20=e20, e50=e50, has_vol=has_vol)


def _detect_at(A, i, p):
    p = dict(p, has_vol=A["has_vol"])
    out = []
    r = _detect_long(A["O"], A["H"], A["L"], A["C"], A["V"], A["atr"], A["e20"], A["e50"], i, p)
    if r:
        r.update(direction="Bullish", level=r["zone_top"]); out.append(r)
    r = _detect_long(-A["O"], -A["L"], -A["H"], -A["C"], A["V"], A["atr"], -A["e20"], -A["e50"], i, p)
    if r:                                             # mirror image -> real prices
        r.update(direction="Bearish", entry=-r["entry"], stop=-r["stop"], level=-r["zone_top"],
                 zone_top=-r["zone_top"], zone_bot=-r["zone_bot"])
        out.append(r)
    return max(out, key=lambda x: x["score"]) if out else None


# ---------------------------------------------------------------- live scan (drop-in for core_scan)
def _ts(ts):
    return ts.astimezone(IST).strftime("%d-%b-%Y %H:%M IST")


def scan_aplus(symbol, interval="15m", min_score=75, rr=2.0, df=None):
    """Returns the app's normal result dict (works with your cards/charts/Telegram) or None."""
    try:
        if df is None:
            df = fetch(symbol, interval, TF_SPEC.get(interval, (0, 0, 0, 20))[3])
        if df is None:
            return None
        df = df.iloc[:-1]                             # drop the still-forming candle (no repainting)
        A = _prep(df)
        n = len(df)
        for back in (0, 1):                           # last closed candle, or one before
            i = n - 1 - back
            if not _in_session(df.index[i], symbol, interval):
                continue
            s = _detect_at(A, i, DEFAULT_PARAMS)
            if s and s["score"] >= min_score:
                d = 1 if s["direction"] == "Bullish" else -1
                risk = (s["entry"] - s["stop"]) * d
                tgt = s["entry"] + d * rr * risk
                grade = "A+" if s["score"] >= 85 else "A"
                vol_txt = f"vol x{s['vol_ratio']}" if A["has_vol"] else "no volume data"
                return {
                    "symbol": symbol, "score": s["score"], "full_match": s["score"] >= 85,
                    "trend": f"{grade} | SL {s['stop']:.5g} | TP {tgt:.5g} ({rr}R) | {vol_txt}",
                    "direction": s["direction"], "price": round(s["entry"], 5), "interval": interval,
                    "near_level": "Retest of " + ("Resistance" if d == 1 else "Support"),
                    "level_price": round(s["level"], 5), "break_level": round(s["level"], 5),
                    "stop": round(s["stop"], 5), "target": round(tgt, 5), "rr": rr,
                    "break_time_ist": _ts(df.index[s["b"]]), "setup_time_ist": _ts(df.index[i]),
                    "scanned_at_ist": datetime.now(IST).strftime("%d-%b-%Y %H:%M IST"),
                    "scores": {"touches": s["touches"], "vol_ratio": s["vol_ratio"]},
                    "details": {k: s[k] for k in ("touches", "vol_ratio", "dry", "trend_ok", "profile_conf", "room_r")},
                    "trend_points": [], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "df": df.tail(80),
                }
        return None
    except Exception:
        return None


# ---------------------------------------------------------------- backtest
def collect_signals(df, symbol, interval, p=None):
    p = p or DEFAULT_PARAMS
    A = _prep(df)
    sigs = []
    for i in range(70, len(df) - 1):
        if not _in_session(df.index[i], symbol, interval):
            continue
        s = _detect_at(A, i, p)
        if s and s["score"] >= 60:
            s.update(i=i, time=df.index[i], symbol=symbol)
            sigs.append(s)
    return sigs


def simulate(df, sigs, min_score=80, rr=2.0, fee_pct=0.05, max_hold=60):
    """Entry = NEXT candle open (no look-ahead). If stop and target are in the same candle -> stop (pessimistic)."""
    O, H, L, C = (df[c].values.astype(float) for c in ("Open", "High", "Low", "Close"))
    n, busy, rows = len(df), -1, []
    for s in sigs:
        if s["score"] < min_score or s["i"] <= busy or s["i"] + 1 >= n:
            continue
        j0 = s["i"] + 1
        d = 1 if s["direction"] == "Bullish" else -1
        entry, stop = O[j0], s["stop"]
        risk = (entry - stop) * d
        if risk <= 0 or risk > 4 * s["atr"]:
            continue
        target = entry + d * rr * risk
        R, ex = None, None
        for j in range(j0, min(n, j0 + max_hold)):
            if d == 1:
                if L[j] <= stop: R, ex = -1.0, j; break
                if H[j] >= target: R, ex = float(rr), j; break
            else:
                if H[j] >= stop: R, ex = -1.0, j; break
                if L[j] <= target: R, ex = float(rr), j; break
        if ex is None:
            ex = min(n - 1, j0 + max_hold - 1)
            R = (C[ex] - entry) * d / risk
        R -= entry * fee_pct / 100.0 / risk
        busy = ex
        rows.append(dict(symbol=s["symbol"], time=s["time"], direction=s["direction"], score=s["score"],
                         entry=round(entry, 5), stop=round(stop, 5), target=round(target, 5),
                         R=round(R, 3), bars=ex - j0 + 1))
    return pd.DataFrame(rows)


def stats(tr):
    if tr is None or len(tr) == 0:
        return dict(trades=0, win_rate=0, avg_r=0, pf=0, net_r=0, max_dd=0)
    R = tr["R"]
    gw, gl = R[R > 0].sum(), -R[R < 0].sum()
    eq = R.cumsum()
    return dict(trades=int(len(R)), win_rate=round(float((R > 0).mean() * 100), 1), avg_r=round(float(R.mean()), 3),
                pf=round(float(gw / gl), 2) if gl > 0 else 99.0, net_r=round(float(R.sum()), 1),
                max_dd=round(float((eq.cummax() - eq).max()), 1))


def verdict(st):
    if st["trades"] < 30:
        return "⚠️ Trades kam hain (<30) - result par bharosa mat karo. Zyada symbols / lamba period lo."
    if st["avg_r"] > 0.15 and st["pf"] >= 1.3:
        return "✅ Edge positive dikh raha hai. Ab chhote paise se forward-test karo."
    if st["avg_r"] > 0:
        return "🟡 Halka positive. Min score badhao ya RR badlo."
    return "❌ Abhi edge nahi. Min score badhao / dusra timeframe try karo."


# ---------------------------------------------------------------- Streamlit page
def render_backtest_ui(st, symbols, market, normalize=None):
    import plotly.graph_objects as go
    fetch_c = st.cache_data(ttl=900, show_spinner=False)(fetch)
    tfs = list(TF_SPEC.keys())
    default_fee = {"Indian Stocks": 0.05, "Crypto": 0.10, "Forex": 0.02}.get(market, 0.05)

    with st.container():
        mode = st.radio("Test on", ["One symbol", "Whole watchlist (best for real stats)"], key="bt2_mode")
        sym = None
        if mode == "One symbol":
            opts = symbols or ["^NSEI"]
            sym = st.selectbox("Symbol", opts, key="bt2_sym")
            typed = st.text_input("or type symbol", key="bt2_typed")
            if typed.strip():
                sym = normalize(typed.strip().upper(), market) if normalize else typed.strip().upper()
        tf = st.selectbox("Timeframe", tfs, index=tfs.index("15m"), key="bt2_tf")
        maxd = TF_SPEC[tf][1]
        st.caption(f"Yahoo gives max ~{maxd} days of {tf} data. Longer = more trades = more reliable.")
        days = st.slider("Days of history", 2, maxd, min(maxd, 58 if maxd > 58 else maxd), key="bt2_days")
        c1, c2 = st.columns(2)
        min_score = c1.slider("Min score (A+ = 85)", 60, 95, 80, 5, key="bt2_ms")
        rr = c2.select_slider("Target (R)", [1.0, 1.5, 2.0, 3.0], value=2.0, key="bt2_rr")
        fee = st.number_input("Round-trip cost % (fee+slippage)", 0.0, 1.0, default_fee, 0.01, key="bt2_fee")
        limit = st.slider("Max symbols (watchlist mode)", 5, 60, 20, key="bt2_lim") if mode != "One symbol" else 1
        run = st.button("RUN BACKTEST", type="primary", use_container_width=True, key="bt2_run")

    if run:
        syms = [sym] if mode == "One symbol" else list(symbols)[:limit]
        prog = st.progress(0.0, text="Starting...")
        all_sigs, dfs = [], {}
        for k, s in enumerate(syms):
            prog.progress(k / max(len(syms), 1), text=f"Scanning history: {s} ({k+1}/{len(syms)})")
            try:
                d = fetch_c(s, tf, days)
                if d is None:
                    continue
                dfs[s] = d
                all_sigs.append((s, collect_signals(d, s, tf)))
            except Exception:
                continue
        prog.progress(1.0, text="Done")
        st.session_state["bt2_data"] = dict(dfs=dfs, sigs=all_sigs, tf=tf)

    data = st.session_state.get("bt2_data")
    if not data or not data["sigs"]:
        if run:
            st.warning("No data / no setups found. Try more days or a lower min score.")
        return

    def run_all(ms, r):
        frames = [simulate(data["dfs"][s], sg, ms, r, fee) for s, sg in data["sigs"]]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True).sort_values("time") if frames else pd.DataFrame()

    tr = run_all(min_score, rr)
    s = stats(tr)
    st.markdown("### Result")
    m1, m2 = st.columns(2)
    m1.metric("Trades", s["trades"]); m2.metric("Win rate", f"{s['win_rate']}%")
    m3, m4 = st.columns(2)
    m3.metric("Avg R / trade", s["avg_r"]); m4.metric("Profit factor", s["pf"])
    m5, m6 = st.columns(2)
    m5.metric("Net (R)", s["net_r"]); m6.metric("Max drawdown (R)", s["max_dd"])
    st.info(verdict(s))
    st.caption("1R = your risk per trade. Avg R > 0.15 and profit factor > 1.3 with 30+ trades = decent edge. "
               "Entry is next-candle open, costs included, stop assumed first if both hit in one candle.")

    if len(tr):
        fig = go.Figure(go.Scatter(y=tr["R"].cumsum(), mode="lines", line=dict(color="#00e676", width=2)))
        fig.update_layout(height=240, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="Equity (R)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        tr["bucket"] = pd.cut(tr["score"], [0, 69, 79, 89, 100], labels=["<70", "70-79", "80-89", "90+"])
        g = tr.groupby("bucket", observed=True)["R"].agg(trades="count", win_rate=lambda x: round((x > 0).mean() * 100, 1),
                                                        avg_R=lambda x: round(x.mean(), 3)).reset_index()
        st.markdown("**Kya high score sach mein better hai?**")
        st.dataframe(g, hide_index=True, use_container_width=True)
        d2 = tr.groupby("direction")["R"].agg(trades="count", avg_R=lambda x: round(x.mean(), 3)).reset_index()
        st.dataframe(d2, hide_index=True, use_container_width=True)

    with st.expander("Auto-tune: best Min score x Target combo"):
        rows = []
        for ms in (70, 75, 80, 85, 90):
            for r in (1.5, 2.0, 3.0):
                t2 = run_all(ms, r); s2 = stats(t2)
                rows.append(dict(min_score=ms, target_R=r, trades=s2["trades"], win_pct=s2["win_rate"],
                                 avg_R=s2["avg_r"], PF=s2["pf"], net_R=s2["net_r"]))
        tune = pd.DataFrame(rows)
        st.dataframe(tune[tune.trades >= 15].sort_values("avg_R", ascending=False).head(8),
                     hide_index=True, use_container_width=True)
        st.caption("Sirf 15+ trades wali combos. Best combo ko aankh band karke mat maano - "
                   "dusre symbols / dusre dino par bhi check karo (over-fitting se bacho).")

    if len(tr):
        with st.expander("All trades"):
            st.dataframe(tr.drop(columns=["bucket"], errors="ignore"), hide_index=True, use_container_width=True)
            st.download_button("Download CSV", tr.to_csv(index=False), "backtest_trades.csv", "text/csv")
