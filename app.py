import time
import streamlit as st

st.set_page_config(page_title="PA Scanner", page_icon="📈", layout="centered")

# ---------------------------------------------------------------- STATE
params = st.query_params
page = params.get("page", "dashboard")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = [
        {"sym": "RELIANCE", "cat": "Stocks", "price": 1167.70, "chg": -1.63},
        {"sym": "TCS", "cat": "Stocks", "price": 2075.00, "chg": 1.19},
        {"sym": "INFY", "cat": "Stocks", "price": 1035.00, "chg": 4.11},
        {"sym": "HDFCBANK", "cat": "Stocks", "price": 721.20, "chg": 1.76},
    ]
if "results" not in st.session_state:
    st.session_state.results = []

# ---------------------------------------------------------------- CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #3b1d8f 0%, #0f1a4d 35%, #0a3a4a 70%, #0b4a2a 100%);
    color: #e8ecff;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Leave room so content is never covered by the floating nav */
.block-container { padding-top: 1.5rem; padding-bottom: 200px !important; max-width: 640px; }

/* Title */
.app-title {
    text-align: center; font-size: 2.6rem; font-weight: 800; font-style: italic;
    background: linear-gradient(90deg, #22d3a6, #a3f04a);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 .5rem 0;
}

/* Cards */
.card-label { letter-spacing: .25em; font-size: .8rem; color: #9fb0e8; font-weight: 600; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(30, 50, 110, .45);
    border: 1px solid rgba(120, 150, 255, .25) !important;
    border-radius: 28px !important;
    padding: .4rem .6rem;
}

/* Big highlighted labels (TIMEFRAME / AUTO REFRESH) */
.big-label {
    display: inline-block;
    font-size: 1.35rem; font-weight: 800; letter-spacing: .12em;
    color: #0b1437;
    background: linear-gradient(90deg, #22d3a6, #a3f04a);
    padding: 10px 16px; border-radius: 14px;
    box-shadow: 0 0 16px rgba(34, 211, 166, .35);
    white-space: nowrap;
}

/* Keep label and dropdown side by side on phones (Streamlit stacks columns by default) */
div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; align-items: center; gap: .6rem; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { min-width: 0 !important; }

/* Bigger dropdown text with colour accent */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(10, 20, 60, .85);
    border: 2px solid #22d3a6; border-radius: 14px; min-height: 52px;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    font-size: 1.3rem !important; font-weight: 700 !important; color: #a3f04a !important;
}

/* Slider */
div[data-testid="stSlider"] [role="slider"] { background: #ff4b4b; }

/* Scan button */
div.stButton > button {
    width: 100%; height: 72px; border: 1px solid #5cff7a; border-radius: 30px;
    background: linear-gradient(90deg, #22c55e, #a3f04a);
    color: #fff; font-size: 1.9rem; font-weight: 800; letter-spacing: .02em;
    box-shadow: 0 0 24px rgba(80, 255, 120, .35);
}
div.stButton > button:hover { filter: brightness(1.08); color: #fff; border-color: #fff; }

/* Watchlist tiles */
.wl-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.wl-tile {
    background: rgba(255, 255, 255, .05); border: 1px solid rgba(140, 170, 255, .25);
    border-radius: 20px; padding: 12px 10px; min-width: 0;
}
.wl-sym { font-weight: 700; font-size: .85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wl-cat { color: #9fb0e8; font-size: .75rem; margin: 6px 0 4px; }
.wl-price { font-weight: 700; font-size: 1rem; }
.up { color: #4ade80; font-size: .85rem; font-weight: 500; }
.down { color: #f87171; font-size: .85rem; font-weight: 500; }

/* ------------------------------------------------------------------
   BOTTOM NAV  (moved UP so it no longer hides behind "Manage app")
   Change the 76px value to move it higher / lower.
------------------------------------------------------------------- */
.bottom-nav {
    position: fixed;
    left: 50%;
    transform: translateX(-50%);
    bottom: 76px;                 /* <-- was 0 / ~20px. Raised above Manage app */
    width: calc(100% - 32px);
    max-width: 620px;
    height: 74px;
    display: flex; align-items: center; justify-content: space-around;
    background: rgba(20, 35, 90, .92);
    border: 1px solid rgba(120, 150, 255, .3);
    border-radius: 34px;
    backdrop-filter: blur(10px);
    z-index: 9999;
}
.bottom-nav a {
    display: flex; flex-direction: column; align-items: center; gap: 2px;
    color: #9fb0e8 !important; text-decoration: none !important;
    font-size: .85rem; font-weight: 600; flex: 1;
}
.bottom-nav a.active { color: #22d3a6 !important; }
.bottom-nav a .ico { font-size: 1.1rem; }
.bottom-nav .center {
    flex: 0 0 90px; height: 90px; margin-top: -42px;
    border-radius: 50%; border: 3px solid #4ade9a;
    background: radial-gradient(circle, #1f8f5f 0%, #0b3d2a 100%);
    justify-content: center; font-size: 1.8rem; color: #fff !important;
    box-shadow: 0 0 22px rgba(74, 222, 154, .45);
}
</style>
""",
    unsafe_allow_html=True,
)

# Shared by Timeframe and Auto Refresh
TIME_OPTIONS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
TIME_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}

# ---------------------------------------------------------------- SCAN LOGIC
def run_scan(market: str, timeframe: str, min_score: int):
    """Replace this with your real price-action scanning logic.
    Must return a list of dicts: {"symbol", "score", "signal"}"""
    demo = [
        {"symbol": "RELIANCE", "score": 72, "signal": "Bullish engulfing"},
        {"symbol": "TCS", "score": 65, "signal": "Breakout retest"},
        {"symbol": "INFY", "score": 81, "signal": "Inside bar breakout"},
        {"symbol": "HDFCBANK", "score": 58, "signal": "Range support"},
    ]
    return [r for r in demo if r["score"] >= min_score]


# ---------------------------------------------------------------- PAGES
st.markdown('<div class="app-title">PA Scanner</div>', unsafe_allow_html=True)

if page in ("dashboard", "scan"):
    market = st.radio("Market", ["Stocks", "Crypto", "Forex"], horizontal=True,
                      label_visibility="collapsed")

    with st.container(border=True):
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown('<span class="big-label">⏱ TIMEFRAME</span>', unsafe_allow_html=True)
        with c2:
            timeframe = st.selectbox("Timeframe", TIME_OPTIONS, index=1,
                                     label_visibility="collapsed")

        st.markdown('<div class="card-label">MIN SCORE ⓘ</div>', unsafe_allow_html=True)
        min_score = st.slider("Min score", 0, 100, 60, label_visibility="collapsed")

        r1, r2 = st.columns([1.3, 1])
        with r1:
            st.markdown('<span class="big-label">⟳ AUTO REFRESH</span>', unsafe_allow_html=True)
        with r2:
            refresh = st.selectbox("Auto refresh", ["Off"] + TIME_OPTIONS, index=0,
                                   label_visibility="collapsed")

        scan_clicked = st.button("SCAN NOW 🎯", use_container_width=True)

    if scan_clicked or refresh != "Off":
        with st.spinner("Scanning..."):
            st.session_state.results = run_scan(market, timeframe, min_score)

    if st.session_state.results:
        st.markdown("#### Results")
        for r in st.session_state.results:
            st.markdown(f"**{r['symbol']}** — score {r['score']} · {r['signal']}")

    # Watchlist
    tiles = ""
    for w in st.session_state.watchlist:
        cls = "up" if w["chg"] >= 0 else "down"
        sign = "+" if w["chg"] >= 0 else ""
        tiles += (
            f'<div class="wl-tile"><div class="wl-sym">📈 {w["sym"]}</div>'
            f'<div class="wl-cat">{w["cat"]}</div>'
            f'<div class="wl-price">{w["price"]:,.2f}</div>'
            f'<div class="{cls}">{sign}{w["chg"]:.2f}%</div></div>'
        )
    with st.container(border=True):
        st.markdown(
            '<div style="display:flex;justify-content:space-between;margin-bottom:10px">'
            '<span class="card-label">☆ WATCHLIST</span><span>View all ›</span></div>'
            f'<div class="wl-grid">{tiles}</div>',
            unsafe_allow_html=True,
        )

elif page == "alerts":
    st.subheader("🔔 Alerts")
    st.info("No alerts yet.")

elif page == "settings":
    st.subheader("⚙️ Settings")
    st.toggle("Dark mode", value=True)

# Auto refresh loop
if page in ("dashboard", "scan"):
    if refresh != "Off":
        time.sleep(TIME_SECONDS[refresh])
        st.rerun()

# ---------------------------------------------------------------- BOTTOM NAV
def active(name):
    return "active" if page == name else ""

st.markdown(
    f"""
<div class="bottom-nav">
  <a href="?page=dashboard" target="_self" class="{active('dashboard')}"><span class="ico">▦</span>Dashboard</a>
  <a href="?page=scan" target="_self" class="{active('scan')}"><span class="ico">◎</span>Scan</a>
  <a href="?page=scan" target="_self" class="center">◉</a>
  <a href="?page=alerts" target="_self" class="{active('alerts')}"><span class="ico">🔔</span>Alerts</a>
  <a href="?page=settings" target="_self" class="{active('settings')}"><span class="ico">⚙</span>Settings</a>
</div>
""",
    unsafe_allow_html=True,
)
