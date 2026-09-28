# Price Action Scanner

Chart-matched setup scanner for Indian Stocks, Crypto, and Forex.

## Setup Rules (Strict)
1. Near clear Support / Resistance (within 0.30%)
2. Higher Low (bullish) or Lower High (bearish)
3. Healthy break candle that closes meaningfully beyond the level
4. Next candle is clear red/green rejection or decision candle

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
