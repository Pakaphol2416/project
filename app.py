from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import ta
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import numpy as np
from textblob import TextBlob
import requests

app = Flask(__name__)
CORS(app)

NEWS_API_KEY = "dd8de0cbe64b45f48e8ce1a56bb3dbc4"

def plot_price_sma(df, symbol):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.plot(df.index, df['sma_50'], linestyle='--', label='SMA 50')
    plt.title(f"{symbol} Close Price & SMA 50")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return img_base64

def fetch_news_api(symbol):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={symbol}&"
        f"language=en&"  # ถ้าต้องการไทยเปลี่ยนเป็น 'th'
        f"sortBy=publishedAt&"
        f"pageSize=10&"
        f"apiKey={NEWS_API_KEY}"
    )
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        articles = data.get('articles', [])
        news_list = []
        for art in articles:
            title = art.get('title')
            source = art.get('source', {}).get('name', '')
            url = art.get('url')
            if title and url:
                blob = TextBlob(title)
                sentiment_score = blob.sentiment.polarity
                sentiment_label = "บวก" if sentiment_score > 0 else "ลบ" if sentiment_score < 0 else "เป็นกลาง"
                news_list.append({
                    'title': title,
                    'publisher': source,
                    'link': url,
                    'sentiment': sentiment_score,
                    'sentiment_label': sentiment_label
                })
        return news_list
    except Exception as e:
        print("Error fetching news:", e)
        return []

@app.route('/analyze_stock')
def analyze_stock():
    symbol = request.args.get('symbol', '').upper()
    if not symbol:
        return jsonify({'error': 'กรุณาระบุสัญลักษณ์หุ้น'}), 400

    try:
        years_3 = 3
        years_1 = 1
        now = datetime.now()

        start_3y = now - timedelta(days=365 * years_3)
        start_1y = now - timedelta(days=365 * years_1)

        stock = yf.Ticker(symbol)

        df_3y = stock.history(start=start_3y.strftime('%Y-%m-%d'))
        df_1y = stock.history(start=start_1y.strftime('%Y-%m-%d'))

        if df_3y.empty or df_1y.empty:
            return jsonify({'error': 'ไม่มีข้อมูลหุ้นย้อนหลัง'}), 404

        df_3y = df_3y[['Close', 'Volume']].rename(columns={'Close': 'close', 'Volume': 'volume'})
        df_1y = df_1y[['Close', 'Volume']].rename(columns={'Close': 'close', 'Volume': 'volume'})

        # Indicators
        df_3y['sma_50'] = ta.trend.sma_indicator(df_3y['close'], window=50)
        df_3y['rsi'] = ta.momentum.RSIIndicator(df_3y['close']).rsi()
        df_3y['macd'] = ta.trend.MACD(df_3y['close']).macd_diff()
        df_3y.dropna(inplace=True)

        df_1y['log_return'] = (df_1y['close'] / df_1y['close'].shift(1)).apply(lambda x: 0 if x == 0 else np.log(x))
        volatility = df_1y['log_return'].std() * (252**0.5) * 100

        start_price_3y = df_3y['close'].iloc[0]
        end_price_3y = df_3y['close'].iloc[-1]
        pct_change_3y = ((end_price_3y - start_price_3y) / start_price_3y) * 100
        trend_3y = "📈 ขาขึ้น" if pct_change_3y > 20 else "📉 ขาลง" if pct_change_3y < -20 else "🔄 ทรงตัว"

        start_price_1y = df_1y['close'].iloc[0]
        end_price_1y = df_1y['close'].iloc[-1]
        pct_change_1y = ((end_price_1y - start_price_1y) / start_price_1y) * 100
        trend_1y = "📈 ขาขึ้น" if pct_change_1y > 20 else "📉 ขาลง" if pct_change_1y < -20 else "🔄 ทรงตัว"

        last_rsi = df_3y['rsi'].iloc[-1]
        last_macd = df_3y['macd'].iloc[-1]

        rsi_signal = "Overbought (RSI>70)" if last_rsi > 70 else "Oversold (RSI<30)" if last_rsi < 30 else "Neutral"
        macd_signal = "Buy Signal" if last_macd > 0 else "Sell Signal"

        if pct_change_1y > 20 and last_macd > 0:
            recommendation = "✅ แนะนำ: ซื้อ (แนวโน้มขาขึ้น + MACD บวก)"
        elif pct_change_1y < -20 and last_macd < 0:
            recommendation = "⚠️ แนะนำ: ขาย (แนวโน้มขาลง + MACD ติดลบ)"
        else:
            recommendation = "📌 แนะนำ: ถือ (แนวโน้มไม่ชัดเจน)"

        # คำแนะนำถือหุ้นระยะยาว
        if pct_change_3y > 20 and last_rsi < 70 and last_macd > 0:
            hold_advice = "แนะนำถือระยะยาว (3 ปีขึ้นไป) เนื่องจากแนวโน้มขาขึ้นและสัญญาณเทคนิคแข็งแรง"
        elif pct_change_1y > 10:
            hold_advice = "แนะนำถือระยะกลาง (6 เดือน - 1 ปี)"
        else:
            hold_advice = "แนะนำถือระยะสั้น หรือพิจารณาขาย"

        summary = f"หุ้น {symbol} มีแนวโน้ม {trend_1y} ในช่วง 1 ปีที่ผ่านมา RSI อยู่ที่ {round(last_rsi,2)} ({rsi_signal}) และ MACD = {round(last_macd,4)} ({macd_signal})"
        risk_warning = "การลงทุนมีความเสี่ยง ควรศึกษาข้อมูลก่อนตัดสินใจ"

        info = stock.info

        def safe_round(val, digits=2):
            try:
                return round(val*100, digits)
            except:
                return 'N/A'

        company = info.get('longName', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        dividend_yield = safe_round(info.get('dividendYield'))
        beta = info.get('beta', 'N/A')
        profit_margin = safe_round(info.get('profitMargins'))
        roe = safe_round(info.get('returnOnEquity'))

        latest_close = df_3y['close'].iloc[-1]
        latest_volume = df_3y['volume'].iloc[-1]
        last_date = df_3y.index[-1].strftime('%Y-%m-%d')

        # ดึงข่าวผ่าน NewsAPI (แทนของ yfinance news)
        news_list = fetch_news_api(symbol)
        avg_sentiment = 0
        if news_list:
            avg_sentiment = sum(n['sentiment'] for n in news_list) / len(news_list)

        # หุ้นกลุ่มเดียวกัน (peer comparison)
        peer_symbols = ['MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
        if symbol not in peer_symbols:
            peer_symbols.append(symbol)

        peer_data = []
        for peer_symbol in peer_symbols:
            if peer_symbol == symbol:
                continue
            peer_stock = yf.Ticker(peer_symbol)
            peer_hist = peer_stock.history(start=start_1y.strftime('%Y-%m-%d'))[['Close']]
            if peer_hist.empty:
                continue
            start_p = peer_hist['Close'].iloc[0]
            end_p = peer_hist['Close'].iloc[-1]
            pct_chg = ((end_p - start_p) / start_p) * 100
            peer_data.append({
                'symbol': peer_symbol,
                'pct_change_1y': round(pct_chg, 2)
            })

        img_base64 = plot_price_sma(df_3y, symbol)

        result = {
            'symbol': symbol,
            'company': company,
            'industry': industry,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'dividend_yield': dividend_yield,
            'beta': beta,
            'profit_margin': profit_margin,
            'roe': roe,
            'latest_close': round(latest_close, 2),
            'latest_volume': int(latest_volume),
            'volatility': round(volatility, 2) if volatility is not None else 'N/A',
            'pct_change_3y': round(pct_change_3y, 2),
            'trend_3y': trend_3y,
            'pct_change_1y': round(pct_change_1y, 2),
            'trend_1y': trend_1y,
            'rsi': round(last_rsi, 2),
            'rsi_signal': rsi_signal,
            'macd': round(last_macd, 4),
            'macd_signal': macd_signal,
            'last_date': last_date,
            'latest_news': news_list,
            'avg_news_sentiment': round(avg_sentiment, 4),
            'peer_comparison': peer_data,
            'graph_base64': img_base64,
            'summary': summary,
            'recommendation': recommendation,
            'hold_advice': hold_advice,
            'risk_warning': risk_warning
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'เกิดข้อผิดพลาดในการวิเคราะห์หุ้น'}), 500
if __name__ == '__main__':
    app.run(debug=True)
