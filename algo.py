import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ==========================================
# 🌊 Ocean Protocol Extension DEMO Algoritms 🌊
# Select your demo mode below (change the number):
# 1 - Bitcoin Trend Analysis (Price + SMA)
# 2 - OCEAN Token Health (Volume & Volatility)
# 3 - AI Price Prediction (Machine Learning)
# 4 - 🔥 Top 7 Trending Coins (Live Data)
# 5 - ⚖️ ETH vs BTC Correlation (Market Check)
# ==========================================
DEMO_MODE = 1
# ==========================================

def fetch_data(coin_id, days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    print(f"Fetching data for {coin_id} from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    prices = data['prices']
    total_volumes = data['total_volumes']
    
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    
    vol_df = pd.DataFrame(total_volumes, columns=['timestamp', 'volume'])
    vol_df['date'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
    vol_df.set_index('date', inplace=True)
    df['volume'] = vol_df['volume']
    
    return df

def run_bitcoin_analysis():
    print("--- Running Mode 1: Bitcoin Trend Analysis ---")
    df = fetch_data('bitcoin', days=30)
    
    df['SMA_7'] = df['price'].rolling(window=7).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['price'], label='BTC Price (USD)', color='#F7931A', linewidth=2)
    plt.plot(df.index, df['SMA_7'], label='7-Day SMA', color='blue', linestyle='--', alpha=0.7)
    
    plt.title('Bitcoin Price Analysis (Last 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot('btc_trend_analysis.png')
    save_data(df, 'btc_data.csv')
    return {"message": "Bitcoin Analysis Complete", "last_price": df['price'].iloc[-1]}

def run_ocean_health():
    print("--- Running Mode 2: OCEAN Token Health ---")
    df = fetch_data('ocean-protocol', days=90)
    
    df['returns'] = df['price'].pct_change()
    volatility = df['returns'].std() * 100
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#FF4081'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)', color=color)
    ax1.plot(df.index, df['price'], color=color, label='OCEAN Price')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = '#7B1FA2'
    ax2.set_ylabel('Volume', color=color)
    ax2.fill_between(df.index, df['volume'], color=color, alpha=0.1, label='Volume')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'OCEAN Token Health (Volatility: {volatility:.2f}%)')
    fig.tight_layout()
    
    save_plot('ocean_health.png')
    save_data(df, 'ocean_data.csv')
    return {"message": "OCEAN Health Analysis Complete", "volatility_percent": volatility}

def run_ai_prediction():
    print("--- Running Mode 3: AI Price Prediction ---")
    df = fetch_data('bitcoin', days=60)
    
    df['days_from_start'] = (df.index - df.index[0]).days
    X = df[['days_from_start']].values
    y = df['price'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_day = X[-1][0] + 1
    prediction = model.predict([[next_day]])[0]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, y, color='gray', alpha=0.5, label='Actual Prices')
    plt.plot(df.index, model.predict(X), color='red', linewidth=2, label='AI Trend Line')
    
    last_date = df.index[-1]
    next_date = last_date + pd.Timedelta(days=1)
    plt.scatter([next_date], [prediction], color='green', s=100, zorder=5, label=f'AI Prediction: ${prediction:,.0f}')
    
    plt.title('AI Price Prediction (Linear Regression)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot('ai_prediction.png')
    return {"message": "AI Prediction Complete", "predicted_price": prediction}

def run_trending_coins():
    print("--- Running Mode 4: Top 7 Trending Coins ---")
    url = "https://api.coingecko.com/api/v3/search/trending"
    print(f"Fetching trending coins from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    coins = data['coins'][:7]
    names = [c['item']['name'] for c in coins]
    ranks = [c['item']['market_cap_rank'] for c in coins]
    ids = [c['item']['id'] for c in coins]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, ranks, color='#4CAF50')
    plt.xlabel('Market Cap Rank (Lower is Better)')
    plt.title('🔥 Top 7 Trending Coins on CoinGecko')
    plt.gca().invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'Rank: {width}', va='center')
        
    plt.grid(axis='x', alpha=0.3)
    
    save_plot('trending_coins.png')
    return {"message": "Trending Coins Analysis Complete", "top_coin": names[0]}

def run_eth_btc_correlation():
    print("--- Running Mode 5: ETH vs BTC Correlation ---")
    
    btc = fetch_data('bitcoin', days=90)
    eth = fetch_data('ethereum', days=90)
    
    df = pd.DataFrame({'BTC': btc['price'], 'ETH': eth['price']})
    df.dropna(inplace=True)
    
    correlation = df['BTC'].corr(df['ETH'])
    
    df_norm = df / df.iloc[0] * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_norm.index, df_norm['BTC'], label='Bitcoin', color='#F7931A')
    plt.plot(df_norm.index, df_norm['ETH'], label='Ethereum', color='#627EEA')
    
    plt.title(f'ETH vs BTC Correlation (Corr: {correlation:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Start=100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot('eth_btc_correlation.png')
    save_data(df, 'correlation_data.csv')
    return {"message": "Correlation Analysis Complete", "correlation": correlation}

def save_plot(filename):
    output_dir = '/data/outputs' if os.path.exists('/data/outputs') else './data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    print(f"Saved chart to: {path}")
    print(f"👉 Local file: results/{filename}")

def save_data(df, filename):
    output_dir = '/data/outputs' if os.path.exists('/data/outputs') else './data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path)
    print(f"Saved data to: {path}")
    print(f"👉 Local file: results/{filename}")

def main():
    print(f"Starting Ocean Compute Job... (Mode: {DEMO_MODE})")
    
    result = {}
    try:
        if DEMO_MODE == 1:
            result = run_bitcoin_analysis()
        elif DEMO_MODE == 2:
            result = run_ocean_health()
        elif DEMO_MODE == 3:
            result = run_ai_prediction()
        elif DEMO_MODE == 4:
            result = run_trending_coins()
        elif DEMO_MODE == 5:
            result = run_eth_btc_correlation()
        else:
            print("Invalid Mode Selected! Defaulting to Mode 1.")
            result = run_bitcoin_analysis()
            
    except Exception as e:
        print(f"Error: {e}")
        result = {"error": str(e)}
        
    output_dir = '/data/outputs' if os.path.exists('/data/outputs') else './data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'result.json'), 'w') as f:
        json.dump(result, f)
    
    print("Job Completed Successfully! 🌊")
    print("📂 Local Results: ./results")

if __name__ == "__main__":
    main()
