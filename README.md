# 🌊 Ocean Protocol Demo Algorithms for Vs Code Extension

##  Getting Started

### 1. Prerequisites
- **VS Code** (Version 1.96.0 or higher)  [Download](https://code.visualstudio.com/)
- **Ocean Protocol VS Code Extension** ([Install from the Marketplace](https://marketplace.visualstudio.com/items?itemName=OceanProtocol.ocean-protocol-vscode-extension&ssr=false#overview))

### 2. Installation
1.  Clone or download this repository.
2.  Open the folder in VS Code.
3.  You should see the Ocean Protocol icon in your Activity Bar.

### 3. How to Run a Job
1.  Open the file `algo.py`.
2.  At the top of the file, you will see a **Configuration Section**:
    ```python
    # ==========================================
    # 🌊 OCEAN PROTOCOL DEMO SUITE 🌊
    # Select your demo mode below (change the number):
    # 1 - Bitcoin Trend Analysis (Price + SMA)
    # 2 - OCEAN Token Health (Volume & Volatility)
    # 3 - AI Price Prediction (Machine Learning)
    # 4 - 🔥 Top 7 Trending Coins (Live Data)
    # 5 - ⚖️ ETH vs BTC Correlation (Market Check)
    # ==========================================
    DEMO_MODE = 1
    ```
3.  Change `DEMO_MODE` to the number of the demo you want to run (e.g., `3` for AI Prediction).
4.  Open the **Ocean Protocol Extension** panel in the sidebar.
5.  Click **Start FREE Compute Job**.

### 4. Viewing Results
Once the job is complete (usually takes 10-30 seconds), check the `results/` folder in your project. You will find:
-   **Charts** (e.g., `ai_prediction.png`, `trending_coins.png`)
-   **Data** (e.g., `btc_data.csv`)
-   **Logs** (`result.json`)

## 🧪 Available Modes

| Mode | Name | Description |
| :--- | :--- | :--- |
| **1** | **Bitcoin Trend Analysis** | Plots the last 30 days of BTC price with a 7-day Simple Moving Average (SMA). |
| **2** | **OCEAN Token Health** | Analyzes the volatility and volume of the OCEAN token over 90 days. |
| **3** | **AI Price Prediction** | Uses a Linear Regression model (Machine Learning) to predict tomorrow's Bitcoin price. |
| **4** | **Trending Coins** | Shows the Top 7 most searched coins on CoinGecko right now. |
| **5** | **ETH vs BTC Correlation** | Compares the price movements of Ethereum and Bitcoin to see how closely they are linked. |

---
*Powered for Ocean Protocol* 🌊
