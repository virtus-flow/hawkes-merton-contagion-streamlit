# src/extractor.py
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class MarketDataExtractor50:
    @staticmethod
    def get_sp500_tickers(n=50):
        sp500_top = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO', 'JPM',
            'V', 'TSLA', 'XOM', 'UNH', 'PG', 'MA', 'JNJ', 'HD', 'COST', 'MRK',
            'ABBV', 'WMT', 'BAC', 'CRM', 'CVX', 'NFLX', 'ADBE', 'KO', 'PEP', 'TMO',
            'LIN', 'DIS', 'ORCL', 'CSCO', 'MCD', 'ACN', 'IBM', 'ABT', 'CAT', 'GE',
            'DHR', 'VZ', 'NOW', 'GS', 'PM', 'SPGI', 'QCOM', 'RTX', 'TXN', 'NEE'
        ]
        return sp500_top[:n]
    
    @staticmethod
    def fetch_equity_data(tickers, start_date, end_date):
        data = {}
        stock_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = stock_data
                else:
                    df = stock_data[ticker]
                if df.empty:
                    continue
                close = df['Close']
                info = yf.Ticker(ticker).info
                market_cap = info.get('marketCap')
                if market_cap is None:
                    shares = info.get('sharesOutstanding')
                    if shares is not None:
                        market_cap = close.iloc[-1] * shares
                    else:
                        market_cap = None
                returns = np.log(close / close.shift(1)).dropna()
                volatility = returns.std() * np.sqrt(252)
                data[ticker] = {
                    'market_cap': market_cap,
                    'volatility': volatility,
                    'prices': close,
                    'returns': returns
                }
            except:
                continue
        return data
    
    @staticmethod
    def extract_all(tickers, start_date, end_date, debt_ratio=0.4):
        equity_data = MarketDataExtractor50.fetch_equity_data(tickers, start_date, end_date)
        valid_tickers = [t for t in tickers if t in equity_data]
        if not valid_tickers:
            raise ValueError("Nijedan ticker nije validan.")
        returns_df = pd.DataFrame()
        for t in valid_tickers:
            returns_df[t] = equity_data[t]['returns']
        returns_df = returns_df.dropna(axis=0, how='any')
        corr_matrix = returns_df.corr().values
        equity_values = []
        equity_vols = []
        debt_values = []
        prices = {}
        for ticker in valid_tickers:
            eq = equity_data[ticker]
            market_cap = eq['market_cap']
            vol = eq['volatility']
            prices[ticker] = eq['prices']
            if market_cap is None:
                print(f"⚠️ Market cap za {ticker} nije dostupan, koristim procenu.")
                market_cap = eq['prices'].iloc[-1] * 1_000_000_000
            debt = debt_ratio * market_cap
            equity_values.append(market_cap)
            equity_vols.append(vol)
            debt_values.append(debt)
        return {
            'tickers': valid_tickers,
            'equity_values': np.array(equity_values),
            'equity_vols': np.array(equity_vols),
            'debt_values': np.array(debt_values),
            'correlation_matrix': corr_matrix,
            'returns': returns_df,
            'prices': pd.DataFrame(prices)
        }
