import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

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
        """
        Preuzima podatke za listu tickera.
        Podržava bilo koji ticker, čak i samo jedan.
        """
        data = {}
        tickers = [t.strip() for t in tickers if t.strip()]
        if not tickers:
            return data
        
        # Pokušaj batch download (brže)
        try:
            stock_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
            
            # Ako je samo jedan ticker, yfinance vraća DataFrame, ne dict
            if len(tickers) == 1:
                # Tada je stock_data DataFrame, ali ga treba pretvoriti u dict sa tickerom kao ključem
                if not stock_data.empty:
                    stock_data = {tickers[0]: stock_data}
                else:
                    stock_data = {}
        except Exception as e:
            print(f"⚠️ Batch download neuspješan: {e}")
            stock_data = {}
        
        # Za svaki ticker, izvuci podatke
        for ticker in tickers:
            try:
                df = None
                # Pokušaj dohvatiti DataFrame iz batch rezultata
                if ticker in stock_data:
                    df = stock_data[ticker]
                else:
                    # Ako nema u batch-u, pokušaj pojedinačno
                    ticker_obj = yf.Ticker(ticker)
                    df = ticker_obj.history(start=start_date, end=end_date)
                    if df.empty:
                        # Pokušaj sa alternativnim nazivom (npr. BRK-B -> BRK.B)
                        if ticker == 'BRK-B':
                            alt = 'BRK.B'
                            ticker_obj = yf.Ticker(alt)
                            df = ticker_obj.history(start=start_date, end=end_date)
                        if df.empty:
                            print(f"⚠️ Nema podataka za {ticker}")
                            continue
                
                if df is None or df.empty:
                    print(f"⚠️ Nema podataka za {ticker}")
                    continue
                
                close = df['Close']
                if close.empty or len(close) < 2:
                    print(f"⚠️ Premalo podataka za {ticker}")
                    continue
                
                # Pokušaj dobiti market cap
                info = yf.Ticker(ticker).info
                market_cap = info.get('marketCap')
                if market_cap is None or market_cap <= 0:
                    shares = info.get('sharesOutstanding')
                    if shares is not None and shares > 0:
                        market_cap = close.iloc[-1] * shares
                    else:
                        market_cap = None
                
                returns = np.log(close / close.shift(1)).dropna()
                if len(returns) < 2:
                    print(f"⚠️ Premalo prinosa za {ticker}")
                    continue
                volatility = returns.std() * np.sqrt(252)
                
                data[ticker] = {
                    'market_cap': market_cap,
                    'volatility': volatility,
                    'prices': close,
                    'returns': returns
                }
            except Exception as e:
                print(f"⚠️ Greška za {ticker}: {e}")
                continue
        
        return data
    
    @staticmethod
    def extract_all(tickers, start_date, end_date, debt_ratio=0.4):
        """
        Glavna metoda – vraća sve potrebne podatke za model.
        """
        equity_data = MarketDataExtractor50.fetch_equity_data(tickers, start_date, end_date)
        valid_tickers = list(equity_data.keys())
        if not valid_tickers:
            raise ValueError("Nijedan ticker nije validan.")
        
        # Kreiraj DataFrame prinosa za korelaciju
        returns_df = pd.DataFrame()
        for t in valid_tickers:
            returns_df[t] = equity_data[t]['returns']
        # Ukloni redove sa NaN (ako ih ima)
        returns_df = returns_df.dropna(axis=0, how='any')
        if returns_df.empty:
            raise ValueError("Nema dovoljno podataka za korelaciju.")
        
        # Korelaciona matrica (može biti 1x1)
        corr_matrix = returns_df.corr().values
        
        # Izvuci vrijednosti
        equity_values = []
        equity_vols = []
        debt_values = []
        prices = {}
        for ticker in valid_tickers:
            eq = equity_data[ticker]
            market_cap = eq['market_cap']
            vol = eq['volatility']
            prices[ticker] = eq['prices']
            if market_cap is None or market_cap <= 0:
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
