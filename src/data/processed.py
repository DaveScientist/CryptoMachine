import yfinance as yf

cryptocurrencies = ['BTC-USD']
bitcoin_df = yf.download(cryptocurrencies, start='2019-01-01',
                end='2022-03-01', parse_dates = ['Month'])

bitcoin_df["Volume"] = bitcoin_df["Volume"].astype("float")
bitcoin_df.info()

bitcoin_df = bitcoin_df.drop(columns = ['Adj Close'])
close = bitcoin_df['Close']

cryptocurrencies = ['ADA-USD']
cardano_df = yf.download(cryptocurrencies, start='2019-01-01',
                end='2022-03-01', parse_dates = ['Month'])

cardano_df["Volume"] = cardano_df["Volume"].astype("float")
cardano_df.info()

cardano_df = cardano_df.drop(columns = ['Adj Close'])
close = cardano_df['Close']

cryptocurrencies = ['ETH-USD']
ethereum_df = yf.download(cryptocurrencies, start='2019-01-01',
                end='2022-03-01', parse_dates = ['Month'])

ethereum_df["Volume"] = ethereum_df["Volume"].astype("float")
ethereum_df = ethereum_df.drop(columns = ['Adj Close'])
close = ethereum_df['Close']

