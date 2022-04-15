import yfinance as yf

cryptocurrencies_btc = ['BTC-USD']
bitcoin_df = yf.download(cryptocurrencies_btc, start='2019-01-01',
                end='2022-03-01', parse_dates = ['Month'])

cryptocurrencies_ada = ['ADA-USD']
cardano_df = yf.download(cryptocurrencies_ada, start='2019-01-01',
                end='2022-03-01', parse_dates = ['Month'])

cryptocurrencies_eth = ['ETH-USD']
ethereum_df = yf.download(cryptocurrencies_eth, start='2019-01-01',
                end='2022-03-01', parse_dates = ['Month'])