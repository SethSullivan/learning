import yfinance as yf
import polars as pl


def get_stock_df(symbols, start_date):
    og_df = yf.download(symbols, "2019-01-01")  # Download all tickers
    if og_df is None:
        raise ValueError("No data downloaded from yfinance.")

    # Stack to long df
    og_dff = og_df.stack(level=1).reset_index()

    print(og_dff.head())
    df = pl.DataFrame(og_dff)
    return df
