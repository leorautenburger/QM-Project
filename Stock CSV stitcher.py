import pandas as pd
import glob
import os

# Set this to your CSV folder path
csv_dir = r"C:\Users\Leo Rautenberg\Documents\QM Project stuff\CSVs"
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

stock_data = {}
for file in csv_files:
    stock_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, parse_dates=["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    # Strip timezone info from the dates
    df["Date"] = df["Date"].apply(lambda x: x.replace(tzinfo=None))
    df.set_index("Date", inplace=True)
    quarterly = df["Close"].resample("Q").last()
    stock_data[stock_name] = quarterly

combined_df = pd.concat(stock_data, axis=1)
combined_df.index.name = "Quarter_End"
combined_df.to_excel("combined_stocks.xlsx")
