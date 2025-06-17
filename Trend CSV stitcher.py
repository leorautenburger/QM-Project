# === Google Trends Data Aggregation Script ===
# This script loads monthly Google Trends CSVs, extracts data from the correct header row,
# cleans and aligns them, averages values by quarter, and combines them into a single DataFrame.

import pandas as pd
import glob
import os

csv_dir = r"C:\Users\###########\Documents\QM Project stuff\trends CSVs"
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

trend_data = {}

# Loop through each file in the folder
for file in csv_files:
    # Use filename (minus .csv) as the column name
    trend_name = os.path.splitext(os.path.basename(file))[0]

    # Identify the line that contains the actual header
    header_line_index = None
    with open(file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if "Month" in line:
                header_line_index = i
                break
    if header_line_index is None:
        continue  # skip file if no "Month" found

    # Read the CSV, starting from the correct header line
    df = pd.read_csv(file, sep=",", skiprows=header_line_index, engine="python")
    df.columns = df.columns.str.strip()  # clean column names

    if "Month" not in df.columns:
        continue  # skip if structure is unexpected

    # Rename for consistency
    df.rename(columns={"Month": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="coerce")

    # Identify the search interest column (assumes only one besides Date)
    trend_cols = [col for col in df.columns if col != "Date"]
    if not trend_cols:
        continue

    trend_col = trend_cols[0]
    df[trend_col] = pd.to_numeric(df[trend_col], errors="coerce")
    df.set_index("Date", inplace=True)

    # Resample monthly data into quarterly averages
    quarterly_avg = df.resample("Q").mean()
    quarterly_avg.rename(columns={trend_col: trend_name}, inplace=True)

    # Store cleaned and averaged data
    trend_data[trend_name] = quarterly_avg[trend_name]

# Combine all trends into a single DataFrame, indexed by quarter-end
combined_df = pd.DataFrame(trend_data)
combined_df.index.name = "Quarter_End"

# Export to Excel for use in regression analysis
combined_df.to_excel("combined_trends_quarterly_avg.xlsx")
