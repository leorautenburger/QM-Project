# === Sector-by-Sector Analysis of Attention and Returns ===
# This script runs a fixed-effects regression for each sector to estimate whether
# lagged Google Trends attention predicts next-quarter stock returns.
# It also outputs sector-level plots and a summary bar chart.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.formula.api as smf

# -------------------
# CONFIGURATION
# -------------------
lag = 0  # Number of quarters to lag attention scores
min_points = 30  # Minimum data points required to include a sector
dpi = 300  # Resolution for saved plots
output_dir = "sector_plots"  # Folder to save individual sector plots
# -------------------

# 1. Load pre-processed quarterly attention and price data
trends = pd.read_excel("combined_trends.xlsx")
prices = pd.read_excel("combined_stocks.xlsx")

# 2. Reshape to long format for merging
trends_long = trends.melt(id_vars="Quarter_End", var_name="ticker", value_name="attention")
prices_long = prices.melt(id_vars="Quarter_End", var_name="ticker", value_name="price")

# 3. Merge and sort chronologically
df = pd.merge(trends_long, prices_long, on=["Quarter_End", "ticker"], how="inner")
df["Quarter_End"] = pd.to_datetime(df["Quarter_End"])
df = df.sort_values(["ticker", "Quarter_End"])

# 4. Compute log returns and lagged attention values
df["log_return"] = df.groupby("ticker")["price"].transform(lambda x: np.log(x) - np.log(x.shift(1)))
df["attention_lag"] = df.groupby("ticker")["attention"].shift(lag)

# 5. Drop rows with missing values
df_clean = df.dropna(subset=["log_return", "attention_lag"]).copy()

# 6. Map company tickers to defined sectors
sector_map = {
    "WMT": "General Merchandisers",
    "COST": "General Merchandisers",
    "HD": "General Merchandisers",
    "KR": "General Merchandisers",
    "AMZN": "Internet Retail",
    "GOOG": "Internet Retail",
    "AAPL": "Tech",
    "MSFT": "Tech",
    "UNH": "Health Insurance",
    "CI": "Health Insurance",
    "CNC": "Health Insurance",
    "ELV": "Health Insurance",
    "BRK-B": "Financial Services",
    "JPM": "Financial Services",
    "BAC": "Financial Services",
    "C": "Financial Services",
    "XOM": "Petroleum",
    "CVX": "Petroleum",
    "MPC": "Petroleum",
    "CAH": "Wholesale HC",
    "MCK": "Wholesale HC",
    "CVS": "Pharmacy",
    "F": "Auto",
    "GM": "Auto"
}
df_clean["sector"] = df_clean["ticker"].map(sector_map).fillna("Other")
df_clean["quarter_str"] = df_clean["Quarter_End"].dt.to_period("Q").astype(str)

# 7. Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# 8. Run regressions and generate sector plots
sector_results = []

for sector in df_clean["sector"].dropna().unique():
    sector_data = df_clean[df_clean["sector"] == sector]

    if len(sector_data) < min_points:
        print(f"Skipping {sector} (only {len(sector_data)} rows)")
        continue

    unique_tickers = sector_data["ticker"].nunique()
    if unique_tickers < 2:
        print(f"Skipping {sector} (only {unique_tickers} unique ticker)")
        continue

    # Fixed-effects regression with firm and time dummies
    model = smf.ols("log_return ~ attention_lag + C(ticker) + C(quarter_str)", data=sector_data)
    results = model.fit(cov_type="cluster", cov_kwds={"groups": sector_data["ticker"]})

    coef = results.params.get("attention_lag", float("nan"))
    pval = results.pvalues.get("attention_lag", float("nan"))
    r2 = results.rsquared

    print(f"{sector:<22} | coef: {coef:+.5f} | p: {pval:.4f} | R²: {r2:.3f}")
    sector_results.append({"sector": sector, "coef": coef, "pval": pval, "r2": r2})

    # Plot scatter with OLS trendline (not fixed effects)
    plt.figure(figsize=(8, 5))
    sns.regplot(
        x="attention_lag",
        y="log_return",
        data=sector_data,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
        ci=None
    )
    plt.title(f"{sector} – Raw Scatter: Attention vs Return (Lag = {lag})")
    plt.xlabel("Lagged Google Trends Score")
    plt.ylabel("Quarterly Log Return")
    plt.grid(True)

    safe_name = sector.replace(" ", "_").replace("/", "_")
    plt.savefig(f"{output_dir}/{safe_name}_regression.png", dpi=dpi)
    plt.close()

# 9. Save summary of sector-level results
summary_df = pd.DataFrame(sector_results)
summary_df = summary_df.sort_values("pval")
summary_df.to_csv("sector_regression_summary.csv", index=False)
print("\nSaved regression summary to 'sector_regression_summary.csv'")

# 10. Generate summary bar chart of coefficients by sector
summary_df = summary_df.sort_values("coef")

plt.figure(figsize=(10, 6))
sns.barplot(
    data=summary_df,
    x="coef",
    y="sector",
    palette="coolwarm",
    orient="h"
)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Effect of Lagged Google Trends on Next-Quarter Returns by Sector")
plt.xlabel("Regression Coefficient (attention_lag)")
plt.ylabel("Sector")
plt.tight_layout()
plt.savefig("sector_regression_bar_chart.png", dpi=300)
plt.show()
