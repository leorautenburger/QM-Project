# === General Analysis: Google Trends vs Stock Returns ===
# This script runs a fixed-effects regression to test whether lagged Google Trends
# scores predict next-quarter log returns across all Fortune 25 companies.

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# Set lag length to use in the regression
lag = 4  # Number of quarters to lag attention data

# Load combined quarterly data
trends = pd.read_excel("combined_trends.xlsx")
prices = pd.read_excel("combined_stocks.xlsx")

# Convert from wide to long format for merging
trends_long = trends.melt(id_vars="Quarter_End", var_name="ticker", value_name="attention")
prices_long = prices.melt(id_vars="Quarter_End", var_name="ticker", value_name="price")

# Merge Trends and Price data
df = pd.merge(trends_long, prices_long, on=["Quarter_End", "ticker"], how="inner")
df["Quarter_End"] = pd.to_datetime(df["Quarter_End"])
df = df.sort_values(["ticker", "Quarter_End"])

# Compute log returns and lagged attention scores
df["log_return"] = df.groupby("ticker")["price"].transform(lambda x: np.log(x) - np.log(x.shift(1)))
df["attention_lag"] = df.groupby("ticker")["attention"].shift(lag)

# Drop rows with missing values (usually early quarters)
df_clean = df.dropna(subset=["log_return", "attention_lag"])

# Create quarter identifiers for fixed effects
df_clean["quarter_str"] = df_clean["Quarter_End"].dt.to_period("Q").astype(str)

# Run fixed-effects regression with firm and time dummies, clustered by ticker
model = smf.ols("log_return ~ attention_lag + C(ticker) + C(quarter_str)", data=df_clean)
results = model.fit(cov_type="cluster", cov_kwds={"groups": df_clean["ticker"]})

# Extract key statistics
coef = results.params.get("attention_lag", np.nan)
stderr = results.bse.get("attention_lag", np.nan)
pval = results.pvalues.get("attention_lag", np.nan)
r2 = results.rsquared

# Format and save regression summary
summary_text = (
    f"=== Regression Summary (Lag = {lag}) ===\n"
    f"Coefficient for attention_lag : {coef:.6f}\n"
    f"Standard Error               : {stderr:.6f}\n"
    f"P-value                      : {pval:.6f}\n"
    f"R-squared                    : {r2:.4f}\n"
)
print("\n" + summary_text)
with open(f"regression_summary_lag_{lag}.txt", "w") as f:
    f.write(summary_text)

# Plot raw scatterplot with regression line (note: this does not reflect fixed effects)
plt.figure(figsize=(8, 5))
sns.regplot(
    x="attention_lag",
    y="log_return",
    data=df_clean,
    scatter_kws={"alpha": 0.4},
    line_kws={"color": "red"},
    ci=None
)
plt.xlabel(f"Lagged by {lag} Quarter Google Trends Score")
plt.ylabel("Quarterly Log Return")
plt.title(f"Raw Scatter: Attention vs Return (Lag = {lag})")
plt.grid(True)
plt.savefig(f"regression_plot_lag_{lag}.png", dpi=300)
plt.show()
