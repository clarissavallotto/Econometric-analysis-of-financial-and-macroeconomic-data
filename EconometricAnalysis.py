import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from linearmodels.panel import PanelOLS
from scipy import stats
from google.colab import files

if os.path.exists("data.xlsx"):
    print("'data.xlsx' is available.")
else:
    print("Please upload 'data.xlsx'")
    uploaded = files.upload()

# Financial statement data
df_financial = pd.read_excel('data.xlsx', sheet_name = "Financial_statement")

# Loan data
df_loan = pd.read_excel ('data.xlsx', sheet_name = "Loan_details")

# Merge the loan data with the financial statement data on c_id
# one observation for each loan granted

# Check key uniqueness in financial statement dataset
unique_c_id = df_financial['c_id'].is_unique

if unique_c_id:
    print("\nFinancial statement data keys are unique")
else:
    print("\nPlease note that financial statement data keys are not unique")

merged_df = pd.merge(df_loan, df_financial, on = 'c_id', how = 'left')

# Number of observations in merged dataset
num_merged_observations = len(merged_df)
print(f"\nNumber of merged observations: {num_merged_observations}")

# interest expense variable
merged_df['interest_expense'] = merged_df['interest_rate'] * merged_df['loan_amount']

# Maximum interest expense on a loan
max_interest_expense = merged_df['interest_expense'].max()
print(f"\nMaximum interest expense: {max_interest_expense:.8f} EUR million\n")

# Descriptive statistics for the leverage ratio, number of employees, loan amount, and return on assets (mean, median, 25th and 75th percentiles, the minimum, the maximum)
desc_stats = merged_df[['leverage', 'employees', 'loan_amount', 'roa']].describe(percentiles = [.25, .5, .75]).loc[['mean', '50%', '25%', '75%', 'min', 'max']]
desc_stats = desc_stats.rename(index = {'50%': 'median'})
print(desc_stats.to_markdown())

# Export descriptive statistics to Excel
desc_stats.to_excel("descriptive_statistics.xlsx", sheet_name = "ds")
files.download("descriptive_statistics.xlsx")

# Data quality remark
print("\nThe minimum number of employees is -98, which is not a valid employee count.\n")

# Scatterplot that shows the relationship between the leverage ratio and the interest rate for loans granted by Lender 2
b2_df = merged_df[merged_df['b_id'] == 2].copy()

plt.figure(figsize = (14, 7))
plt.scatter(b2_df['leverage'], b2_df['interest_rate'], alpha = 0.5, s = 20)
plt.title('Interest rate vs. leverage for Lender 2')
plt.xlabel('leverage ratio')
plt.ylabel('interest rate')
plt.grid(True, alpha = 0.3)
plt.savefig('leverage_ratio_interest_rate.png')
plt.show()

print("""
The scatterplot shows a positive relationship between the leverage ratio and the interest rate.
This suggests that as a firm's leverage increases, Lender 2 charges a higher interest rate, reflecting a higher perceived risk.
""")

# Regression: interest rate  = α + β_1*roa + β_2*total assets + β_3*public + ε
formula_ir = 'interest_rate ~ roa + total_assets + public'
model_ir = smf.ols(formula_ir, data = merged_df).fit()

print(model_ir.summary())

# Export regression summary to html function
def export_summary(model, filename):
    summary = model.summary
    try:
        html = model.summary().as_html()
    except:
        try:
            html = summary.as_html()
        except:
            html = f"<pre>{str(summary)}</pre>"
    with open(filename, "w") as f:
        f.write(html)
    files.download(filename)

# Export regression summary to html (open with Excel)
export_summary(model_ir, "regression_summary_IR.html")

print("""
The interest rate firms pay is significantly influenced by their profitability (p-value = 0.000 < 0.010) and size (p-value = 0.000 < 0.010).
More profitable (higher ROA) and larger (higher total assets) firms get lower interest rates."
There is no statistically significant relationship between the firm being publicly listed and the interest rate (p-value = 0.382).

Omitted Variable Bias:
the model is missing important variables that likely influence interest rates, such as the firm's leverage ratio and the term of the loan.
Omitting these variables can lead to biased coefficient estimates.

Multicollinearity:
the high condition number in the regression output suggests there may be multicollinearity among the independent variables,
which could make the coefficient estimates unstable and unreliable.
""")

################################################################################

df_gdp_debt = pd.read_excel('data.xlsx', sheet_name = "data_A")

df_gdp_debt.rename(columns = {df_gdp_debt.columns[0]: 'Date'}, inplace = True)
df_gdp_debt['Date'] = pd.to_datetime(df_gdp_debt['Date'])
df_gdp_debt.set_index('Date', inplace = True)

# debt-to-GDP ratio
df_gdp_debt['debt_to_GDP_ratio'] = (df_gdp_debt['g_debt'] / df_gdp_debt['GDP']) * 100

# debt-to-GDP ratio in December 2003
debt_2003 = df_gdp_debt.loc['2003-12-01', 'g_debt']
gdp_2003 = df_gdp_debt.loc['2003-12-01', 'GDP']
ratio_2003 = (debt_2003 / gdp_2003) * 100
print(f"debt-to-GDP ratio in December 2003: {ratio_2003:.2f}%")

# debt-to-GDP ratio in December 2023
debt_2023 = df_gdp_debt.loc['2023-12-01', 'g_debt']
gdp_2023 = df_gdp_debt.loc['2023-12-01', 'GDP']
ratio_2023 = (debt_2023 / gdp_2023) * 100
print(f"debt-to-GDP ratio in December 2023: {ratio_2023:.2f}%")

# debt-to-GDP ratio plot
plt.figure(figsize = (14, 7))
plt.plot(df_gdp_debt.index, df_gdp_debt['debt_to_GDP_ratio'], marker = 'o', linestyle = '-', color = 'b')
plt.title('debt-to-GDP ratio over time', fontsize = 16)
plt.xlabel('date', fontsize = 12)
plt.ylabel('debt-to-GDP ratio (%)', fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.savefig('debt_to_gdp_ratio.png')
plt.tight_layout()
plt.show()

# Interest rate plot
plt.figure(figsize = (14, 7))
plt.plot(df_gdp_debt.index, df_gdp_debt['i_rate'], marker = 'o', linestyle = '-', color = 'r')
plt.title('Interest rate over time', fontsize = 16)
plt.xlabel('date', fontsize = 12)
plt.ylabel('interest rate (%)', fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.savefig('interest_rate.png')
plt.tight_layout()
plt.show()

print("""
The debt-to-GDP ratio has been on a consistent upward trend since the beginning of the series,
with sharp increases during major economic events like the 2008-2009 financial crisis and the COVID-19 pandemic.

The interest rate experienced a prolonged decline after 2009, reaching historic lows in the mid-2010s, but has since begun to increase sharply.
""")

################################################################################

dfB = pd.read_excel('data.xlsx', sheet_name = "data_B")

# loans-to-total assets ratio
dfB['loans_to_total_assets'] = dfB['loans'] / dfB['total_assets']

# operating expenses-to-total assets ratio
dfB['exp_operating_to_total_assets'] = dfB['exp_operating'] / dfB['total_assets']

# Pooled OLS regression.
# The standard errors are clustered by Lenders to account for the panel structure and heteroskedasticity (robustness)

# Regression: roe  = α + β_1*loans_to_total_assets + β_2*exp_operating_to_total_assets + β_3*cet1_ratio + β_4*gdp + β_5*interest_rate + ε
formula_roe = 'roe ~ loans_to_total_assets + exp_operating_to_total_assets + cet1_ratio + gdp + interest_rate'
model_roe = smf.ols(formula_roe, data = dfB).fit(cov_type = 'cluster', cov_kwds = {'groups': dfB['b_id']})

print(model_roe.summary())

# Export regression summary to html (open with Excel)
export_summary(model_roe, "regression_summary_ROE.html")

print("""
Based on the regression results, the following variables have a statistically significant positive effect on ROE:
- gdp --> The coefficient (0.3437) is positive and significant at a 1% level (p-value = 0.000 < 0.010).
- interest_rate --> The coefficient (0.8466) is positive and statistically significant at a 1% level (p-value = 0.000 < 0.010).
""")

# Regression: roe  = α + β_1*loans_to_total_assets + β_2*exp_operating_to_total_assets + β_3*cet1_ratio + β_4*gdp + β_5*interest_rate + β_5*FE(Lender) + ε
formula_roe_FE = 'roe ~ loans_to_total_assets + exp_operating_to_total_assets + cet1_ratio + gdp + interest_rate + C(b_id)'
model_roe_FE = smf.ols(formula_roe_FE, data = dfB).fit(cov_type = 'cluster', cov_kwds = {'groups': dfB['b_id']})

print(model_roe_FE.summary())

# Export regression summary to html (open with Excel)
export_summary(model_roe_FE, "regression_summary_ROE_FE.html")

print("""
The purpose of including fixed effects is to control for any unobserved, time-invariant heterogeneity that exists across different Lenders.
This accounts for unique characteristics of each lender that do not change over the sample period, such as its location.
By including a dummy variable for each Lender, the fixed effects model isolates the effect of changes within each Lender over time,
rather than comparing differences between Lenders.
This results in more reliable coefficient estimates for the other variables.
""")

# Regression: roe  = α + β_1*loans_to_total_assets + β_2*exp_operating_to_total_assets + β_3*cet1_ratio + β_4*gdp + β_5*interest_rate*FE(interest_rate_fixation) + ε
formula_roe_ir_fix = 'roe ~ loans_to_total_assets + exp_operating_to_total_assets + cet1_ratio + gdp + interest_rate * C(interest_rate_fixation)'
model_roe_ir_fix = smf.ols(formula_roe_ir_fix, data = dfB).fit(cov_type = 'cluster', cov_kwds = {'groups': dfB['b_id']})

print(model_roe_ir_fix.summary())

# Export regression summary to html (open with Excel)
export_summary(model_roe_ir_fix, "regression_summary_ROE_IR_fix.html")

print("""
The output uses floating-rate loans as the base category.
Therefore, for Lenders in fixed-rate countries, a 1 percentage point increase in the GDP is associated with a 0.3455 percentage point increase in the roe.
This effect is statistically significant (p-value = 0.000 < 0.010).
""")

################################################################################

dfC = pd.read_excel('data.xlsx', sheet_name = "data_C")

# Convert the 'Time' column to a proper datetime index
def convert_to_date(time_str):
    year = int(time_str[-4:])
    quarter = int(time_str[0])
    month = 3 * quarter
    return pd.to_datetime(f'{year}-{month}-01')

dfC['Date'] = dfC['Time'].apply(convert_to_date)
dfC = dfC.set_index('Date').sort_index()

# Variables
vars_p = ['GDP', 'PGDP', 'TC', 'HHTC', 'NFCTC', 'IR']

# Check missing values
print("Missing values per variable:")
print(dfC[vars_p].isna().sum())

# Fill missing data: linear interpolation
dfC[vars_p] = dfC[vars_p].interpolate(method = 'linear')

# Final check
remaining_nans = dfC[vars_p].isna().sum().sum()

if remaining_nans == 0:
    print("\nAfter filling the missing data using linear interpolation, the dataset is now complete.\n")
else:
    print(f"\nAfter filling the missing data using linear interpolation, the dataset still has {remaining_nans} missing observations.\n")

# Folder to save plots
output_folder = "plots"

os.makedirs(output_folder, exist_ok = True)

# Loop and save each plot separately
for var in vars_p:
    plt.figure(figsize = (14, 7))
    plt.plot(dfC.index, dfC[var], linewidth = 1.5)
    plt.title(f"Time series of {var}")
    plt.xlabel("year")
    plt.grid(True, alpha = 0.3)

    filename = f"{output_folder}/{var}_timeseries.png"
    plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
    plt.show()

    plt.close()

# The annual growth rate of a variable 'X' is calculated as: (Xt / Xt-4 - 1) * 100

# Annual growth rate of (real) GDP
dfC['GDP_annual_growth'] = (dfC['GDP'] / dfC['GDP'].shift(4) - 1) * 100

# Real total credit (RTC) is derived by dividing TC by PGDP
dfC['RTC'] = dfC['TC'] / dfC['PGDP']

# Annual growth rate of real total credit to non-financial private sector (RTC)
dfC['RTC_annual_growth'] = (dfC['RTC'] / dfC['RTC'].shift(4) - 1) * 100

plt.figure(figsize = (14, 7))
plt.plot(dfC.index, dfC['GDP_annual_growth'], label = 'annual GDP growth rate (%)')
plt.plot(dfC.index, dfC['RTC_annual_growth'], label = 'annual RTC growth rate (%)')
plt.title('Annual growth rates of real GDP and real total credit')
plt.xlabel('year')
plt.ylabel('annual growth rate (%)')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.savefig('annual_growth_rates.png')
plt.tight_layout()
plt.show()

print("""
The chart shows that real total credit (RTC) appears to be procyclical, meaning it generally moves in tandem with the business cycle, as represented by real GDP growth.

The annual growth rate of RTC rises during periods of economic expansion (high GDP growth) and falls during economic contractions (low or negative GDP growth).
For example, during recessions, such as the early 1980s, early 1990s, and the Great Recession of 2008-2009, both GDP growth and RTC growth experienced sharp declines.
Conversely, during periods of strong economic performance, the growth rates of both variables tend to increase.
""")

# First Difference of total credit to non-financial private sector (TC)
dfC['TC_FD'] = dfC['TC'].diff()

# Annual growth rate of total credit to non-financial private sector (TC)
dfC['TC_annual_growth'] = (dfC['TC'] / dfC['TC'].shift(4) - 1) * 100

# Annual change in the interest rate (independent variable)
dfC['IR_change'] = dfC['IR'] - dfC['IR'].shift(4)

# Drop NaN values created by the shifting operation
dfC = dfC.dropna()

# Select only numeric columns (avoids errors with non-numeric data like 'Date')
num_cols = dfC.select_dtypes(include = 'number').columns
num_cols = num_cols.drop(['Year', 'Quarter'], errors='ignore')

# Mean and standard deviation
des = dfC[num_cols].describe().T[['mean', 'std']]

# Lag-1 autocorrelation for all numeric variables
des['autocorr_lag1'] = dfC[num_cols].apply(lambda x: x.autocorr(lag = 1))

print("Basic statistics:")
print(des)

# Share of credit to households (HHTC) in the total credit to the non‐financial sector (TC)
dfC["HHTC_TC"] = (dfC["HHTC"] / dfC["TC"]) * 100

# Share of credit to non-financial firms (NFCTC) in the total credit to the non‐financial sector (TC)
dfC["NFCTC_TC"] = (dfC["NFCTC"] / dfC["TC"]) * 100

# Share of credit to households (HHTC) in the total credit to the non‐financial sector (TC) in December 2014
HHTC_TC_2014_end = dfC.loc['2014-12-01', 'HHTC_TC']
print(f"\nShare of credit to households in the total credit to the non‐financial sector in December 2014: {HHTC_TC_2014_end:.2f}%")

# Share of credit to non-financial firms (NFCTC) in the total credit to the non‐financial sector (TC) in December 2014
NFCTC_TC_2014_end = dfC.loc['2014-12-01', 'NFCTC_TC']
print(f"\nShare of credit to non-financial firms in the total credit to the non‐financial sector in December 2014: {NFCTC_TC_2014_end:.2f}%\n")

# Credit to households (HHTC) and credit to non-financial firms (NFCTC) annual growth rate contributions to the annual growth rate of total credit to non-financial private sector (TC)
dfC['HHTC_ag_contribution'] = (dfC['HHTC'] - dfC['HHTC'].shift(4)) / dfC['TC'].shift(4) * 100
dfC['NFCTC_ag_contribution'] = (dfC['NFCTC'] - dfC['NFCTC'].shift(4)) / dfC['TC'].shift(4) * 100

# Stacked bar chart
df_plot = dfC[['HHTC_ag_contribution', 'NFCTC_ag_contribution']].dropna()

# Convert x-axis to numeric positions
x = np.arange(len(df_plot))

fig, ax = plt.subplots(figsize = (14, 7))

# Stacked bars (use numeric x)
ax.bar(x,
       df_plot['HHTC_ag_contribution'],
       color = 'steelblue',
       label = 'HHTC annual growth contribution')

ax.bar(x,
       df_plot['NFCTC_ag_contribution'],
       bottom = df_plot['HHTC_ag_contribution'],
       color = 'orange',
       label = 'NFCTC annual growth contribution')

# Q1 positions only for xticks
q1_mask = df_plot.index.month == 3
q1_positions = np.where(q1_mask)[0]

ax.set_xticks(q1_positions)
ax.set_xticklabels(df_plot.index[q1_mask].year, rotation = 45)
ax.set_title('Annual growth of total credit')
ax.set_xlabel('year')
ax.set_ylabel('percentage points')
ax.legend()
ax.grid(True, alpha = 0.3)

plt.savefig('annual_TC_growth_rate_contributions.png')
plt.tight_layout()
plt.show()

# Regression: RTC_annual_growth = β0 + β1 * GDP_annual_growth + β2 * IR_change + ε
formula_RTC_growth_IRc = 'RTC_annual_growth ~ GDP_annual_growth + IR_change'
model_RTC_growth_IRc = smf.ols(formula_RTC_growth_IRc, data = dfC).fit()

print(model_RTC_growth_IRc.summary())

# Export regression summary to html (open with Excel)
export_summary(model_RTC_growth_IRc, "regression_summary_RTC_growth_IRc.html")

print("""
The regression results indicate that changes in the interest rate and GDP growth are significant factors in explaining the growth dynamics of real total credit.

The coefficient on GDP growth (0.8551) is significant at the 1% level (p-value = 0.000 < 0.010),
suggesting that a 1 percentage point increase in the annual growth rate of real GDP is associated with a 0.86 percentage point increase in the annual growth rate of real total credit.
This confirms that credit is procyclical.

The coefficient on interest rate change (0.1788) is significant at the 5% level (p-value = 0.043 < 0.050).
This implies that a 1 percentage point increase in the annual change of the nominal interest rate is associated with a 0.18 percentage point increase in the annual growth rate of real total credit.
This positive relationship might seem counterintuitive, but it could reflect that interest rates are often increased by central banks during periods of strong economic growth and rising inflation, which are also periods when credit demand is high.

The model has an R-squared of 0.393, indicating that approximately 39.3% of the variation in the annual growth rate of real total credit is explained by the annual growth rate of GDP and the annual change in the interest rate.
""")

result_RTC = adfuller(dfC['RTC_annual_growth'])
print("Annual RTC growth")
print("ADF statistic:", result_RTC[0])
print("p-value:", result_RTC[1])
print("Critical values:", result_RTC[4])
print("The Augmented Dickey-Fuller (ADF) test for the annual RTC growth rate series provides weak evidence of stationarity, as the null hypothesis of a unit root is rejected at the 10% significance level.")

result_GDP = adfuller(dfC['GDP_annual_growth'])
print("\nAnnual GDP growth")
print("ADF statistic:", result_GDP[0])
print("p-value:", result_GDP[1])
print("Critical values:", result_GDP[4])
print("The Augmented Dickey-Fuller (ADF) test for the annual real GDP growth rate series provides evidence of stationarity, as the null hypothesis of a unit root is rejected at the 5% significance level.")

result_IRc = adfuller(dfC['IR_change'])
print("\nAnnual interest rate change")
print("ADF statistic:", result_IRc[0])
print("p-value:", result_IRc[1])
print("Critical values:", result_IRc[4])
print("The Augmented Dickey-Fuller (ADF) test for the annual change in the interest rate series provides strong evidence of stationarity, as the null hypothesis of a unit root is rejected at the 1% significance level.")

print("\nThe chosen transformations for the independent variables address the potential non-stationarity of the original time series, which is a crucial step to avoid spurious regression.")

# Seasonality check
print("\nSeasonality check")

decomposition_RTC = seasonal_decompose(dfC['RTC_annual_growth'].dropna(), model = 'additive', period = 4)  # quarterly data
print("\nRTC growth seasonal component")
print(decomposition_RTC.seasonal)

decomposition_GDP = seasonal_decompose(dfC['GDP_annual_growth'].dropna(), model = 'additive', period = 4)  # quarterly data
print("\nGDP growth seasonal component")
print(decomposition_GDP.seasonal)

decomposition_IRc = seasonal_decompose(dfC['IR_change'].dropna(), model = 'additive', period = 4)  # quarterly data
print("\nIR change seasonal component")
print(decomposition_IRc.seasonal)

# Annual growth rate of total credit to non-financial private sector (TC) - more economically meaningful for credit data
tc_growth = dfC['TC_annual_growth']
y1 = tc_growth

results1 = {}

# MA model - TC annual growth rate
ma_model1 = ARIMA(y1, order=(0, 0, 1)).fit()
results1['MA(1)'] = ma_model1

print("\nMA(1) for TC annual growth rate estimated.\n")

# AR model - TC annual growth rate
ar_model1 = ARIMA(y1, order=(1, 0, 0)).fit()
results1['AR(1)'] = ar_model1

print("AR(1) for TC annual growth rate estimated.\n")

# ARIMA - TC annual growth rate
arima_model1 = ARIMA(y1, order=(1, 0, 1)).fit()
results1['ARIMA(1,0,1)'] = arima_model1

print("ARIMA(1,0,1) for TC annual growth rate estimated.\n")

# Model comparison
comparison = pd.DataFrame({
    "Model": results1.keys(),
    "AIC": [results1[m].aic for m in results1],
    "BIC": [results1[m].bic for m in results1]
})

print(comparison.sort_values(by = "AIC"))
best_model_aic = comparison.sort_values(by = "AIC").iloc[0]
print("\nBest model by AIC:", best_model_aic["Model"])

print(results1[best_model_aic["Model"]].summary())

# Export regression summary to html (open with Excel)
export_summary(results1[best_model_aic["Model"]], "regression_summary_TC_growth_AIC.html")

# First Difference of total credit to non-financial private sector (TC)
tc_diff = dfC['TC_FD']
y2 = tc_diff

results2 = {}

# MA model - TC First Difference
ma_model2 = ARIMA(y2, order=(0, 0, 1)).fit()
results2['MA(1)'] = ma_model2

print("\nMA(1) for TC First Difference estimated.\n")

# AR model - TC First Difference
ar_model2 = ARIMA(y2, order=(1, 0, 0)).fit()
results2['AR(1)'] = ar_model2

print("AR(1) for TC First Difference estimated.\n")

# ARIMA model - TC First Difference
arima_model2 = ARIMA(y2, order=(1, 0, 1)).fit()
results2['ARIMA(1,0,1)'] = arima_model2

print("ARIMA(1,0,1) for TC First Difference estimated.\n")

comparison = pd.DataFrame({
    "Model": list(results2.keys()),
    "AIC": [results2[m].aic for m in results2],
    "BIC": [results2[m].bic for m in results2]
})

print(comparison.sort_values(by = "AIC"))
best_model_aic = comparison.sort_values(by = "AIC").iloc[0]
print("\nBest model by AIC:", best_model_aic["Model"])

print(results2[best_model_aic["Model"]].summary(), "\n")

# Export regression summary to html (open with Excel)
export_summary(results2[best_model_aic["Model"]], "regression_summary_TC_FD_AIC.html")

################################################################################

dfD = pd.read_excel('data.xlsx', sheet_name = "data_D")

# Convert the 'Time' column to a proper datetime index for plotting
def date_conversion(time_str):
    year = int(time_str[:4])
    quarter = int(time_str[5])
    month = 3 * quarter
    return pd.to_datetime(f'{year}-{month}-01')

dfD['Date'] = dfD['Time'].apply(date_conversion)

# Set panel structure (country and date)
dfD = dfD.set_index(['Country_No', 'Date'])

# Reset index for plotting
df_reset = dfD.reset_index()

# Scatterplot that shows the relationship between unemployment rate and ROE
plt.figure(figsize = (14, 7))
plt.scatter(df_reset['UnemplRate'], df_reset['ROE'], alpha = 0.5, s = 20)
plt.title('ROE vs unemployment rate')
plt.xlabel('unemployment rate')
plt.ylabel('ROE')
plt.grid(True, alpha = 0.3)
plt.savefig('ROE_unemployment_rate.png')
plt.show()

# Correlation ROE vs unemployment rate
corr_unemp, p_unemp = stats.pearsonr(df_reset['ROE'], df_reset['UnemplRate'])

print("ROE vs Unemployment rate")
print("Correlation:", corr_unemp)
print("p-value:", p_unemp)

print("""
This indicates a moderate and statistically significant at the 5% level negative relationship between ROE and unemployment.
When unemployment rises, ROE tends to fall.
Higher unemployment reflects weaker aggregate demand, which compresses firm revenues and margins.
Lower demand reduces pricing power and efficiency, leading to weaker profitability.
Conversely, when unemployment is low, demand is stronger and firms tend to generate higher returns on equity.
This result suggests that ROE is pro-cyclical and closely tied to labor market conditions.
""")

# Scatterplot that shows the relationship between GDP growth and ROE
plt.figure(figsize = (14, 7))
plt.scatter(df_reset['GDPgrowth'], df_reset['ROE'], alpha = 0.5, s = 20)
plt.xlabel('GDP growth')
plt.ylabel('ROE')
plt.title('ROE vs GDP growth')
plt.grid(True, alpha = 0.3)
plt.savefig('ROE_GDP_growth.png')
plt.show()

# Correlation ROE vs GDP growth
corr_gdp, p_gdp = stats.pearsonr(df_reset['ROE'], df_reset['GDPgrowth'])

print("ROE vs GDP growth")
print("Correlation:", corr_gdp)
print("p-value:", p_gdp)

print("""
This shows a weak and statistically significant at the 10% level positive relationship between ROE and GDP growth.
The sign of the correlation coefficient is consistent with theory, since ROE should increase with GDP growth.
There are several reasons for the weakness of this relationship:
- GDP growth is noisy and volatile, especially at quarterly frequency, which weakens correlations.
- Timing mismatch: GDP growth reflects current or leading conditions, while ROE often reacts with a lag.
- Growth vs. level: profitability depends more on the level of economic slack than growth rates.
- Financial factors like leverage, interest rates, taxes, and buybacks affect ROE independently.

Unemployment appears to be a better proxy for the macro environment affecting profitability (ROE) because:
- it is smoother and less noisy than GDP growth;
- it captures labor market slack directly affecting demand and margins;
- it better reflects the cyclical position of the economy.
""")

# To improve the analysis, consider using lagged GDP growth, output gap measures, or unemployment gaps, which are more closely aligned with firm profitability.

# Panel regression with Fixed Effects estimator
y = dfD['NPLRatio']
X = dfD[['GDPgrowth', 'UnemplRate']]

# Add constant
X = sm.add_constant(X)

# Fixed Effects
model_NPL = PanelOLS(y, X, entity_effects = True, time_effects = True)
results_NPL = model_NPL.fit(cov_type = 'clustered', cluster_entity = True)

print(results_NPL.summary, "\n")

# Export regression summary to html (open with Excel)
export_summary(results_NPL, "regression_summary_NPL_panel.html")

def run_reg(data):
    y = data['NPLRatio']
    X = sm.add_constant(data[['GDPgrowth', 'UnemplRate']])
    model = sm.OLS(y, X).fit()
    return model

# Germany
df_DE = dfD.loc[6]
res_DE = run_reg(df_DE)

print("Germany\n", res_DE.summary())

# Export regression summary to html (open with Excel)
export_summary(res_DE, "regression_summary_NPL_DE.html")

# Denmark
df_DK = dfD.loc[7]
res_DK = run_reg(df_DK)

print("\nDenmark\n", res_DK.summary())

# Export regression summary to html (open with Excel)
export_summary(res_DK, "regression_summary_NPL_DK.html")

print("""
Germany has a larger absolute GDP growth coefficient (-0.3851), significant at the 10% level (p-value = 0.078) compared to the one for Denmark (-0.0997), which is not statistically significant (p-value = 0.544).
This indicates that NPL ratios in Germany are more sensitive to GDP shocks.
In a stress scenario, a negative GDP shock would increase NPLs in Germany.
Higher NPLs imply larger losses and a stronger deterioration in capital ratios.
Therefore, the same GDP shock would have a stronger impact on capital ratios in Germany than in Denmark.
""")

dfD['NPL_lag'] = dfD.groupby(level = 0)['NPLRatio'].shift(1)
df_lag = dfD.dropna()

# Panel regression with Fixed Effects estimator
y = df_lag['NPLRatio']
X = df_lag[['GDPgrowth', 'UnemplRate', 'NPL_lag']]

# Add constant
X = sm.add_constant(X)

# Fixed Effects
model_lag = PanelOLS(y, X, entity_effects = True, time_effects = True)
results_lag = model_lag.fit(cov_type = 'clustered', cluster_entity = True)

print(results_lag.summary, "\n")

# Export regression summary to html (open with Excel)
export_summary(results_lag, "regression_summary_NPL_lag_panel.html")

# Panel regression with Fixed Effects estimator
y = dfD['ROE']
X = dfD[['GDPgrowth', 'UnemplRate']]

# Add constant
X = sm.add_constant(X)

# Fixed Effects
mod = PanelOLS(y, X, entity_effects = True, time_effects = True)
res = mod.fit(cov_type = 'clustered', cluster_entity = True)

print(res.summary)

# Export regression summary to html (open with Excel)
export_summary(res, "regression_summary_ROE_unempl_panel.html")

dfD['GDP_lag'] = dfD.groupby(level = 0)['GDPgrowth'].shift(1)
dfD['Unemp_lag'] = dfD.groupby(level = 0)['UnemplRate'].shift(1)

df_lag2 = dfD.dropna()

# Panel regression with Fixed Effects estimator
y = df_lag2['ROE']
X = df_lag2[['GDPgrowth', 'UnemplRate', 'GDP_lag', 'Unemp_lag']]

# Add constant
X = sm.add_constant(X)

mod2 = PanelOLS(y, X, entity_effects = True, time_effects = True)
res2 = mod2.fit(cov_type = 'clustered', cluster_entity = True)

print(res2.summary)

# Export regression summary to html (open with Excel)
export_summary(res2, "regression_summary_ROE_unempl_lag_panel.html")

dfD['ROE_lag'] = dfD.groupby(level = 0)['ROE'].shift(1)
df_lag3 = dfD.dropna()

# Panel regression with Fixed Effects estimator
y = df_lag3['ROE']
X = df_lag3[['GDPgrowth', 'UnemplRate', 'ROE_lag']]

# Add constant
X = sm.add_constant(X)

mod3 = PanelOLS(y, X, entity_effects = True, time_effects = True)
res3 = mod3.fit(cov_type = 'clustered', cluster_entity = True)

print(res3.summary)

# Export regression summary to html (open with Excel)
export_summary(res3, "regression_summary_ROE_lag_panel.html")

################################################################################

# Creditor data
df_creditor = pd.read_excel("data.xlsx", sheet_name = "data_creditor")

# Exposure data
df_exposure = pd.read_excel("data.xlsx", sheet_name = "data_exposure")

merged_df = pd.merge(df_creditor, df_exposure, on = ["DT_RFRNC", "CRDTR_ID"], how = "left")

merged_table = merged_df[["DT_RFRNC", "CRDTR_NM", "CRDTR_CNTRY", "TOT_ASS", "TYP_INSTRMNT", "EXP_VALUE", "DBTR_NM", "DBTR_CNTRY"]].copy()

merged_table["DT_RFRNC"] = pd.to_datetime(merged_table["DT_RFRNC"].astype(str), format = "%Y%m")
merged_table['Year'] = merged_table['DT_RFRNC'].dt.year
merged_table['Month'] = merged_table['DT_RFRNC'].dt.month

# Convert to exposure value from EUR to EUR million and change its datatype to integer
merged_table["EXP_VALUE"] = (merged_table["EXP_VALUE"] / 1_000_000).astype(int)

# Add a domestic exposure variable which takes value 1 if creditor's country = debtor's country and 0 otherwise
merged_table["DOMESTIC_EXPOSURE"] = (merged_table["CRDTR_CNTRY"] == merged_table["DBTR_CNTRY"]).astype(int)

# Add a large exposure variable which takes value 1 if the exposure value > 10% of total assets and 0 otherwise
merged_table["LARGE_EXPOSURE"] = (merged_table["EXP_VALUE"] > 0.10 * merged_table["TOT_ASS"]).astype(int)

# Domestic exposure of Debtor1710 in June 2021
DOME = merged_table[
    (merged_table['Year'] == 2021) &
    (merged_table['Month'] == 6) &
    (merged_table["DBTR_NM"] == "Debtor1710") &
    (merged_table["DOMESTIC_EXPOSURE"] == 1)
]["EXP_VALUE"].sum()

print(f"\nDomestic exposure of Debtor1710 in June 2021: {DOME}")

# Credit instruments with an exposure value >= 1 EUR million in June 2020 granted by Lenders resident in Germany
CREI = merged_table[
    (merged_table['Year'] == 2020) &
    (merged_table['Month'] == 6) &
    (merged_table["CRDTR_CNTRY"] == "DE") &
    (merged_table["EXP_VALUE"] >= 1)
][["CRDTR_NM", "TYP_INSTRMNT", "EXP_VALUE"]].sort_values(by = "CRDTR_NM")

print("\nCredit instruments with an exposure value >= 1 EUR million in June 2020 granted by Lenders resident in Germany")
print(CREI)

# Aggregate exposure value grouped by country of residence of creditor for all reference dates
agg = (merged_table.groupby(["DT_RFRNC", "CRDTR_CNTRY"])["EXP_VALUE"].sum().reset_index())
pivot = agg.pivot(index = "DT_RFRNC", columns = "CRDTR_CNTRY", values = "EXP_VALUE")

# Chart to visualise the change in aggregate exposures by country of residence of creditor over time
pivot.plot(figsize = (14, 7))

plt.title("Aggregate exposure by creditor country over time")
plt.xlabel("reference date")
plt.ylabel("exposure (EUR millions)")
plt.grid(True, alpha = 0.3)
plt.legend(loc = 'upper left', bbox_to_anchor = (1, 1))
plt.tight_layout()
plt.savefig('change_agg_exposure_creditor_country.png')
plt.show()
