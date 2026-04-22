from google.colab import files
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os

if os.path.exists("data.xlsx"):
    print("'data.xlsx' is available.")
else:
    print("Please upload 'data.xlsx'")
    uploaded = files.upload()

# Financial statement data
df_financial = pd.read_excel('data.xlsx', sheet_name = "Financial_statement")

# Data on loans
df_loan = pd.read_excel ('data.xlsx', sheet_name = "Loan_details")

# Merge the data on loans with the financial statement data on c_id
# one observation for each loan granted

# df_financial key check
unique_c_id = df_financial['c_id'].is_unique

if unique_c_id:
    print("\nFinancial statement data keys are unique")
else:
    print("\nPlease note that financial statement data keys are not unique")

merged_df = pd.merge(df_loan, df_financial, on = 'c_id', how = 'left')

# Merged obs
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

# Remarks
print("\nThe minimum number of employees is -98, which is not a valid employee count.\n")

# Scatterplot that shows the relationship between the leverage ratio and the interest rate for loans granted by Lender 2
b2_df = merged_df[merged_df['b_id'] == 2]

plt.figure(figsize = (10, 6))
plt.scatter(b2_df['leverage'], b2_df['interest_rate'], alpha = 0.5, s = 20)
plt.title('Interest rate vs. leverage for Lender 2')
plt.xlabel('leverage ratio')
plt.ylabel('interest rate')
plt.grid(True)
plt.savefig('leverage_ratio_interest_rate.png')
plt.show()

print("\nThe scatterplot shows a positive relationship between the leverage ratio and the interest rate.\n"
      "This suggests that as a firm leverage increases, Lender 2 charges a higher interest rate, reflecting a higher perceived risk.\n")

# Regression: interest rate  = α + β_1*roa + β_2*total assets + β_3*public + ε
formula_ir = 'interest_rate ~ roa + total_assets + public'
model_ir = smf.ols(formula_ir, data = merged_df).fit()

print(model_ir.summary())

# Export regression summary to html (open with Excel)
with open("regression_summary_ir.html", "w") as f:
    f.write(model_ir.summary().as_html())
files.download("regression_summary_ir.html")

print("\nThe interest rate firms pay is significantly influenced by their profitability (p-value = 0.000 < 0.010) and size (p-value = 0.000 < 0.010).\n"
      "More profitable (higher ROA) and larger (higher total assets) firms get lower interest rates.\n"
      "There is no statistically significant relationship between the firm being publicly listed and the interest rate (p-value = 0.382).")

print("\nOmitted Variable Bias:\n"
      "the model is missing important variables that likely influence interest rates, such as the firm's leverage ratio and the term of the loan.\n"
      "Omitting these variables can lead to biased coefficient estimates.")

print("\nMulticollinearity:\n"
      "the high condition number in the regression output suggests there may be multicollinearity among the independent variables,\n"
      "which could make the coefficient estimates unstable and unreliable.")

df_gdp_debt = pd.read_excel('data.xlsx', sheet_name = "data_A")

df_gdp_debt.rename(columns = {df_gdp_debt.columns[0]: 'Date'}, inplace = True)
df_gdp_debt['Date'] = pd.to_datetime(df_gdp_debt['Date'])
df_gdp_debt.set_index('Date', inplace = True)

# debt-to-gdp ratio
df_gdp_debt['debt_to_GDP_ratio'] = (df_gdp_debt['g_debt'] / df_gdp_debt['GDP']) * 100

# debt-to-gdp ratio in December 2003
debt_2003 = df_gdp_debt.loc['2003-12-01', 'g_debt']
gdp_2003 = df_gdp_debt.loc['2003-12-01', 'GDP']
ratio_2003 = (debt_2003 / gdp_2003) * 100
print(f"\ndebt-to-GDP ratio in December 2003: {ratio_2003:.2f}%")

# debt-to-gdp ratio in December 2023
debt_2023 = df_gdp_debt.loc['2023-12-01', 'g_debt']
gdp_2023 = df_gdp_debt.loc['2023-12-01', 'GDP']
ratio_2023 = (debt_2023 / gdp_2023) * 100
print(f"debt-to-GDP ratio in December 2023: {ratio_2023:.2f}%")

# debt-to-gdp ratio plot
plt.figure(figsize = (12, 7))
plt.plot(df_gdp_debt.index, df_gdp_debt['debt_to_GDP_ratio'], marker = 'o', linestyle = '-', color = 'b')
plt.title('debt-to-GDP ratio over time', fontsize = 16)
plt.xlabel('date', fontsize = 12)
plt.ylabel('debt-to-GDP ratio (%)', fontsize = 12)
plt.grid(True)
plt.savefig('debt_to_gdp_ratio.png')
plt.tight_layout()

# Interest rate plot
plt.figure(figsize = (12, 7))
plt.plot(df_gdp_debt.index, df_gdp_debt['i_rate'], marker = 'o', linestyle = '-', color = 'r')
plt.title('Interest rate over time', fontsize = 16)
plt.xlabel('date', fontsize = 12)
plt.ylabel('interest rate (%)', fontsize = 12)
plt.grid(True)
plt.savefig('interest_rate.png')
plt.tight_layout()

print("\nThe debt-to-GDP ratio has been on a consistent upward trend since the beginning of the series,\n"
      "with sharp increases during major economic events like the 2008-2009 financial crisis and the COVID-19 pandemic.")

print("\nThe interest rate experienced a prolonged decline after 2009, reaching historic lows in the mid-2010s, but has since begun to increase sharply.\n")

df = pd.read_excel('data.xlsx', sheet_name = "data_B")

# loans-to-total assets ratio
df['loans_to_total_assets'] = df['loans'] / df['total_assets']

# operating expenses-to-total assets ratio
df['exp_operating_to_total_assets'] = df['exp_operating'] / df['total_assets']

# Pooled OLS regression.
# The standard errors are clustered by Lenders to account for the panel structure and heteroskedasticity (robustness)

# Regression - roe  = α + β_1*loans_to_total_assets + β_2*exp_operating_to_total_assets + β_3*cet1_ratio + β_4*gdp + β_5*interest_rate + ε
formula_roe = 'roe ~ loans_to_total_assets + exp_operating_to_total_assets + cet1_ratio + gdp + interest_rate'
model_roe = smf.ols(formula_roe, data = df).fit(cov_type = 'cluster', cov_kwds = {'groups': df['b_id']})

print(model_roe.summary())

# Export regression summary to html (open with Excel)
with open("regression_summary_roe.html", "w") as f:
    f.write(model_roe.summary().as_html())
files.download("regression_summary_roe.html")

print("\nBased on the regression results, the following variables have a statistically significant positive effect on the roe:")
print("- gdp --> The coefficient (0.3437) is positive and significant at a 1% level (p-value = 0.000 < 0.010).")
print("- interest_rate --> The coefficient (0.8466) is positive significant at a 1% level (p-value = 0.000 < 0.010).\n")

# Regression: roe  = α + β_1*loans_to_total_assets + β_2*exp_operating_to_total_assets + β_3*cet1_ratio + β_4*gdp + β_5*interest_rate + β_5*FE(Lender) + ε
formula_roe_FE = 'roe ~ loans_to_total_assets + exp_operating_to_total_assets + cet1_ratio + gdp + interest_rate + C(b_id)'
model_roe_FE = smf.ols(formula_roe_FE, data = df).fit(cov_type = 'cluster', cov_kwds = {'groups': df['b_id']})

print(model_roe_FE.summary())

# Export regression summary to html (open with Excel)
with open("regression_summary_roe_FE.html", "w") as f:
    f.write(model_roe_FE.summary().as_html())
files.download("regression_summary_roe_FE.html")

print("\nThe purpose of including fixed effects is to control for any unobserved, time-invariant heterogeneity that exists across different Lenders.\n"
      "This accounts for unique characteristics of each lender that do not change over the sample period, such as its location.\n"
      "By including a dummy variable for each Lender, the fixed effects model isolates the effect of changes within each Lender over time,\n"
      "rather than comparing differences between Lenders.\n"
      "This results in more reliable coefficient estimates for the other variables.\n")

# Regression: roe  = α + β_1*loans_to_total_assets + β_2*exp_operating_to_total_assets + β_3*cet1_ratio + β_4*gdp + β_5*interest_rate*FE(interest_rate_fixation) + ε
formula_roe_ir_fix = 'roe ~ loans_to_total_assets + exp_operating_to_total_assets + cet1_ratio + gdp + interest_rate * C(interest_rate_fixation)'
model_roe_ir_fix = smf.ols(formula_roe_ir_fix, data = df).fit(cov_type = 'cluster', cov_kwds = {'groups': df['b_id']})

print(model_roe_ir_fix.summary())

# Export regression summary to html (open with Excel)
with open("regression_summary_roe_ir_fix.html", "w") as f:
    f.write(model_roe_ir_fix.summary().as_html())
files.download("regression_summary_roe_ir_fix.html")

print("\nThe output uses floating-rate loans as the base category\n"
      "Therefore, for Lenders in fixed-rate countries, a 1 percentage point increase in the GDP is associated with a 0.3455 percentage point increase in the roe.\n"
      "This effect can be considered statistically significant (p-value = 0.000 < 0.010).")

df = pd.read_excel('data.xlsx', sheet_name = "data_C")

# Convert the 'Time' column to a proper datetime index for plotting
def convert_to_date(time_str):
    year = int(time_str[-4:])
    quarter = int(time_str[0])
    month = 3 * quarter
    return pd.to_datetime(f'{year}-{month}-01')

df['Date'] = df['Time'].apply(convert_to_date)
df = df.set_index('Date')

# The annual growth rate of a variable 'X' is calculated as: (Xt / Xt-4 - 1) * 100

# Annual growth rate of (real) GDP
df['GDP_annual_growth'] = (df['GDP'] / df['GDP'].shift(4) - 1) * 100

# Real total credit (RTC) is derived by dividing TC by PGDP
df['RTC'] = df['TC'] / df['PGDP']

# Annual growth rate of real total credit to non-financial private sector (RTC)
df['RTC_annual_growth'] = (df['RTC'] / df['RTC'].shift(4) - 1) * 100

plt.figure(figsize = (12, 7))
plt.plot(df.index, df['GDP_annual_growth'], label = 'annual GDP growth rate (%)')
plt.plot(df.index, df['RTC_annual_growth'], label = 'annual RTC growth rate (%)')
plt.title('Annual growth rates of real GDP and real total credit')
plt.xlabel('date')
plt.ylabel('annual growth rate (%)')
plt.legend()
plt.grid(True)
plt.savefig('annual_growth_rates.png')
plt.tight_layout()

print("\nThe chart shows that real total credit (RTC) appears to be procyclical, meaning it generally moves in tandem with the business cycle, as represented by real GDP growth.")

print("\nThe annual growth rate of RTC rises during periods of economic expansion (high GDP growth) and falls during economic contractions (low or negative GDP growth).\n"
      "For example, during recessions, such as the early 1980s, early 1990s, and the Great Recession of 2008-2009, both GDP growth and RTC growth experienced sharp declines.\n"
      "Conversely, during periods of strong economic performance, the growth rates of both variables tend to increase.\n")

# Annual change in the interest rate (independent variable)
df['IR_change'] = df['IR'] - df['IR'].shift(4)

# Drop NaN values created by the shifting operation
df_clean = df.dropna()

# Regression: RTC_annual_growth = β0 + β1 * GDP_annual_growth + β2 * IR_change + ε
formula_RTC_growth = 'RTC_annual_growth ~ GDP_annual_growth + IR_change'
model_RTC_growth = smf.ols(formula_RTC_growth, data = df_clean).fit()

print(model_RTC_growth.summary())

# Export regression summary to html (open with Excel)
with open("regression_summary_RTC_growth.html", "w") as f:
    f.write(model_RTC_growth.summary().as_html())
files.download("regression_summary_RTC_growth.html")

print("\nThe regression results indicate that changes in the interest rate and GDP growth are significant factors in explaining the growth dynamics of real total credit.")

print("\nThe coefficient on GDP growth (0.8551) is significant at the 1% level (p-value = 0.000 < 0.010),\n"
      "suggesting that a 1 percentage point increase in the annual growth rate of real GDP is associated with an 0.86 percentage point increase in the annual growth rate of real total credit.\n"
      "This confirms that credit is procyclical.")

print("\nThe coefficient on interest rate change (0.1788) is significant at the 5% level (P = 0.043 < 0.050).\n"
      "This implies that a 1 percentage point increase in the annual change of the nominal interest rate is associated with a 0.18 percentage point increase in the annual growth rate of real total credit.\n"
      "This positive relationship might seem counterintuitive, but it could reflect that interest rates are often increased by central banks during periods of strong economic growth and rising inflation, which are also periods when credit demand is high.")

print("\nThe model has an R-squared of 0.393, indicating that approximately 39.3% of the variation in the annual growth rate of real total credit is explained by the annual growth rate of GDP and the annual change in the interest rate.")

result_RTC = adfuller(df['RTC_annual_growth'].dropna())
print("\nAnnual RTC growth")
print("ADF statistic:", result_RTC[0])
print("p-value:", result_RTC[1])
print("Critical values:", result_RTC[4])
print("The Augmented Dickey-Fuller (ADF) test for the annual RTC growth rate series provides weak evidence of stationarity, as the null hypothesis of a unit root is rejected at the 10% significance level.")

result_GDP = adfuller(df['GDP_annual_growth'].dropna())
print("\nAnnual GDP growth")
print("ADF statistic:", result_GDP[0])
print("p-value:", result_GDP[1])
print("Critical values:", result_GDP[4])
print("The Augmented Dickey-Fuller (ADF) test for the annual real GDP growth rate series provides evidence of stationarity, as the null hypothesis of a unit root is rejected at the 5% significance level.")

result_IRc = adfuller(df['IR_change'].dropna())
print("\nAnnual interest rate change")
print("ADF statistic:", result_IRc[0])
print("p-value:", result_IRc[1])
print("Critical values:", result_IRc[4])
print("The Augmented Dickey-Fuller (ADF) test for the annual change in the interest rate series provides strong evidence of stationarity, as the null hypothesis of a unit root is rejected at the 1% significance level.")

print("\nThe chosen transformations for the independent variables address the potential non-stationarity of the original time series, which is a crucial step to avoid spurious regression.\n")
