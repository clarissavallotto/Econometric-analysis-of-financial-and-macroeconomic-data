# Econometric Analysis of Financial and Macroeconomic Data

## Overview

This project performs a comprehensive empirical analysis of fictitious financial and macroeconomic data using Python. It combines **financial statement data, loan-level data, lending panel datasets, and macroeconomic time series** to study relationships between firm characteristics, lending behavior, credit cycles, and macroeconomic indicators.

The analysis includes:

* Data merging and cleaning
* Construction of financial and macroeconomic variables
* Descriptive statistics
* Data visualization
* Cross-sectional and panel regressions (OLS and Fixed Effects)
* Time-series analysis (growth rates, ARIMA models, stationarity tests)
* Credit exposure aggregation and lending risk analysis
* Economic interpretation of results

The project is designed for use in a [**Google Colab**](https://colab.research.google.com/) environment.

---

## Objective

To examine how firm characteristics and macroeconomic conditions influence:
- **loan pricing**   
- **lender profitability**  
- **credit growth** 
- **credit risk (NPLs)**

---

## Input Data

The project relies on a single Excel file:

```
data.xlsx
```

It contains multiple sheets:

| Sheet Name            | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `Financial_statement` | Firm-level financial data (leverage, ROA, employees, etc.) |
| `Loan_details`        | Loan-level data (interest rates, loan amounts, lender IDs) |
| `data_A`              | Macroeconomic data (GDP, debt, interest rates)             |
| `data_B`              | Lender-level panel data                                      |
| `data_C`              | Quarterly macro-financial time series                      |
| `data_D`               | Cross-country panel data |
| `data_creditor`        | Creditor-level exposure data |
| `data_exposure`        | Exposure-level lending data |

---

## Dependencies

The following Python libraries are required:

```python
pandas
numpy
matplotlib
statsmodels
scipy
linearmodels
openpyxl (for Excel file operations)
google.colab (for Colab file handling)
```

---

## Workflow

### 1. Data Loading & Upload (Google Colab)

* Checks whether `data.xlsx` exists
* If not, prompts the user to upload it
* Loads multiple sheets into pandas DataFrames

### 2. Data Merging & Cleaning

* Merges loan and financial statement data using `c_id`
* Checks uniqueness of financial statement keys
* Left join ensures all loan observations are retained

```python
merged_df = pd.merge(df_loan, df_financial, on = 'c_id', how = 'left')
```

### 3. Variable Construction

New variables are created, including:

* **Interest expense**
* **Debt-to-GDP ratio**
* **Loans-to-assets ratio**
* **Operating expense ratio**
* **Real Total Credit (RTC)**
* **Annual growth rates (GDP, credit)**
* **Credit composition shares (households vs firms)**  
* **Exposure indicators (domestic, large exposure flags)**

### 4. Descriptive Statistics

Computes

* mean
* median
* 25th and 75th percentiles
* minimum and maximum

for

* leverage
* employees
* loan amount
* ROA

Output:

* printed in markdown format
* exported to Excel (`descriptive_statistics.xlsx`)

## 5. Regression Analysis

### Interest Rate Determinants

```text
interest_rate ~ ROA + total_assets + public
```

Findings:
* ROA → negative and significant effect  
* firm size → negative and significant effect  
* public listing → not significant  

### Lender Profitability

```text
ROE ~ loans_to_total_assets + exp_operating_to_total_assets + cet1_ratio + gdp + interest_rate
```

Clustered standard errors (by lender ID)

Findings:
* GDP → positive effect on ROE  
* interest rates → positive effect on ROE  

### Fixed Effects Model

```text
ROE ~ ... + C(b_id)
```

Purpose:
* controls for lender-specific heterogeneity  

Insight:
* macro variables remain significant within lenders  

### Interaction Effects

```text
ROE ~ ... + interest_rate * C(interest_rate_fixation)
```

Insight: 
* fixed vs floating-rate environments differ in sensitivity to macro shocks  

### Credit Growth

```text
RTC_annual_growth ~ GDP_annual_growth + IR_change
```
Findings:
* GDP growth → strong positive effect (procyclicality of credit)
* interest rate changes → positive and significant effect

## 6. Macroeconomic Time-Series Analysis

Constructed variables:

* GDP annual growth  
* RTC annual growth  
* interest rate change  

## 7. Stationarity Tests (ADF)

* GDP growth → stationary  
* RTC growth → weakly stationary  
* interest rate change → strongly stationary  

## 8. Time-Series Models

Applied models:

* AR(1)  
* MA(1)  
* ARIMA(1,0,1)  

Selection criteria:
* Akaike Information Criterion (AIC)  
* Bayesian Information Criterion (BIC)  

## 9. Cross-Country Panel Analysis 

## Cross-Country Panel Analysis

### Baseline Fixed Effects Model (NPL)

```text
NPLRatio ~ GDPgrowth + UnemplRate
```

* country and time fixed effects included
* clustered standard errors at country level

### Dynamic Panel Model (NPL Persistence)

```text
NPLRatio ~ GDPgrowth + UnemplRate + NPL_lag
```

Purpose:
* captures persistence in non-performing loans
* allows for dynamic adjustment of credit risk

### Country-Specific Regressions

Estimated using OLS on country-level subsamples (Germany, Denmark):

```text
NPLRatio ~ GDPgrowth + UnemplRate
```
Insight:
* sensitivity of NPLs to macroeconomic conditions differs across countries

### ROE and Macroeconomic Conditions (Panel FE)

```text
ROE ~ GDPgrowth + UnemplRate
```

Findings:
* unemployment → negative effect on profitability
* GDP growth → weak positive effect

### ROE with Lagged Macroeconomic Variables

```text
ROE ~ GDPgrowth + UnemplRate + GDP_lag + Unemp_lag
```

Purpose:
* captures delayed effects of macroeconomic conditions on profitability

### Dynamic ROE Model (Persistence)

```text
ROE ~ GDPgrowth + UnemplRate + ROE_lag
```

Purpose:
* accounts for persistence in profitability over time

## 10. Credit Exposure Analysis

Key features:

* domestic vs cross-border exposures  
* large exposure (>10% assets)  
* aggregation by creditor country  
* time evolution of exposures  


## Key Economic Insights

* risk-based pricing dominates lending  
* firm size and profitability reduce borrowing costs  
* credit appears procyclical in the sample
* macroeconomic conditions drive lending performance  
* unemployment is a strong predictor of credit risk 
* NPL ratios exhibit persistence over time 
* lending systems are heterogeneous across countries

---

## Outputs

The script generates:

### Excel Files
* `descriptive_statistics.xlsx`  

### HTML Regression Outputs (HTML)
* `regression_summary_IR.html`
* `regression_summary_ROE.html`
* `regression_summary_ROE_FE.html`
* `regression_summary_ROE_IR_fix.html`
* `regression_summary_RTC_growth_IRc.html`
* `regression_summary_NPL_panel.html`
* `regression_summary_NPL_DE.html`
* `regression_summary_NPL_DK.html`
* `regression_summary_NPL_lag_panel.html`
* `regression_summary_ROE_unempl_panel.html`
* `regression_summary_ROE_unempl_lag_panel.html`
* `regression_summary_ROE_lag_panel.html`
* `regression_summary_TC_growth_AIC.html`
* `regression_summary_TC_FD_AIC.html`

### Figures
* `leverage_ratio_interest_rate.png`  
* `debt_to_gdp_ratio.png`  
* `interest_rate.png`  
* `annual_growth_rates.png`
* `annual_TC_growth_rate_contributions.png`  
* `ROE_unemployment_rate.png`  
* `ROE_GDP_growth.png`  
* `change_agg_exposure_creditor_country.png`  

Inside `plots` folder:
* `GDP_timeseries.png`
* `PGDP_timeseries.png`
* `TC_timeseries.png`
* `HHTC_timeseries.png`
* `NFCTC_timeseries.png`
* `IR_timeseries.png`

---

## How to Run

1. Open [**Google Colab**](https://colab.research.google.com/)  
2. Upload the script 
3. Upload `data.xlsx` when prompted  
4. Run the code

---

## Limitations

- data is fictitious   
- potential issues: omitted variables, endogeneity, measurement error  
- project designed for small-to-medium datasets
