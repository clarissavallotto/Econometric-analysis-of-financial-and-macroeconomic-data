# Econometric Analysis of Financial and Macroeconomic Data

## Overview

This project performs a comprehensive empirical analysis of fictitious financial and macroeconomic data using Python. It combines **financial statement data, loan-level data, and macroeconomic time series** to study relationships between firm characteristics, lending behavior, credit cycles, and macroeconomic indicators.

The analysis includes:

* Data merging and cleaning
* Descriptive statistics
* Data visualization
* Cross-sectional and panel regressions (OLS with and without Fixed Effects)
* Time-series analysis (growth rates and stationarity tests)
* Economic interpretation of results

The project is designed for use in a [**Google Colab**](https://colab.research.google.com/) environment.

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

---

## Dependencies

The following Python libraries are required:

```python
pandas
matplotlib
statsmodels
google.colab (for Colab file handling)
openpyxl (for Excel file operations)
```

---

## Workflow

### 1. Data Loading & Upload (Google Colab)

* Checks whether `data.xlsx` exists
* If not, prompts the user to upload it
* Loads multiple sheets into pandas DataFrames

---

### 2. Data Merging

* Merges:

  * Loan data (`df_loan`)
  * Financial statements (`df_financial`)
* Join key: `c_id` (firm ID)
* Left join ensures all loan observations are retained

```python
merged_df = pd.merge(df_loan, df_financial, on = 'c_id', how = 'left')
```

---

### 3. Variable Construction

New variables are created, including:

* **Interest expense**
* **Debt-to-GDP ratio**
* **Loans-to-assets ratio**
* **Operating expense ratios**
* **Annual growth rates (GDP, credit)**

---

### 4. Descriptive Statistics

Computes:

* Mean
* Median
* 25th and 75th percentiles
* Minimum and maximum

For:

* Leverage
* Employees
* Loan amount
* ROA

Output is:

* Printed in markdown format
* Exported to Excel (`descriptive_statistics.xlsx`)

---

### 5. Data Visualization

#### Scatter Plot

Relationship between:

* Leverage ratio
* Interest rate (Lender 2 only)

Saved as:

`leverage_ratio_interest_rate.png`

#### Time Series Plots

* Debt-to-GDP ratio over time
* Interest rate over time
* Annual GDP and credit growth

Saved as:

* `debt_to_gdp_ratio.png`
* `interest_rate.png`
* `annual_growth_rates.png`

---

### 6. Regression Analysis

#### Regression 1: Interest Rate Determinants

```text
interest_rate ~ roa + total_assets + public
```

Findings:

* ROA and firm size reduce interest rates
* Public listing not statistically significant

---

#### Regression 2: Lender Profitability (ROE)

```text
roe ~ loans_to_total_assets + exp_operating_to_total_assets + cet1_ratio + gdp + interest_rate
```

Includes:

* Clustered standard errors (by lender ID)

Key insights:

* GDP and interest rates positively affect ROE

---

#### Regression 3: Lender Profitability (ROE) with Fixed Effects

```text
roe ~ ... + C(b_id)
```

Includes:

* Clustered standard errors (by lender ID)
* Fixed effects (`C(b_id)`)

Key insights:

* GDP and interest rates positively affect ROE
* Fixed effects control for lender-specific heterogeneity

---

#### Regression 4: Lender Profitability (ROE) with Interaction Effects

```text
roe ~ ... + interest_rate * C(interest_rate_fixation)
```

Includes:

* Clustered standard errors (by lender ID)
* Interaction effects (`interest_rate * C(interest_rate_fixation)`)

Key insights:

* GDP and interest rates positively affect ROE
* Interaction effects capture differences between fixed-rate and floating-rate environments

---

### 7. Macroeconomic Time Series Analysis

#### Variables constructed:

* Annual real GDP growth rate
* Annual real total credit to non-financial private sector (RTC) growth rate
* Annual change in the interest rate

#### Key insight:

* Credit is **procyclical** (moves with GDP)

---

### 8. Stationarity Testing (ADF Test)

Augmented Dickey-Fuller tests applied to:

* Annual real GDP growth rate (independent variable): stationary 
* RTC growth rate (dependent variable): weakly stationary
* Annual change in the interest rate (independent variable): strongly stationary

This ensures regression validity and avoids **spurious regression problems**.

---

## Key Economic Insights

1. **Risk-based pricing**

   * Higher leverage → higher interest rates

2. **Firm fundamentals matter**

   * Larger and more profitable firms borrow more cheaply

3. **Macroeconomic conditions matter**

   * GDP growth strongly affects lender profitability and credit expansion

4. **Credit cycles are procyclical**

   * Credit expands during economic booms and contracts during recessions

5. **Interest rates reflect macro conditions**

   * Often rise during strong economic periods

---

## Outputs

The script generates:

### Excel Files

* `descriptive_statistics.xlsx`

### HTML Regression Outputs

* `regression_summary_ir.html`
* `regression_summary_roe.html`
* `regression_summary_roe_FE.html`
* `regression_summary_roe_ir_fix.html`
* `regression_summary_RTC_growth.html`

### Figures

* `leverage_ratio_interest_rate.png`
* `debt_to_gdp_ratio.png`
* `interest_rate.png`
* `annual_growth_rates.png`
