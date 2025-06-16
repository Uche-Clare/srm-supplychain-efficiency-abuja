import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Manually extract the actual data from the survey responses
# Trust-based relationship variables (Section B)
trust_data = {
    'Trust_Supply_Based': [2, 4, 4, 1, 2, 5, 4, 5, 2, 1, 1, 2, 1, 1, 4, 4, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 3, 3, 2, 3, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 3, 2, 3],
    
    'Trust_Enhanced_Cooperation': [1, 2, 3, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 4, 5, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2],
    
    'Trust_Info_Exchange': [2, 3, 4, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 3, 4, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 3, 1, 3, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3],
    
    'Trust_Financial_Benefits': [1, 2, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 4, 2, 2, 2, 4, 5, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 3, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2],
    
    'Trust_Intentional_Development': [2, 2, 3, 1, 1, 2, 1, 3, 2, 2, 1, 2, 2, 3, 1, 2, 3, 4, 1, 3, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 3, 2, 1, 2, 2, 2, 3, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 3, 3],
    
    'Trust_Costly_Process': [4, 2, 1, 1, 2, 4, 5, 3, 4, 2, 2, 3, 5, 4, 5, 3, 5, 3, 4, 3, 2, 1, 5, 1, 4, 2, 5, 2, 5, 5, 5, 5, 2, 2, 5, 5, 5, 5, 3, 3, 3, 1, 1, 2, 5, 4, 2, 2, 2, 4, 5, 4, 5, 2, 5, 5, 2, 3]
}

# Supply Chain Efficiency variables (Section E - converted H=3, M=2, L=1)
efficiency_data = {
    'Cost_Efficiency': [2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 3, 2, 2, 1, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
    
    'Quality_Level': [3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3],
    
    'Lead_Time': [2, 2, 2, 2, 3, 2, 3, 1, 2, 3, 3, 2, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 2, 2],
    
    'Supplier_Reliability': [3, 2, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 2, 2, 3, 2, 3, 1, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2],
    
    'Production_Flexibility': [3, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 2, 3, 3, 3, 2, 2, 3, 3, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 2, 3, 3, 3],
    
    'Operational_Performance': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
}

# Company information
companies = ['Dangata Industries Ltd'] * 8 + ['Zeenab Food Ltd.'] * 19 + ['Urban Roots Plastic'] * 7 + ['Halibiz Industries Ltd'] * 10 + ['Qualitrends Concrete, formworks and scaffolding'] * 14

# Create comprehensive dataset
all_data = {**trust_data, **efficiency_data, 'Company': companies}

# Create DataFrame
df = pd.DataFrame(all_data)

print("=== TRUST-BASED RELATIONSHIPS IMPACT ON SUPPLY CHAIN EFFICIENCY ===")
print("Regression Analysis of Manufacturing Firms in Abuja")
print(f"Sample Size: {len(df)} responses from {len(df['Company'].unique())} companies\n")

# Display basic information
print("SAMPLE COMPOSITION:")
print("=" * 50)
company_counts = df['Company'].value_counts()
for company, count in company_counts.items():
    print(f"{company}: {count} responses")

print("\n1. DESCRIPTIVE STATISTICS")
print("=" * 50)

# Trust variables descriptive statistics
trust_vars = ['Trust_Supply_Based', 'Trust_Enhanced_Cooperation', 'Trust_Info_Exchange', 
              'Trust_Financial_Benefits', 'Trust_Intentional_Development', 'Trust_Costly_Process']

print("\nTrust-Based Relationship Variables (1=Strongly Agree to 5=Strongly Disagree):")
trust_desc = df[trust_vars].describe()
print(trust_desc.round(3))

# Efficiency variables descriptive statistics  
efficiency_vars = ['Cost_Efficiency', 'Quality_Level', 'Lead_Time', 
                   'Supplier_Reliability', 'Production_Flexibility', 'Operational_Performance']

print("\nSupply Chain Efficiency Variables (1=Low, 2=Medium, 3=High):")
efficiency_desc = df[efficiency_vars].describe()
print(efficiency_desc.round(3))
print(df.columns)  # Line 4: Shows all column names

# Create composite scores
print("\n2. COMPOSITE SCORE CREATION")
print("=" * 50)

# For trust variables, reverse code the "costly process" item since it's negatively worded
df['Trust_Costly_Process_Rev'] = 6 - df['Trust_Costly_Process']

# Create trust composite (lower scores = higher trust since most items are reverse coded)
df['Trust_Composite'] = (6 - df['Trust_Supply_Based'] + 
                        6 - df['Trust_Enhanced_Cooperation'] + 
                        6 - df['Trust_Info_Exchange'] + 
                        6 - df['Trust_Financial_Benefits'] + 
                        6 - df['Trust_Intentional_Development'] + 
                        df['Trust_Costly_Process_Rev']) / 6

# Create efficiency composite score
df['Efficiency_Composite'] = df[efficiency_vars].mean(axis=1)

print(f"Trust Composite Score - Mean: {df['Trust_Composite'].mean():.3f}, Std: {df['Trust_Composite'].std():.3f}")
print(f"   Range: {df['Trust_Composite'].min():.3f} to {df['Trust_Composite'].max():.3f}")
print(f"Efficiency Composite Score - Mean: {df['Efficiency_Composite'].mean():.3f}, Std: {df['Efficiency_Composite'].std():.3f}")
print(f"   Range: {df['Efficiency_Composite'].min():.3f} to {df['Efficiency_Composite'].max():.3f}")

# Correlation Analysis
print("\n3. CORRELATION ANALYSIS")
print("=" * 50)

# Calculate correlation between trust composite and efficiency composite
overall_correlation = df['Trust_Composite'].corr(df['Efficiency_Composite'])
print(f"Overall Correlation (Trust vs Efficiency): r = {overall_correlation:.4f}")

# Individual trust-efficiency correlations
print("\nCorrelations between individual Trust variables and Efficiency Composite:")
for trust_var in trust_vars:
    # Reverse code trust variables for interpretation (higher = more trust)
    trust_reversed = 6 - df[trust_var] if trust_var != 'Trust_Costly_Process' else df[trust_var]
    corr = trust_reversed.corr(df['Efficiency_Composite'])
    print(f"{trust_var.replace('_', ' ')}: r = {corr:.4f}")

print("\nCorrelations between Trust Composite and individual Efficiency variables:")
for eff_var in efficiency_vars:
    corr = df['Trust_Composite'].corr(df[eff_var])
    print(f"{eff_var.replace('_', ' ')}: r = {corr:.4f}")

# Multiple Regression Analysis
print("\n4. MULTIPLE REGRESSION ANALYSIS")
print("=" * 50)

# Prepare predictor variables (reverse coded for proper interpretation)
X_vars = []
X_names = []
for trust_var in trust_vars:
    if trust_var != 'Trust_Costly_Process':
        X_vars.append(6 - df[trust_var])  # Reverse code
        X_names.append(trust_var.replace('_', ' ').replace('Trust ', ''))
    else:
        X_vars.append(df[trust_var])  # Keep as is (already negatively worded)
        X_names.append('Trust Not Costly Process')

X = np.column_stack(X_vars)
y = df['Efficiency_Composite'].values

# Fit multiple regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate R-squared and adjusted R-squared
r2 = r2_score(y, y_pred)
n = len(y)
k = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

print("MULTIPLE REGRESSION RESULTS:")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
print(f"Standard Error: {np.sqrt(np.mean((y - y_pred) ** 2)):.4f}")

print("\nRegression Coefficients:")
print(f"{'Variable':<25} {'Coefficient':<12} {'Std Error':<12} {'t-value':<10} {'p-value':<10} {'Sig':<5}")
print("-" * 75)

# Calculate standard errors and t-statistics
residuals = y - y_pred
mse = np.sum(residuals ** 2) / (n - k - 1)
X_with_intercept = np.column_stack([np.ones(n), X])
cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
se_coefs = np.sqrt(np.diag(cov_matrix))

# Intercept
t_stat_intercept = model.intercept_ / se_coefs[0]
p_val_intercept = 2 * (1 - stats.t.cdf(abs(t_stat_intercept), n - k - 1))
sig_intercept = "***" if p_val_intercept < 0.001 else "**" if p_val_intercept < 0.01 else "*" if p_val_intercept < 0.05 else ""
print(f"{'Intercept':<25} {model.intercept_:<12.4f} {se_coefs[0]:<12.4f} {t_stat_intercept:<10.3f} {p_val_intercept:<10.4f} {sig_intercept:<5}")

# Coefficients
for i, (name, coef) in enumerate(zip(X_names, model.coef_)):
    se = se_coefs[i + 1]
    t_stat = coef / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{name:<25} {coef:<12.4f} {se:<12.4f} {t_stat:<10.3f} {p_val:<10.4f} {sig:<5}")

print("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05")

# Simple Regression (Trust Composite vs Efficiency Composite)
print("\n5. SIMPLE REGRESSION ANALYSIS")
print("=" * 50)

X_simple = df[['Trust_Composite']]
y_simple = df['Efficiency_Composite']

model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)
y_pred_simple = model_simple.predict(X_simple)
r2_simple = r2_score(y_simple, y_pred_simple)

print("SIMPLE REGRESSION RESULTS (Trust Composite → Efficiency Composite):")
print(f"R-squared: {r2_simple:.4f}")
print(f"Correlation coefficient: {np.sqrt(r2_simple) * np.sign(model_simple.coef_[0]):.4f}")

# Calculate significance for simple regression
residuals_simple = y_simple - y_pred_simple.flatten()
se_regression = np.sqrt(np.sum(residuals_simple ** 2) / (len(y_simple) - 2))
# Fix for DataFrame with one column
mean_val = X_simple.values.flatten().mean()
se_slope = se_regression / np.sqrt(np.sum((X_simple.values.flatten() - mean_val) ** 2))
t_stat_simple = model_simple.coef_[0] / se_slope
p_val_simple = 2 * (1 - stats.t.cdf(abs(t_stat_simple), len(y_simple) - 2))

print(f"Regression equation: Efficiency = {model_simple.intercept_:.4f} + {model_simple.coef_[0]:.4f} × Trust")
print(f"Slope significance: t = {t_stat_simple:.3f}, p = {p_val_simple:.4f}")

sig_simple = "***" if p_val_simple < 0.001 else "**" if p_val_simple < 0.01 else "*" if p_val_simple < 0.05 else "ns"
print(f"Significance level: {sig_simple}")

# Analysis by Company
print("\n6. ANALYSIS BY COMPANY")
print("=" * 50)

company_stats = df.groupby('Company').agg({
    'Trust_Composite': ['mean', 'std', 'count'],
    'Efficiency_Composite': ['mean', 'std']
}).round(3)

print("Trust and Efficiency by Company:")
print(company_stats)

# Effect Size and Practical Significance
print("\n7. EFFECT SIZE AND PRACTICAL SIGNIFICANCE")
print("=" * 50)

# Cohen's conventions for correlation effect sizes
abs_corr = abs(overall_correlation)
if abs_corr < 0.1:
    effect_size = "negligible"
elif abs_corr < 0.3:
    effect_size = "small"
elif abs_corr < 0.5:
    effect_size = "medium"
else:
    effect_size = "large"

print(f"Effect size (Cohen's conventions): {effect_size}")
print(f"Variance explained: {r2_simple * 100:.1f}% of efficiency variance explained by trust")
print(f"Cohen's f²: {r2_simple / (1 - r2_simple):.4f}")

# Practical interpretation
trust_range = df['Trust_Composite'].max() - df['Trust_Composite'].min()
efficiency_change = model_simple.coef_[0] * trust_range
print(f"Practical significance: Moving from lowest to highest trust level")
print(f"corresponds to a {efficiency_change:.3f} unit change in efficiency")

# Final Answer to Research Question
print("\n" + "=" * 80)
print("ANSWER TO RESEARCH QUESTION:")
print("To what extent do trust-based relationships impact supply chain efficiency")
print("of manufacturing firms in Abuja?")
print("=" * 80)

significance_text = "statistically significant" if p_val_simple < 0.05 else "not statistically significant"
direction = "positive" if overall_correlation > 0 else "negative"

print(f"""
KEY FINDINGS:

1. RELATIONSHIP STRENGTH: 
   - Correlation coefficient: r = {overall_correlation:.4f}
   - This represents a {effect_size} {direction} relationship

2. STATISTICAL SIGNIFICANCE:
   - The relationship is {significance_text} (p = {p_val_simple:.4f})
   - {'Reject' if p_val_simple < 0.05 else 'Fail to reject'} the null hypothesis of no relationship

3. PRACTICAL SIGNIFICANCE:
   - Trust explains {r2_simple * 100:.1f}% of the variance in supply chain efficiency
   - Effect size (Cohen's f²): {r2_simple / (1 - r2_simple):.4f}

4. REGRESSION EQUATION:
   - Efficiency = {model_simple.intercept_:.4f} + {model_simple.coef_[0]:.4f} × Trust
   - For every 1-unit increase in trust, efficiency increases by {model_simple.coef_[0]:.4f} units

5. INDIVIDUAL TRUST DIMENSIONS:
   - Multiple regression R² = {r2:.4f} (explains {r2 * 100:.1f}% of variance)
   - Different trust dimensions show varying levels of impact

CONCLUSION:
Trust-based relationships have a {effect_size} {direction} impact on supply chain efficiency 
in manufacturing firms in Abuja. The relationship is {significance_text}, indicating that 
organizations with stronger trust-based supplier relationships tend to achieve 
{'better' if overall_correlation > 0 else 'poorer'} supply chain efficiency outcomes.

PRACTICAL IMPLICATIONS:
- Trust-building should be considered a strategic priority for supply chain management
- Investment in supplier relationship management can yield measurable efficiency gains
- The relationship, while {significance_text}, explains a {'substantial' if r2_simple > 0.25 else 'moderate' if r2_simple > 0.09 else 'small'} portion of efficiency variance
""")

print(f"\nRECOMMENDations for Manufacturing Firms:")
print("1. Develop systematic trust-building programs with key suppliers")
print("2. Invest in long-term collaborative relationships")
print("3. Establish transparent communication and information-sharing mechanisms")
print("4. Monitor both trust levels and efficiency metrics regularly")
print("5. Train procurement and supply chain staff on relationship management")