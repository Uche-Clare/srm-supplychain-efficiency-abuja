import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Create the dataset from the survey responses
data = {
    'ID': range(1, 59),
    'Company': ['Dangata Industries Ltd']*8 + ['Zeenab Food Ltd.']*19 + ['Urban Roots Plastic']*7 + 
              ['Halibiz Industries Ltd']*10 + ['Qualitrends Concrete, formworks and scaffolding']*14,
    
    # Information Sharing Variables (Section C, Question 7)
    'Frequent_Info_Share': [2,4,4,1,2,5,4,5,2,1,1,2,1,1,4,1,4,4,1,1,1,1,2,1,2,2,1,2,2,2,1,2,2,1,1,2,1,3,3,2,3,2,1,1,1,1,1,2,1,1,2],
    'Effective_Supplier_Buyer_Structure': [1,2,3,2,2,2,1,2,2,1,1,1,1,2,1,2,4,5,1,1,1,1,1,1,1,2,1,2,1,1,2,1,2,1,1,2,1,2,2,1,2,3,2,2,1,2,2,2,2,2,2],
    'Easy_Access_Info': [2,3,4,2,2,2,1,1,2,2,2,1,2,2,2,2,2,4,1,1,2,1,2,1,1,3,2,1,1,2,1,1,2,1,1,2,1,1,1,1,3,2,2,1,2,2,2,2,1,2,3],
    'Right_Info_Performance': [1,2,2,2,2,3,2,3,3,2,1,2,4,2,2,2,4,5,2,1,1,1,2,1,2,2,2,1,1,1,2,2,2,1,1,2,1,2,2,1,1,2,2,2,1,2,2,2,2,1,2],
    'Data_Connectivity': [2,2,3,1,1,2,1,3,2,2,1,1,2,3,1,1,3,4,1,1,1,1,1,1,2,2,3,1,2,1,2,2,2,1,1,2,1,3,2,1,1,2,2,1,1,1,1,2,1,2,3],
    
    # Information Types Shared (converted to numeric: Yes=1, No=0)
    'Inventory_Levels': [0,0,0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,0,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,1,0,1,1,0,1],
    'Sales_Forecasts': [1,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,0,1,1,1,0,1,1,0,1,1],
    'Promotion_Strategies': [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,1,1,0,0,0,1,0,0,0,1,0,1,1,0,1,0,0,1,1,0,0],
    'Marketing_Plans': [0,1,0,0,1,0,0,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,1,1,0,1,0,1,1,0,0,1,0,0],
    'General_Feedback': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    
    # Supply Chain Efficiency Variables (Section E, Questions 10 & 11)
    'Effective_Management_Performance': [1,2,1,2,1,1,1,1,1,1,2,1,1,1,1,2,3,5,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,2,2,2,1,2,1,1,1,2,3,2,2,1,1,1,2],
    'Internal_Environment_Influence': [2,1,1,2,1,1,2,4,2,1,2,2,2,1,2,3,4,4,1,1,3,1,2,1,2,1,3,2,1,1,2,2,2,1,1,2,1,3,2,2,2,1,2,2,2,2,2,2,2,3],
    'Effective_Environment_Interaction': [2,2,2,2,1,1,2,2,2,2,2,1,1,2,1,4,3,4,1,1,2,1,1,1,2,2,2,1,2,1,2,2,2,1,2,2,2,1,3,1,2,2,1,2,2,2,2,1,2,2],
    'Reliable_Supply_Systems': [2,3,1,2,1,1,1,2,2,2,2,2,1,1,2,4,4,4,1,1,1,1,1,1,1,1,2,2,1,1,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,1,1,2,3],
    
    # Supply Chain Efficiency Ratings (converted to numeric: H=3, M=2, L=1)
    'Cost_Efficiency': [2,1,2,2,2,2,3,1,3,3,3,2,3,2,2,1,3,2,3,2,2,2,2,3,2,2,3,2,2,2,2,3,3,3,3,3,3,3,3,3,3,2,3,3,3,3,1,1,1,2],
    'Quality_Level': [3,3,3,3,3,3,2,3,3,3,3,3,3,3,3,2,3,3,3,3,3,3,3,3,3,2,3,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,1,3],
    'Lead_Time': [2,2,2,2,3,2,3,1,2,3,3,2,2,2,3,3,2,2,3,3,2,2,3,3,2,2,2,2,2,2,2,2,1,2,3,3,2,2,2,2,2,3,3,3,2,2,3,3,3,2],
    'Supplier_Reliability': [3,2,2,3,3,3,3,2,3,3,3,3,3,2,3,3,3,3,3,3,3,3,3,3,2,3,3,1,2,3,3,2,3,2,2,3,3,3,2,2,2,3,3,3,3,2,3,3,3,2],
    'Production_Flexibility': [3,2,3,3,3,3,2,2,2,3,3,2,3,3,3,3,2,3,2,3,3,2,3,3,3,2,1,2,3,3,3,2,2,3,2,3,3,2,2,2,3,3,3,3,3,3,1,3,3,3],
    'Operational_Performance': [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,3,3,2,2,3,3,3,3,3,3,3,3,3,3,3,2,3,3,3,3,3,2,3,3,3,2,2,3,3,3,3,3,3,3]
}
# Ensure all lists in data have the same length
max_len = max(len(v) for v in data.values())
for k, v in data.items():
    if len(v) < max_len:
        data[k] += [np.nan] * (max_len - len(v))  # Pad with NaN

# Create DataFrame
df = pd.DataFrame(data)



print("=== INFORMATION SHARING IMPACT ON SUPPLY CHAIN EFFICIENCY ===")
print("=== REGRESSION ANALYSIS FOR ABUJA MANUFACTURING FIRMS ===\n")

print("Dataset Overview:")
print(f"Number of responses: {len(df)}")
print(f"Number of companies: {df['Company'].nunique()}")
print("\nCompany distribution:")
print(df['Company'].value_counts())

# Create composite scores for Information Sharing and Supply Chain Efficiency
print("\n=== CREATING COMPOSITE SCORES ===")

# Information Sharing Score (reverse scale for questions 7 since 1=Very Likely is positive)
info_sharing_vars = ['Frequent_Info_Share', 'Effective_Supplier_Buyer_Structure', 
                    'Easy_Access_Info', 'Right_Info_Performance', 'Data_Connectivity']

# Reverse the scale (1->5, 2->4, 3->3, 4->2, 5->1) since lower values indicate better performance
for var in info_sharing_vars:
    df[f'{var}_reversed'] = 6 - df[var]

# Information types shared (already 0/1)
info_types = ['Inventory_Levels', 'Sales_Forecasts', 'Promotion_Strategies', 
              'Marketing_Plans', 'General_Feedback']

# Calculate Information Sharing Composite Score
df['Info_Sharing_Practices'] = df[[f'{var}_reversed' for var in info_sharing_vars]].mean(axis=1)
df['Info_Types_Shared'] = df[info_types].mean(axis=1)
df['Information_Sharing_Score'] = (df['Info_Sharing_Practices'] + df['Info_Types_Shared']) / 2

# Supply Chain Efficiency Score
efficiency_perceptions = ['Effective_Management_Performance', 'Internal_Environment_Influence', 
                         'Effective_Environment_Interaction', 'Reliable_Supply_Systems']

# Reverse scale for efficiency perceptions (lower is better)
for var in efficiency_perceptions:
    df[f'{var}_reversed'] = 6 - df[var]

efficiency_ratings = ['Cost_Efficiency', 'Quality_Level', 'Lead_Time', 
                     'Supplier_Reliability', 'Production_Flexibility', 'Operational_Performance']

df['Efficiency_Perceptions'] = df[[f'{var}_reversed' for var in efficiency_perceptions]].mean(axis=1)
df['Efficiency_Ratings'] = df[efficiency_ratings].mean(axis=1)
df['Supply_Chain_Efficiency_Score'] = (df['Efficiency_Perceptions'] + df['Efficiency_Ratings']) / 2

print("Composite Scores Created:")
print(f"Information Sharing Score: Mean = {df['Information_Sharing_Score'].mean():.3f}, Std = {df['Information_Sharing_Score'].std():.3f}")
print(f"Supply Chain Efficiency Score: Mean = {df['Supply_Chain_Efficiency_Score'].mean():.3f}, Std = {df['Supply_Chain_Efficiency_Score'].std():.3f}")

# Correlation Analysis
print("\n=== CORRELATION ANALYSIS ===")
correlation = df['Information_Sharing_Score'].corr(df['Supply_Chain_Efficiency_Score'])
print(f"Correlation between Information Sharing and Supply Chain Efficiency: {correlation:.4f}")

# Interpret correlation strength
if abs(correlation) >= 0.7:
    strength = "Strong"
elif abs(correlation) >= 0.5:
    strength = "Moderate"
elif abs(correlation) >= 0.3:
    strength = "Weak"
else:
    strength = "Very Weak"

direction = "Positive" if correlation > 0 else "Negative"
print(f"Interpretation: {strength} {direction} Correlation")

# Simple Linear Regression
print("\n=== SIMPLE LINEAR REGRESSION ANALYSIS ===")
X = df[['Information_Sharing_Score']]
y = df['Supply_Chain_Efficiency_Score']
mask = X.notna().values.flatten() & y.notna()
X = X[mask]
y = y[mask]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Model Statistics
r_squared = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"Regression Equation: Supply Chain Efficiency = {model.intercept_:.4f} + {model.coef_[0]:.4f} * Information Sharing")
print(f"R-squared: {r_squared:.4f} ({r_squared*100:.2f}% of variance explained)")
print(f"RMSE: {rmse:.4f}")

# Statistical Significance Test
n = len(df)
degrees_freedom = n - 2
t_stat = correlation * np.sqrt(degrees_freedom) / np.sqrt(1 - correlation**2)
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), degrees_freedom))

print(f"\nStatistical Significance:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significance level: {'Significant at α=0.05' if p_value < 0.05 else 'Not significant at α=0.05'}")

# Multiple Regression with Individual Information Sharing Components
print("\n=== MULTIPLE REGRESSION ANALYSIS ===")
data_multi = df[['Info_Sharing_Practices', 'Info_Types_Shared', 'Supply_Chain_Efficiency_Score']].dropna()
X_multi = data_multi[['Info_Sharing_Practices', 'Info_Types_Shared']]
y_multi = data_multi['Supply_Chain_Efficiency_Score']

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)
y_pred_multi = model_multi.predict(X_multi)

r_squared_multi = r2_score(y_multi, y_pred_multi)
print(f"Multiple R-squared: {r_squared_multi:.4f} ({r_squared_multi*100:.2f}% of variance explained)")
print("Regression Coefficients:")
for name, coef in zip(X_multi.columns, model_multi.coef_):
    print(f"  {name}: {coef:.4f}")
print(f"  Intercept: {model_multi.intercept_:.4f}")

# Detailed breakdown by Information Sharing aspects
print("\n=== DETAILED ANALYSIS BY INFORMATION SHARING ASPECTS ===")

# Individual correlations
aspects = {
    'Frequent Information Sharing': 'Frequent_Info_Share',
    'Effective Supplier-Buyer Structure': 'Effective_Supplier_Buyer_Structure', 
    'Easy Access to Information': 'Easy_Access_Info',
    'Right Information on Performance': 'Right_Info_Performance',
    'Data Connectivity': 'Data_Connectivity'
}

print("Individual Information Sharing Aspects vs Supply Chain Efficiency:")
for aspect_name, aspect_var in aspects.items():
    # Reverse scale for correlation
    aspect_reversed = 6 - df[aspect_var]
    corr = aspect_reversed.corr(df['Supply_Chain_Efficiency_Score'])
    print(f"  {aspect_name}: r = {corr:.4f}")

# Information types analysis
print("\nInformation Types Shared vs Supply Chain Efficiency:")
for info_type in info_types:
    corr = df[info_type].corr(df['Supply_Chain_Efficiency_Score'])
    share_rate = df[info_type].mean() * 100
    print(f"  {info_type}: r = {corr:.4f} (Shared by {share_rate:.1f}% of companies)")

# Company-wise analysis
print("\n=== COMPANY-WISE ANALYSIS ===")
company_analysis = df.groupby('Company').agg({
    'Information_Sharing_Score': ['mean', 'std'],
    'Supply_Chain_Efficiency_Score': ['mean', 'std']
}).round(4)

print("Average scores by company:")
for company in df['Company'].unique():
    company_data = df[df['Company'] == company]
    info_mean = company_data['Information_Sharing_Score'].mean()
    eff_mean = company_data['Supply_Chain_Efficiency_Score'].mean()
    n = len(company_data)
    print(f"  {company} (n={n}): Info={info_mean:.3f}, Efficiency={eff_mean:.3f}")

# Effect size interpretation
print("\n=== EFFECT SIZE AND PRACTICAL SIGNIFICANCE ===")
# Cohen's guidelines for correlation effect sizes
if abs(correlation) >= 0.5:
    effect_size = "Large"
elif abs(correlation) >= 0.3:
    effect_size = "Medium"
elif abs(correlation) >= 0.1:
    effect_size = "Small"
else:
    effect_size = "Negligible"

print(f"Effect Size: {effect_size} (Cohen's guidelines)")
print(f"Coefficient of Determination (r²): {r_squared:.4f}")
print(f"This means {r_squared*100:.2f}% of the variation in supply chain efficiency")
print(f"can be explained by information sharing practices.")

# Practical implications
print("\n=== PRACTICAL IMPLICATIONS ===")
if correlation > 0.3 and p_value < 0.05:
    print("✅ SIGNIFICANT POSITIVE IMPACT FOUND")
    print("Information sharing has a statistically significant positive impact on supply chain efficiency.")
    print(f"For every 1-unit increase in information sharing score, supply chain efficiency increases by {model.coef_[0]:.4f} units.")
elif correlation > 0 and p_value < 0.05:
    print("✅ POSITIVE IMPACT FOUND (Moderate)")
    print("Information sharing has a statistically significant but moderate positive impact on supply chain efficiency.")
elif p_value >= 0.05:
    print("⚠️ NO SIGNIFICANT IMPACT FOUND")
    print("The relationship between information sharing and supply chain efficiency is not statistically significant.")
else:
    print("❌ NEGATIVE OR NO CLEAR IMPACT")

# Recommendations based on findings
print("\n=== RECOMMENDATIONS FOR ABUJA MANUFACTURING FIRMS ===")

strongest_aspect = max(aspects.items(), key=lambda x: (6 - df[x[1]]).corr(df['Supply_Chain_Efficiency_Score']))
strongest_info_type = max([(name, df[name].corr(df['Supply_Chain_Efficiency_Score'])) for name in info_types], key=lambda x: x[1])

print(f"1. Focus on '{strongest_aspect[0]}' as it shows the strongest correlation with efficiency")
print(f"2. Prioritize sharing '{strongest_info_type[0]}' information (highest correlation: {strongest_info_type[1]:.4f})")
print(f"3. General Feedback sharing is universal ({df['General_Feedback'].mean()*100:.0f}% adoption) - maintain this practice")

if correlation > 0.3:
    print("4. Invest in information sharing infrastructure - it significantly impacts efficiency")
    print("5. Develop structured information sharing protocols with suppliers")
elif correlation > 0:
    print("4. Consider improving information sharing practices for modest efficiency gains")
    print("5. Evaluate current information sharing barriers and address them systematically")
else:
    print("4. Review current information sharing processes - they may need fundamental restructuring")
    print("5. Focus on other supply chain improvement areas that may have stronger impact")

print(f"\n=== CONCLUSION ===")
print(f"Based on the analysis of {len(df)} responses from {df['Company'].nunique()} manufacturing firms in Abuja:")
print(f"Information Sharing explains {r_squared*100:.2f}% of the variance in Supply Chain Efficiency")
print(f"The relationship is {strength.lower()} and {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")
print(f"Correlation coefficient: {correlation:.4f} (p-value: {p_value:.4f})")