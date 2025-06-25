# Load and explore the unemployment data from India
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Unemployment in India.csv', encoding='UTF-8-SIG')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\
Column names:")
print(df.columns.tolist())
print("\
First few rows:")
print(df.head())

# Data cleaning and preprocessing
# Clean column names by removing leading/trailing spaces
df.columns = [col.strip() for col in df.columns]

# Display cleaned column names
print("Cleaned column names:")
print(df.columns.tolist())

# Check data types and missing values
print("\
Data types:")
print(df.dtypes)
print("\
Missing values:")
print(df.isnull().sum())

# Check unique values in categorical columns
print("\
Unique regions:", df['Region'].nunique())
print("Unique areas:", df['Area'].unique())
print("Unique frequencies:", df['Frequency'].unique())

# Handle missing values and clean the data
# Remove rows with missing values
df_clean = df.dropna()
print("Shape after removing missing values:", df_clean.shape)

# Convert Date column to datetime
df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d-%m-%Y')

# Clean the Frequency column (remove leading spaces)
df_clean['Frequency'] = df_clean['Frequency'].str.strip()

# Display basic statistics
print("\
Basic statistics for unemployment rate:")
print(df_clean['Estimated Unemployment Rate (%)'].describe())

print("\
Date range:")
print("From:", df_clean['Date'].min())
print("To:", df_clean['Date'].max())

print("\
Unique regions:")
print(df_clean['Region'].unique())

# Fix the date format issue by stripping spaces
df_clean['Date'] = df_clean['Date'].str.strip()

# Now convert to datetime
df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d-%m-%Y')

# Clean the Frequency column (remove leading spaces)
df_clean['Frequency'] = df_clean['Frequency'].str.strip()

# Display basic statistics
print("Basic statistics for unemployment rate:")
print(df_clean['Estimated Unemployment Rate (%)'].describe())

print("\
Date range:")
print("From:", df_clean['Date'].min())
print("To:", df_clean['Date'].max())

print("\
Data shape after cleaning:", df_clean.shape)

# Visualization 1: Overall unemployment rate distribution
plt.figure(figsize=(12, 8))

# Subplot 1: Histogram of unemployment rates
plt.subplot(2, 2, 1)
plt.hist(df_clean['Estimated Unemployment Rate (%)'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Unemployment Rates')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')

# Subplot 2: Box plot by area (Rural vs Urban)
plt.subplot(2, 2, 2)
sns.boxplot(data=df_clean, x='Area', y='Estimated Unemployment Rate (%)')
plt.title('Unemployment Rate by Area Type')
plt.ylabel('Unemployment Rate (%)')

# Subplot 3: Time series of average unemployment rate
plt.subplot(2, 2, 3)
monthly_avg = df_clean.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()
plt.plot(monthly_avg['Date'], monthly_avg['Estimated Unemployment Rate (%)'], marker='o', linewidth=2)
plt.title('Average Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)

# Subplot 4: Unemployment rate by year
plt.subplot(2, 2, 4)
yearly_avg = df_clean.groupby('Year')['Estimated Unemployment Rate (%)'].mean()
plt.bar(yearly_avg.index, yearly_avg.values, color=['lightcoral', 'lightblue'])
plt.title('Average Unemployment Rate by Year')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')

plt.tight_layout()
plt.show()

# COVID-19 Impact Analysis
# Define pre-COVID and COVID periods
# COVID-19 was declared a pandemic in March 2020, so we'll use March 2020 as the cutoff

import datetime

covid_start = datetime.date(2020, 3, 1)
df_clean['COVID_Period'] = df_clean['Date'].dt.date >= covid_start

# Create period labels
df_clean['Period'] = df_clean['COVID_Period'].map({False: 'Pre-COVID (2019-Feb 2020)', True: 'COVID Period (Mar-Jun 2020)'})

# Compare unemployment rates before and during COVID
print("COVID-19 Impact on Unemployment Rates:")
print("=" * 50)

period_stats = df_clean.groupby('Period')['Estimated Unemployment Rate (%)'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
print(period_stats)

print("\
Number of observations by period:")
print(df_clean['Period'].value_counts())

# Statistical significance test
from scipy import stats
pre_covid = df_clean[df_clean['Period'] == 'Pre-COVID (2019-Feb 2020)']['Estimated Unemployment Rate (%)']
covid_period = df_clean[df_clean['Period'] == 'COVID Period (Mar-Jun 2020)']['Estimated Unemployment Rate (%)']

t_stat, p_value = stats.ttest_ind(pre_covid, covid_period)
print(f"\
Statistical Test Results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

# Seasonal and temporal pattern analysis
# First, let's examine the data structure for seasonal analysis

print("Analyzing Seasonal and Temporal Patterns")
print("=" * 45)

# Create month and year columns for better analysis
df_clean['Month'] = df_clean['Date'].dt.month
df_clean['Year'] = df_clean['Date'].dt.year
df_clean['Quarter'] = df_clean['Date'].dt.quarter

# Monthly patterns across all years
monthly_patterns = df_clean.groupby('Month')['Estimated Unemployment Rate (%)'].agg(['mean', 'std', 'count']).round(2)
monthly_patterns.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print("Monthly Unemployment Patterns (All Years):")
print(monthly_patterns)

# Year-over-year comparison
yearly_patterns = df_clean.groupby('Year')['Estimated Unemployment Rate (%)'].agg(['mean', 'std', 'count']).round(2)
print("\
Yearly Unemployment Patterns:")
print(yearly_patterns)

# Quarterly patterns
quarterly_patterns = df_clean.groupby('Quarter')['Estimated Unemployment Rate (%)'].agg(['mean', 'std']).round(2)
quarterly_patterns.index = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']
print("\
Quarterly Unemployment Patterns:")
print(quarterly_patterns)

# Separate analysis for 2019 (normal year) vs 2020 (COVID year) to identify true seasonal patterns
print("Seasonal Patterns Analysis - Separating Normal vs COVID Impact")
print("=" * 60)

# 2019 patterns (normal year)
df_2019 = df_clean[df_clean['Year'] == 2019]
monthly_2019 = df_2019.groupby('Month_Name')['Estimated Unemployment Rate (%)'].mean().round(2)
month_order = ['May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_2019_ordered = monthly_2019.reindex([m for m in month_order if m in monthly_2019.index])

print("2019 Monthly Patterns (Normal Year):")
for month, rate in monthly_2019_ordered.items():
    print(f"{month}: {rate}%")

# Pre-COVID months in 2020 (Jan-Feb)
df_2020_precovid = df_clean[(df_clean['Year'] == 2020) & (df_clean['Month'] <= 2)]
if not df_2020_precovid.empty:
    monthly_2020_precovid = df_2020_precovid.groupby('Month_Name')['Estimated Unemployment Rate (%)'].mean().round(2)
    print(f"\
2020 Pre-COVID Months:")
    for month, rate in monthly_2020_precovid.items():
        print(f"{month}: {rate}%")

# Rural vs Urban seasonal patterns (2019 only)
print(f"\
Rural vs Urban Seasonal Patterns (2019):")
seasonal_rural_urban = df_2019.groupby(['Month_Name', 'Area'])['Estimated Unemployment Rate (%)'].mean().unstack().round(2)
print(seasonal_rural_urban)

# Create visualizations for seasonal patterns
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Monthly patterns comparison (2019 vs overall)
ax1 = axes[0, 0]
months_full = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_overall = df_clean.groupby('Month')['Estimated Unemployment Rate (%)'].mean()
monthly_2019_full = df_2019.groupby('Month')['Estimated Unemployment Rate (%)'].mean()

ax1.plot(range(1, 13), monthly_overall, marker='o', label='Overall (2019-2020)', linewidth=2, color='red')
ax1.plot(monthly_2019_full.index, monthly_2019_full.values, marker='s', label='2019 Only (Normal)', linewidth=2, color='blue')
ax1.set_xlabel('Month')
ax1.set_ylabel('Unemployment Rate (%)')
ax1.set_title('Monthly Unemployment Patterns')
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(months_full)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Quarterly patterns
ax2 = axes[0, 1]
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
quarterly_rates = df_clean.groupby('Quarter')['Estimated Unemployment Rate (%)'].mean()
bars = ax2.bar(quarters, quarterly_rates, color=['lightblue', 'orange', 'lightgreen', 'pink'])
ax2.set_ylabel('Unemployment Rate (%)')
ax2.set_title('Quarterly Unemployment Patterns')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, quarterly_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{value:.1f}%', ha='center', va='bottom')

# 3. Rural vs Urban seasonal patterns (2019)
ax3 = axes[1, 0]
rural_2019 = df_2019[df_2019['Area'] == 'Rural'].groupby('Month')['Estimated Unemployment Rate (%)'].mean()
urban_2019 = df_2019[df_2019['Area'] == 'Urban'].groupby('Month')['Estimated Unemployment Rate (%)'].mean()

ax3.plot(rural_2019.index, rural_2019.values, marker='o', label='Rural', linewidth=2, color='green')
ax3.plot(urban_2019.index, urban_2019.values, marker='s', label='Urban', linewidth=2, color='purple')
ax3.set_xlabel('Month')
ax3.set_ylabel('Unemployment Rate (%)')
ax3.set_title('Rural vs Urban Seasonal Patterns (2019)')
ax3.set_xticks(range(5, 13))
ax3.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Year-over-year comparison by month
ax4 = axes[1, 1]
df_pivot = df_clean.pivot_table(values='Estimated Unemployment Rate (%)', 
                                index='Month', columns='Year', aggfunc='mean')
ax4.plot(df_pivot.index, df_pivot[2019], marker='o', label='2019', linewidth=2, color='blue')
ax4.plot(df_pivot.index, df_pivot[2020], marker='s', label='2020', linewidth=2, color='red')
ax4.set_xlabel('Month')
ax4.set_ylabel('Unemployment Rate (%)')
ax4.set_title('Year-over-Year Monthly Comparison')
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(months_full)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical analysis of seasonal patterns
print("\
Statistical Analysis of Seasonal Patterns:")
print("=" * 45)

# Calculate coefficient of variation for each year to measure seasonal volatility
cv_2019 = (df_2019.groupby('Month')['Estimated Unemployment Rate (%)'].mean().std() / 
           df_2019.groupby('Month')['Estimated Unemployment Rate (%)'].mean().mean()) * 100

cv_overall = (monthly_overall.std() / monthly_overall.mean()) * 100

print(f"Seasonal Volatility (Coefficient of Variation):")
print(f"2019 (Normal Year): {cv_2019:.1f}%")
print(f"Overall (2019-2020): {cv_overall:.1f}%")

# Identify peak and trough months
peak_month_2019 = monthly_2019_full.idxmax()
trough_month_2019 = monthly_2019_full.idxmin()
peak_month_overall = monthly_overall.idxmax()
trough_month_overall = monthly_overall.idxmin()

print(f"\
Peak and Trough Analysis:")
print(f"2019 Peak Month: {months_full[peak_month_2019-1]} ({monthly_2019_full.max():.2f}%)")
print(f"2019 Trough Month: {months_full[trough_month_2019-1]} ({monthly_2019_full.min():.2f}%)")
print(f"Overall Peak Month: {months_full[peak_month_overall-1]} ({monthly_overall.max():.2f}%)")
print(f"Overall Trough Month: {months_full[trough_month_overall-1]} ({monthly_overall.min():.2f}%)") 

# Regional seasonal patterns analysis
print("Regional Seasonal Pattern Analysis")
print("=" * 40)

# Identify states with strongest seasonal patterns (using 2019 data)
regional_seasonal = df_2019.groupby(['Region', 'Month'])['Estimated Unemployment Rate (%)'].mean().unstack()
regional_cv = {}

for region in regional_seasonal.index:
    monthly_rates = regional_seasonal.loc[region].dropna()
    if len(monthly_rates) > 3:  # Only consider regions with sufficient data
        cv = (monthly_rates.std() / monthly_rates.mean()) * 100
        regional_cv[region] = cv

# Sort by seasonal volatility
regional_cv_sorted = dict(sorted(regional_cv.items(), key=lambda x: x[1], reverse=True))

print("Top 10 States with Highest Seasonal Volatility (2019):")
for i, (region, cv) in enumerate(list(regional_cv_sorted.items())[:10]):
    print(f"{i+1:2d}. {region}: {cv:.1f}%")

print(f"\
Top 10 States with Lowest Seasonal Volatility (2019):")
for i, (region, cv) in enumerate(list(regional_cv_sorted.items())[-10:]):
    print(f"{i+1:2d}. {region}: {cv:.1f}%")

# Analyze agricultural vs non-agricultural states seasonal patterns
# Agricultural states typically show more seasonality
agricultural_states = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Bihar', 'West Bengal', 
                      'Madhya Pradesh', 'Rajasthan', 'Maharashtra', 'Karnataka', 'Andhra Pradesh']

agri_seasonal = []
non_agri_seasonal = []

for region, cv in regional_cv.items():
    if region in agricultural_states:
        agri_seasonal.append(cv)
    else:
        non_agri_seasonal.append(cv)

if agri_seasonal and non_agri_seasonal:
    print(f"\
Agricultural vs Non-Agricultural States Seasonality:")
    print(f"Agricultural States Average CV: {sum(agri_seasonal)/len(agri_seasonal):.1f}%")
    print(f"Non-Agricultural States Average CV: {sum(non_agri_seasonal)/len(non_agri_seasonal):.1f}%")