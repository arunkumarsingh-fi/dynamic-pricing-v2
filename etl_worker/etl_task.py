import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from datetime import datetime

print("ETL task started: Preparing enhanced training data with dual data sources...")

# Business Rationale: Create a single "source of truth" by combining historical data 
# with richer synthetic data for consistent analytics and ML model training

# Input paths for dual data sources (adjust for local development)
base_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
iphone_price_path = os.path.join(base_data_path, 'iphone_price_data.csv')
synthetic_sales_path = os.path.join(base_data_path, 'synthetic_sales_data.csv')

# Output paths
analytics_output_path = os.path.join(base_data_path, 'analytics_data.csv')
ml_output_path = os.path.join(base_data_path, 'processed_ml_data.csv')
encoder_path = os.path.join(base_data_path, 'encoder.joblib')
scaler_path = os.path.join(base_data_path, 'scaler.joblib')

print("Loading original iPhone price data...")
iphone_df = pd.read_csv(iphone_price_path)

print("Loading synthetic sales data...")
synthetic_df = pd.read_csv(synthetic_sales_path)

print("Step 1: Harmonizing original iPhone price data to match synthetic schema...")

def harmonize_original_data(df):
    """
    Harmonize original iPhone price data to match synthetic data schema.
    
    Business Logic:
    - Convert currency from INR to EUR (1 EUR = 90 INR)
    - Rename columns to match synthetic schema
    - Create missing columns with sensible defaults
    - Calculate profit metrics based on selling price
    """
    harmonized_df = df.copy()
    
    # Currency conversion: INR to EUR (1 EUR = 90 INR)
    EUR_TO_INR_RATE = 90
    
    # Column renaming to match synthetic data schema
    harmonized_df['model'] = harmonized_df['Model']  # Product_name -> model
    harmonized_df['selling_price_eur'] = harmonized_df['current_price(LKR)'] / EUR_TO_INR_RATE  # Convert to EUR
    
    # Create missing columns with sensible defaults
    harmonized_df['date'] = datetime.now().strftime('%Y-%m-%d')  # Current date as default
    harmonized_df['market'] = 'Greece'  # Default market (replaced India with Greece)
    
    # Infer has_damage from existing damage columns
    harmonized_df['has_damage'] = (
        harmonized_df['backglass_damages'].astype(bool) | 
        (harmonized_df['screen_damages'] != 'undamaged')
    ).astype(bool)
    
    # Calculate acquisition cost (assume 70% of selling price)
    harmonized_df['acquisition_cost_eur'] = harmonized_df['selling_price_eur'] * 0.70
    
    # Calculate profit (selling price - acquisition cost - operational costs)
    operational_cost_rate = 0.15  # 15% operational costs
    harmonized_df['profit_eur'] = (
        harmonized_df['selling_price_eur'] - 
        harmonized_df['acquisition_cost_eur'] - 
        (harmonized_df['selling_price_eur'] * operational_cost_rate)
    )
    
    # Default days to sell (assume 14 days for historical data)
    harmonized_df['days_to_sell'] = 14
    
    # Select only the columns that match synthetic data schema
    final_columns = ['date', 'model', 'market', 'battery_health', 'has_damage', 
                    'acquisition_cost_eur', 'selling_price_eur', 'profit_eur', 'days_to_sell']
    
    return harmonized_df[final_columns]

# Apply harmonization
harmonized_original = harmonize_original_data(iphone_df)
print(f"Harmonized {len(harmonized_original)} original records")

print("Step 2: Combining datasets...")
# Ensure synthetic data has the same column structure
synthetic_df_clean = synthetic_df[['date', 'model', 'market', 'battery_health', 'has_damage', 
                                  'acquisition_cost_eur', 'selling_price_eur', 'profit_eur', 'days_to_sell']]

# Combine the datasets using pd.concat
combined_df = pd.concat([synthetic_df_clean, harmonized_original], ignore_index=True)
print(f"Combined dataset contains {len(combined_df)} total records")

print("Step 3: Adding baseline comparison metrics...")
# Calculate vanilla profit baseline (fixed "Market Rate" tier of 1.0)
def calculate_vanilla_profit(row):
    """
    Calculate profit if device had been sold at fixed Market Rate (1.0 tier).
    This creates a baseline for comparison with the intelligent pricing model.
    """
    market_rate_multiplier = 1.0
    baseline_selling_price = row['selling_price_eur'] * market_rate_multiplier
    baseline_profit = baseline_selling_price - row['acquisition_cost_eur'] - (baseline_selling_price * 0.15)
    return baseline_profit

combined_df['vanilla_profit_eur'] = combined_df.apply(calculate_vanilla_profit, axis=1)

print("Step 4: Creating analytics-ready dataset...")
# Add time-series features for analytics
combined_df['date'] = pd.to_datetime(combined_df['date'])
combined_df['year'] = combined_df['date'].dt.year
combined_df['month'] = combined_df['date'].dt.month
combined_df['quarter'] = combined_df['date'].dt.quarter
combined_df['day_of_week'] = combined_df['date'].dt.dayofweek

# Add derived analytics features
combined_df['revenue_eur'] = combined_df['selling_price_eur']  # Revenue = selling price
combined_df['profit_margin'] = combined_df['profit_eur'] / combined_df['selling_price_eur']  # Profit margin %

# Save analytics-ready dataset
combined_df.to_csv(analytics_output_path, index=False)
print(f"Analytics dataset saved to {analytics_output_path}")

print("Step 5: Creating ML model-ready dataset...")
# Prepare features for ML model
ml_df = combined_df.copy()

# Create categorical and numerical features for ML model
# Note: Simplified feature set (removed Storage, RAM, Screen Size as per requirements)
categorical_features = ['market', 'model']
numerical_features = ['battery_health', 'days_to_sell']

# Convert boolean to int
ml_df['has_damage_int'] = ml_df['has_damage'].astype(int)
numerical_features.append('has_damage_int')

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_categorical = encoder.fit_transform(ml_df[categorical_features])

# Scale numerical features
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(ml_df[numerical_features])

# Combine features
feature_names = list(encoder.get_feature_names_out(categorical_features)) + numerical_features
features_combined = np.hstack([encoded_categorical, scaled_numerical])

# Create final ML dataset
ml_final_df = pd.DataFrame(features_combined, columns=feature_names)
ml_final_df['selling_price_eur'] = ml_df['selling_price_eur'].values
ml_final_df['profit_eur'] = ml_df['profit_eur'].values
ml_final_df['vanilla_profit_eur'] = ml_df['vanilla_profit_eur'].values

# Save ML-ready dataset and artifacts
ml_final_df.to_csv(ml_output_path, index=False)
joblib.dump(encoder, encoder_path)
joblib.dump(scaler, scaler_path)

print(f"ML dataset saved to {ml_output_path}")
print(f"Encoder saved to {encoder_path}")
print(f"Scaler saved to {scaler_path}")
print("ETL task completed successfully!")
