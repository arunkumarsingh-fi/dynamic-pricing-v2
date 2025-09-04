from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mabwiser.mab import MAB, LearningPolicy
import uuid
import os
import joblib
from datetime import datetime

app = Flask(__name__)

# Global variables
models = {}
encoder = None
scaler = None
feature_names = None
active_decisions = {}
evaluation_history = []

# Currency conversion rates (as of 2024)
CURRENCY_RATES = {
    'LKR_TO_EUR': 0.0031,  # 1 LKR = 0.0031 EUR (approximately)
    'USD_TO_EUR': 0.92     # 1 USD = 0.92 EUR (approximately)
}

# Market profiles for strategic multi-market analysis
MARKET_PROFILES = {
    "Romania": {"identity": "Cheaper", "price_index": 0.85, "logistics_cost_eur": 25},
    "Bulgaria": {"identity": "Cheaper", "price_index": 0.82, "logistics_cost_eur": 28},
    "Greece": {"identity": "Competitive", "price_index": 1.05, "logistics_cost_eur": 18},
    "Poland": {"identity": "Decent", "price_index": 1.0, "logistics_cost_eur": 20},
    "Finland": {"identity": "Competitive", "price_index": 1.20, "logistics_cost_eur": 15}
}

class PriceRecommendationEngine:
    """Enhanced pricing engine that returns actual EUR prices instead of tiers"""
    
    def __init__(self):
        self.tier_multipliers = [0.9, 1.0, 1.1]  # Internal tiers
        self.tier_names = {
            0.9: "Competitive",
            1.0: "Market Rate", 
            1.1: "Premium"
        }
    
    def convert_lkr_to_eur(self, lkr_price):
        """Convert LKR price to EUR"""
        return round(lkr_price * CURRENCY_RATES['LKR_TO_EUR'], 2)
    
    def calculate_recommended_prices(self, base_price_lkr, recommended_tier):
        """Calculate actual price recommendations for all tiers"""
        base_price_eur = self.convert_lkr_to_eur(base_price_lkr)
        
        prices = {}
        for tier in self.tier_multipliers:
            tier_price_eur = round(base_price_eur * tier, 2)
            prices[tier] = {
                'price_eur': tier_price_eur,
                'price_lkr': round(base_price_lkr * tier, 0),
                'strategy': self.tier_names[tier],
                'recommended': (tier == recommended_tier)
            }
        
        return prices
    
    def estimate_market_price(self, device_info, market_profile=None):
        """Estimate market price based on simplified device specifications and model"""
        # Get core device information
        model = device_info.get('Model', 'iPhone 11')
        battery_health = device_info.get('Battery', 95)
        
        # Model-based base pricing (simplified approach)
        model_lower = str(model).lower()
        if '15' in model_lower:
            base_value = 120000  # iPhone 15 series
        elif '14' in model_lower:
            base_value = 100000  # iPhone 14 series
        elif '13' in model_lower:
            if 'pro' in model_lower:
                base_value = 85000   # iPhone 13 Pro series
            else:
                base_value = 70000   # iPhone 13 series
        elif '12' in model_lower:
            if 'pro' in model_lower:
                base_value = 75000   # iPhone 12 Pro series
            else:
                base_value = 60000   # iPhone 12 series
        elif '11' in model_lower:
            if 'pro' in model_lower:
                base_value = 55000   # iPhone 11 Pro series
            else:
                base_value = 45000   # iPhone 11 series
        else:
            base_value = 35000   # Older models
        
        # Battery condition adjustment
        battery_multiplier = battery_health / 100
        
        # Damage penalties
        screen_damage = device_info.get('Screen_Damage', 0)
        backglass_damage = device_info.get('Backglass_Damage', 0)
        damage_penalty = (screen_damage + backglass_damage) * 0.15
        
        # Calculate estimated market price
        estimated_price = base_value * battery_multiplier * (1 - damage_penalty)
        
        # Apply market profile adjustment if provided
        if market_profile:
            estimated_price *= market_profile.get('price_index', 1.0)
        
        return max(15000, estimated_price)  # Minimum 15,000 LKR

price_engine = PriceRecommendationEngine()

def enhanced_feature_engineering(df):
    """Create simplified features focused on Model, Battery, and Condition"""
    df = df.copy()
    
    # Handle missing columns with defaults
    if 'Months_since_release' not in df.columns:
        df['Months_since_release'] = 24
    else:
        df['Months_since_release'] = df['Months_since_release'].fillna(24)
    
    # Simplified interaction features (Storage, RAM, Screen Size removed)
    df['Battery_Age_interaction'] = df['Battery'] * (100 - df['Months_since_release'])
    df['Damage_Total'] = df['Backglass_Damage'] + df['Screen_Damage']
    
    # Polynomial features (only battery-related)
    df['Battery_squared'] = df['Battery'] ** 2
    
    # Simplified market segmentation based on model instead of storage
    def get_market_segment(row):
        model_lower = str(row['Model']).lower()
        if 'pro max' in model_lower or '15' in model_lower:
            return 'premium'
        elif 'pro' in model_lower or '14' in model_lower or '13' in model_lower:
            return 'high_end'
        elif '12' in model_lower or '11' in model_lower:
            return 'mid_range'
        else:
            return 'budget'
    
    df['Market_Segment'] = df.apply(get_market_segment, axis=1)
    
    # Device condition score (simplified)
    df['Condition_Score'] = (
        df['Battery'] * 0.5 +
        (1 - df['Backglass_Damage']) * 25 +
        (1 - df['Screen_Damage']) * 25
    )
    
    return df

def calculate_business_reward(price_tier, base_price, context):
    """Calculate business-oriented reward function with more realistic variability"""
    battery_health = context.get('Battery', 95)
    damage_penalty = context.get('Backglass_Damage', 0) + context.get('Screen_Damage', 0)
    inventory = context.get('inventory_level', 'decent')
    new_model_imminent = context.get('new_model_imminent', False)
    
    # More varied profit margins by tier with randomness
    import random
    base_margins = {0.9: 0.12, 1.0: 0.22, 1.1: 0.32}  # Reduced base margins
    margin_variance = random.uniform(-0.05, 0.05)  # ±5% variance
    profit_margins = {k: max(0.05, v + margin_variance) for k, v in base_margins.items()}
    
    # More realistic sale probabilities with greater variation
    base_probs = {0.9: 0.75, 1.0: 0.60, 1.1: 0.45}  # More differentiated
    sale_prob = base_probs.get(price_tier, 0.60)
    
    # Add randomness to simulate market uncertainty
    prob_variance = random.uniform(-0.15, 0.15)  # ±15% variance
    sale_prob = max(0.2, min(0.95, sale_prob + prob_variance))
    
    # Stronger condition penalties
    condition_factor = (battery_health / 100) * (1 - damage_penalty * 0.3)  # Increased penalty
    sale_prob *= condition_factor
    
    # More significant inventory impact
    if inventory == 'high':
        sale_prob *= 1.15  # Higher boost for high inventory (need to move stock)
    elif inventory == 'low':
        sale_prob *= 0.85  # Bigger penalty for low inventory (can be picky)
    
    # New model imminent penalty
    if new_model_imminent:
        sale_prob *= 0.75  # 25% penalty for imminent releases
    
    # Calculate expected profit with more realism
    expected_profit = sale_prob * base_price * price_tier * profit_margins.get(price_tier, 0.22)
    
    # Add some noise to prevent identical outcomes
    noise = random.uniform(-0.1, 0.1)  # ±10% noise
    expected_profit *= (1 + noise)
    
    return max(0, expected_profit)  # Ensure non-negative

def initialize_models():
    """Initialize bandit models with simplified features"""
    global models, encoder, scaler, feature_names
    
    # Load preprocessed encoder and scaler from ETL
    encoder_path = 'data/encoder.joblib'
    scaler_path = 'data/scaler.joblib'
    processed_data_path = 'data/processed_ml_data.csv'
    
    if not all(os.path.exists(p) for p in [encoder_path, scaler_path, processed_data_path]):
        print(f"Required ML files not found. Please run ETL first.")
        return False

    # Load the ML-ready dataset and artifacts
    df = pd.read_csv(processed_data_path)
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    
    arms = [0.9, 1.0, 1.1]
    
    # Use the feature columns from the ML dataset (preprocessing already applied)
    feature_columns = [col for col in df.columns if col not in ['selling_price_eur', 'profit_eur', 'vanilla_profit_eur']]
    contexts = df[feature_columns].values
    feature_names = feature_columns

    # Calculate business-oriented rewards using simplified profit from dataset
    rewards = []
    decisions = []
    
    for idx, row in df.iterrows():
        # Use actual profit from dataset as reward, converted to LKR equivalent
        optimal_tier = np.random.choice(arms)
        reward = row['profit_eur'] / CURRENCY_RATES['LKR_TO_EUR']  # Convert to LKR for internal calculations
        rewards.append(reward)
        decisions.append(optimal_tier)

    # Initialize different bandit algorithms with contextual support
    from mabwiser.mab import NeighborhoodPolicy
    
    algorithms = {
        'LinTS': LearningPolicy.LinTS(alpha=1.5),
        'LinUCB': LearningPolicy.LinUCB(alpha=1.0),
        'EpsilonGreedy': LearningPolicy.EpsilonGreedy(epsilon=0.1)
    }
    
    for name, policy in algorithms.items():
        # Create contextual bandit model with neighborhood policy for contexts
        model = MAB(
            arms=arms, 
            learning_policy=policy,
            neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=3)
        )
        model.fit(decisions=decisions, rewards=rewards, contexts=contexts)
        models[name] = model
    
    print(f"Initialized {len(models)} pricing models with EUR conversion.")
    return True

def prepare_input_context(device_info):
    """Prepare input context for model prediction using the same preprocessing as ETL"""
    # Create a temporary DataFrame with the input device info
    temp_df = pd.DataFrame([{
        'market': device_info.get('market', 'Poland'),  # Default market
        'model': device_info.get('Model', 'iPhone 11'),
        'battery_health': device_info.get('Battery', 95),
        'has_damage': device_info.get('Screen_Damage', 0) == 1 or device_info.get('Backglass_Damage', 0) == 1,
        'days_to_sell': 14  # Default value for prediction
    }])
    
    # Apply the same preprocessing as in ETL
    categorical_features = ['market', 'model']
    numerical_features = ['battery_health', 'days_to_sell']
    
    # Convert boolean to int
    temp_df['has_damage_int'] = temp_df['has_damage'].astype(int)
    numerical_features.append('has_damage_int')
    
    # Use the loaded encoder and scaler
    encoded_categorical = encoder.transform(temp_df[categorical_features])
    scaled_numerical = scaler.transform(temp_df[numerical_features])
    
    # Combine features
    context = np.hstack([encoded_categorical, scaled_numerical])
    
    return context

def calculate_dynamic_refurbishing_cost(estimated_market_price_lkr, screen_damage, backglass_damage):
    """Calculate dynamic refurbishing costs based on damage level"""
    # Convert LKR to EUR for cost calculations
    estimated_market_price_eur = estimated_market_price_lkr * CURRENCY_RATES['LKR_TO_EUR']
    
    if screen_damage == 0 and backglass_damage == 0:
        # Tier 1 (Minor): No damage
        tier = "Minor"
        cost_percentage = 0.04  # 4%
    else:
        # Tier 2 (Major): Has screen or backglass damage
        tier = "Major"
        cost_percentage = 0.15  # 15%
    
    refurbishing_cost_eur = estimated_market_price_eur * cost_percentage
    refurbishing_cost_lkr = refurbishing_cost_eur / CURRENCY_RATES['LKR_TO_EUR']
    
    return {
        'refurbishing_cost_eur': round(refurbishing_cost_eur, 2),
        'refurbishing_cost_lkr': round(refurbishing_cost_lkr, 0),
        'refurbishing_tier': tier,
        'cost_percentage': cost_percentage * 100  # For display purposes
    }

def calculate_updated_business_reward(selling_price_eur, acquisition_cost_eur, refurbishing_cost_eur, operational_cost_rate=0.10):
    """Calculate updated business reward with dynamic refurbishing costs"""
    operational_cost = selling_price_eur * operational_cost_rate
    total_costs = acquisition_cost_eur + refurbishing_cost_eur + operational_cost
    profit = selling_price_eur - total_costs
    return profit

@app.route('/recommend_price', methods=['POST'])
def recommend():
    """Enhanced recommendation with new_model_imminent, dynamic costs, and target acquisition price"""
    data = request.get_json()
    model_name = data.get('model', 'LinTS')
    
    if model_name not in models:
        return jsonify({'error': f'Model {model_name} not available'}), 400
    
    # Extract new features
    new_model_imminent = data.get('new_model_imminent', False)
    screen_damage = data.get('Screen_Damage', 0)
    backglass_damage = data.get('Backglass_Damage', 0)
    
    # Prepare input context using simplified preprocessing
    context = prepare_input_context(data)
    
    # Get prediction from selected model
    recommended_tier = models[model_name].predict(context)
    
    # Get market-specific pricing if market is specified
    selected_market = data.get('market', 'poland')
    market_profile = None
    
    # Find the market profile (case insensitive)
    for market_name, profile in MARKET_PROFILES.items():
        if market_name.lower() == selected_market.lower():
            market_profile = profile
            break
    
    # Estimate market price for this device with market adjustment
    estimated_market_price_lkr = price_engine.estimate_market_price(data, market_profile)
    estimated_market_price_eur = price_engine.convert_lkr_to_eur(estimated_market_price_lkr)
    
    # Calculate dynamic refurbishing costs
    refurbishing_details = calculate_dynamic_refurbishing_cost(
        estimated_market_price_lkr, screen_damage, backglass_damage
    )
    
    # Calculate target buying price with penalties for risk factors
    base_buying_percentage = 0.70  # Start with 70% of market value
    
    # Apply penalties for various risk factors
    inventory_level = data.get('inventory_level', 'decent')
    
    # Damage penalties (reduce buying price for damaged devices)
    damage_penalty = (screen_damage + backglass_damage) * 0.08  # 8% penalty per damage type
    
    # Low battery penalty
    battery = data.get('Battery', 95)
    if battery < 80:
        battery_penalty = (80 - battery) * 0.002  # 0.2% penalty per % below 80%
    else:
        battery_penalty = 0
    
    # Inventory level adjustments
    if inventory_level == 'high':
        inventory_penalty = 0.05  # 5% penalty for high inventory (less selective)
    elif inventory_level == 'low':
        inventory_bonus = 0.02  # 2% bonus for low inventory (can be picky)
        inventory_penalty = -inventory_bonus
    else:
        inventory_penalty = 0
    
    # New model imminent penalty
    if new_model_imminent:
        new_model_penalty = 0.10  # 10% penalty for imminent releases
    else:
        new_model_penalty = 0
    
    # Calculate adjusted buying percentage
    adjusted_percentage = base_buying_percentage - damage_penalty - battery_penalty - inventory_penalty - new_model_penalty
    adjusted_percentage = max(0.40, min(0.75, adjusted_percentage))  # Clamp between 40%-75%
    
    target_buying_price_lkr = estimated_market_price_lkr * adjusted_percentage
    target_buying_price_eur = estimated_market_price_eur * adjusted_percentage
    
    # Calculate all pricing options
    price_options = price_engine.calculate_recommended_prices(
        estimated_market_price_lkr, 
        recommended_tier
    )
    
    # Get the recommended price details
    recommended_option = price_options[recommended_tier]
    
    # Calculate simplified market segment based on model
    model_lower = str(data.get('Model', 'iPhone 11')).lower()
    if 'pro max' in model_lower or '15' in model_lower:
        market_segment = 'premium'
    elif 'pro' in model_lower or '14' in model_lower or '13' in model_lower:
        market_segment = 'high_end'
    elif '12' in model_lower or '11' in model_lower:
        market_segment = 'mid_range'
    else:
        market_segment = 'budget'
    
    # Calculate condition score
    battery = data.get('Battery', 95)
    condition_score = battery * 0.5 + (1 - backglass_damage) * 25 + (1 - screen_damage) * 25
    
    # Prepare market context (includes new_model_imminent flag)
    market_context = {
        'new_model_imminent': new_model_imminent,
        'market_segment': market_segment,
        'condition_tier': refurbishing_details['refurbishing_tier']
    }
    
    decision_id = str(uuid.uuid4())
    
    active_decisions[decision_id] = {
        'context': context,
        'recommended_tier': recommended_tier,
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'input_data': data,
        'estimated_market_price_lkr': estimated_market_price_lkr,
        'refurbishing_cost_eur': refurbishing_details['refurbishing_cost_eur']
    }
    
    return jsonify({
        'decision_id': decision_id,
        'recommended_tier': float(recommended_tier),
        'recommended_price_eur': recommended_option['price_eur'],
        'recommended_price_lkr': recommended_option['price_lkr'],
        'pricing_strategy': recommended_option['strategy'],
        'model_used': model_name,
        'market_segment': market_segment,
        'condition_score': float(condition_score),
        'estimated_market_value': {
            'eur': estimated_market_price_eur,
            'lkr': estimated_market_price_lkr
        },
        'target_acquisition_cost': {
            'lkr': round(target_buying_price_lkr, 0),
            'eur': round(target_buying_price_eur, 2)
        },
        'cost_breakdown': {
            'refurbishing_cost_lkr': refurbishing_details['refurbishing_cost_lkr'],
            'refurbishing_cost_eur': refurbishing_details['refurbishing_cost_eur'],
            'refurbishing_tier': refurbishing_details['refurbishing_tier'],
            'cost_percentage': refurbishing_details['cost_percentage']
        },
        'market_context': market_context,
        'all_pricing_options': price_options,
        'currency_info': {
            'primary_currency': 'EUR',
            'conversion_rate': f"1 LKR = {CURRENCY_RATES['LKR_TO_EUR']} EUR"
        }
    })

@app.route('/report_outcome', methods=['POST'])
def report():
    """Enhanced outcome reporting with evaluation metrics"""
    data = request.get_json()
    decision_id = data.get('decision_id')
    reward = data.get('reward')  # Expected to be in EUR now
    
    if decision_id not in active_decisions:
        return jsonify({'error': 'Decision ID not found'}), 404
    
    decision = active_decisions.pop(decision_id)
    model_name = decision['model']
    
    # Convert EUR reward to LKR for internal calculations
    reward_lkr = reward / CURRENCY_RATES['LKR_TO_EUR'] if reward else 0
    
    # Update the model
    models[model_name].partial_fit(
        decisions=[decision['recommended_tier']], 
        rewards=[reward_lkr], 
        contexts=decision['context']
    )
    
    # Store evaluation history
    evaluation_entry = {
        'timestamp': datetime.now().isoformat(),
        'decision_id': decision_id,
        'model': model_name,
        'recommended_tier': float(decision['recommended_tier']),
        'reward_eur': reward,
        'reward_lkr': reward_lkr
    }
    evaluation_history.append(evaluation_entry)
    
    return jsonify({'status': 'success', 'model_updated': model_name})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'currency': 'EUR',
        'total_decisions': len(active_decisions),
        'evaluation_history_size': len(evaluation_history)
    })

@app.route('/optimize_market_and_price', methods=['POST'])
def optimize_market_and_price():
    """Strategic endpoint to find the most profitable market and pricing combination"""
    device_info = request.get_json()
    model_name = device_info.get('model', 'LinTS')
    
    if model_name not in models:
        return jsonify({'error': f'Model {model_name} not available'}), 400
    
    # Extract device characteristics for cost calculations
    screen_damage = device_info.get('Screen_Damage', 0)
    backglass_damage = device_info.get('Backglass_Damage', 0)
    battery_health = device_info.get('Battery', 95)
    
    market_results = []
    
    # Loop through each market and analyze profitability
    for market_name, market_profile in MARKET_PROFILES.items():
        # Calculate market-adjusted price using the price index
        base_estimated_price_lkr = price_engine.estimate_market_price(device_info)
        market_adjusted_price_lkr = price_engine.estimate_market_price(device_info, market_profile)
        
        # Prepare context for this market
        market_device_info = device_info.copy()
        market_device_info['market'] = market_name
        context = prepare_input_context(market_device_info)
        
        # Get the best tier for this device in this market
        recommended_tier = models[model_name].predict(context)
        
        # Calculate final selling price with tier adjustment
        price_options = price_engine.calculate_recommended_prices(
            market_adjusted_price_lkr, recommended_tier
        )
        recommended_option = price_options[recommended_tier]
        final_selling_price_eur = recommended_option['price_eur']
        
        # Calculate all costs using same logic as single market
        # 1. Target acquisition cost with same penalty logic as single market
        # Use MARKET-ADJUSTED price for acquisition cost calculation (not base price)
        market_adjusted_price_eur = market_adjusted_price_lkr * CURRENCY_RATES['LKR_TO_EUR']
        base_buying_percentage = 0.70  # Start with 70% of market value
        
        # Apply same penalties as single market
        inventory_level = device_info.get('inventory_level', 'decent')
        new_model_imminent = device_info.get('new_model_imminent', False)
        
        # Damage penalties (reduce buying price for damaged devices)
        damage_penalty = (screen_damage + backglass_damage) * 0.08  # 8% penalty per damage type
        
        # Low battery penalty
        if battery_health < 80:
            battery_penalty = (80 - battery_health) * 0.002  # 0.2% penalty per % below 80%
        else:
            battery_penalty = 0
        
        # Inventory level adjustments
        if inventory_level == 'high':
            inventory_penalty = 0.05  # 5% penalty for high inventory
        elif inventory_level == 'low':
            inventory_penalty = -0.02  # 2% bonus for low inventory
        else:
            inventory_penalty = 0
        
        # New model imminent penalty
        if new_model_imminent:
            new_model_penalty = 0.10  # 10% penalty for imminent releases
        else:
            new_model_penalty = 0
        
        # Calculate adjusted buying percentage (same as single market)
        adjusted_percentage = base_buying_percentage - damage_penalty - battery_penalty - inventory_penalty - new_model_penalty
        adjusted_percentage = max(0.40, min(0.75, adjusted_percentage))  # Clamp between 40%-75%
        
        acquisition_cost_eur = market_adjusted_price_eur * adjusted_percentage
        
        # 2. Dynamic refurbishing costs
        refurbishing_details = calculate_dynamic_refurbishing_cost(
            market_adjusted_price_lkr, screen_damage, backglass_damage
        )
        refurbishing_cost_eur = refurbishing_details['refurbishing_cost_eur']
        
        # 3. Market-specific logistics cost
        logistics_cost_eur = market_profile['logistics_cost_eur']
        
        # 4. Operational costs (10% of selling price)
        operational_cost_eur = final_selling_price_eur * 0.10
        
        # Calculate net profit
        total_costs = acquisition_cost_eur + refurbishing_cost_eur + logistics_cost_eur + operational_cost_eur
        net_profit_eur = final_selling_price_eur - total_costs
        
        # Store results for this market
        market_result = {
            'market': market_name,
            'market_identity': market_profile['identity'],
            'price_index': market_profile['price_index'],
            'recommended_tier': float(recommended_tier),
            'pricing_strategy': recommended_option['strategy'],
            'selling_price_eur': final_selling_price_eur,
            'net_profit_eur': round(net_profit_eur, 2),
            'cost_breakdown': {
                'acquisition_cost_eur': round(acquisition_cost_eur, 2),
                'refurbishing_cost_eur': refurbishing_cost_eur,
                'logistics_cost_eur': logistics_cost_eur,
                'operational_cost_eur': round(operational_cost_eur, 2),
                'total_costs_eur': round(total_costs, 2)
            },
            'refurbishing_tier': refurbishing_details['refurbishing_tier'],
            'market_adjusted_price_eur': round(market_adjusted_price_lkr * CURRENCY_RATES['LKR_TO_EUR'], 2)
        }
        
        market_results.append(market_result)
    
    # Sort markets by net profit (descending)
    market_results.sort(key=lambda x: x['net_profit_eur'], reverse=True)
    
    # Get the best option
    best_option = market_results[0] if market_results else None
    
    # Generate decision_id for the best option to enable feedback
    decision_id = str(uuid.uuid4())
    
    if best_option:
        # Store decision for feedback (using the best market's context)
        best_market_device_info = device_info.copy()
        best_market_device_info['market'] = best_option['market']
        best_context = prepare_input_context(best_market_device_info)
        
        active_decisions[decision_id] = {
            'context': best_context,
            'recommended_tier': best_option['recommended_tier'],
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'input_data': device_info,
            'is_multimarket': True,
            'best_market': best_option['market'],
            'selling_price_eur': best_option['selling_price_eur'],
            'acquisition_cost_eur': best_option['cost_breakdown']['acquisition_cost_eur']
        }
        # Add decision_id to best_option
        best_option['decision_id'] = decision_id
    
    return jsonify({
        'decision_id': decision_id if best_option else None,
        'device_info': device_info,
        'analysis_timestamp': datetime.now().isoformat(),
        'best_option': best_option,
        'market_analysis': market_results,
        'total_markets_analyzed': len(market_results),
        'currency_info': {
            'primary_currency': 'EUR',
            'all_costs_in_eur': True
        },
        'business_insight': {
            'highest_profit_market': best_option['market'] if best_option else None,
            'profit_range': {
                'highest': market_results[0]['net_profit_eur'] if market_results else 0,
                'lowest': market_results[-1]['net_profit_eur'] if market_results else 0
            },
            'recommendation': f"Sell in {best_option['market']} using {best_option['pricing_strategy']} strategy for maximum profit" if best_option else "No profitable market found"
        }
    })

@app.route('/price_analysis', methods=['POST'])
def price_analysis():
    """Analyze pricing for a device across all tiers"""
    data = request.get_json()
    
    # Estimate market price
    estimated_market_price_lkr = price_engine.estimate_market_price(data)
    
    # Get all pricing options
    price_options = {}
    for tier in [0.9, 1.0, 1.1]:
        prices = price_engine.calculate_recommended_prices(estimated_market_price_lkr, tier)
        price_options.update(prices)
    
    return jsonify({
        'device_specs': data,
        'estimated_market_value': {
            'eur': price_engine.convert_lkr_to_eur(estimated_market_price_lkr),
            'lkr': estimated_market_price_lkr
        },
        'pricing_analysis': price_options,
        'currency_info': {
            'primary_currency': 'EUR',
            'conversion_rate': f"1 LKR = {CURRENCY_RATES['LKR_TO_EUR']} EUR"
        }
    })

if __name__ == '__main__':
    import time
    while not initialize_models():
        print("Waiting for data...")
        time.sleep(5)
    app.run(host='0.0.0.0', port=5002)  # Different port to avoid conflicts
