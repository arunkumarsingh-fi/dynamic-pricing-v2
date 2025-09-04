#!/usr/bin/env python3
"""
Data Simulator for Full Circle Exchange
Generates 10k+ realistic transactions based on Tab 1 pricing parameters
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_simulated_transactions(num_records=10000):
    """Generate realistic iPhone resale transaction data"""
    
    # Configuration arrays
    iphone_models = [
        'iPhone 11', 'iPhone 11 Pro', 'iPhone 11 Pro Max',
        'iPhone 12', 'iPhone 12 Mini', 'iPhone 12 Pro', 'iPhone 12 Pro Max',
        'iPhone 13', 'iPhone 13 Mini', 'iPhone 13 Pro', 'iPhone 13 Pro Max',
        'iPhone 14', 'iPhone 14 Plus', 'iPhone 14 Pro', 'iPhone 14 Pro Max',
        'iPhone 15', 'iPhone 15 Plus', 'iPhone 15 Pro', 'iPhone 15 Pro Max'
    ]
    
    markets = ['romania', 'bulgaria', 'greece', 'poland', 'finland']
    inventory_levels = ['low', 'decent', 'high']
    pricing_tiers = [0.9, 1.0, 1.1]  # Competitive, Market, Premium
    
    # Base prices for iPhone models (EUR) - realistic market prices
    base_prices = {
        'iPhone 11': 320, 'iPhone 11 Pro': 420, 'iPhone 11 Pro Max': 480,
        'iPhone 12': 480, 'iPhone 12 Mini': 420, 'iPhone 12 Pro': 580, 'iPhone 12 Pro Max': 650,
        'iPhone 13': 600, 'iPhone 13 Mini': 540, 'iPhone 13 Pro': 720, 'iPhone 13 Pro Max': 780,
        'iPhone 14': 720, 'iPhone 14 Plus': 780, 'iPhone 14 Pro': 900, 'iPhone 14 Pro Max': 980,
        'iPhone 15': 880, 'iPhone 15 Plus': 940, 'iPhone 15 Pro': 1100, 'iPhone 15 Pro Max': 1200
    }
    
    # Market multipliers for logistics and demand
    market_multipliers = {
        'romania': 0.95, 'bulgaria': 0.93, 'greece': 1.02, 
        'poland': 0.98, 'finland': 1.08
    }
    
    # Logistics costs by market (EUR)
    logistics_costs = {
        'romania': 15, 'bulgaria': 18, 'greece': 25, 
        'poland': 20, 'finland': 30
    }
    
    transactions = []
    
    # Generate date range (last 18 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=540)
    
    for i in range(num_records):
        # Random device characteristics
        model = np.random.choice(iphone_models, p=get_model_weights())
        market = random.choice(markets)
        battery = np.random.normal(88, 8)  # Normal distribution around 88%
        battery = max(60, min(100, int(battery)))  # Clamp to valid range
        
        # Damage probability (20% have some damage)
        has_screen_damage = random.random() < 0.12  # 12% screen damage
        has_backglass_damage = random.random() < 0.15  # 15% back glass damage
        has_damage = has_screen_damage or has_backglass_damage
        
        # Market context
        inventory_level = np.random.choice(inventory_levels, p=[0.3, 0.5, 0.2])  # More decent inventory
        new_model_imminent = random.random() < 0.15  # 15% of time new model coming
        
        # Pricing strategy selection (AI bandit simulation)
        # More intelligent selection based on context
        if inventory_level == 'high':
            tier_probs = [0.6, 0.3, 0.1]  # More competitive when high inventory
        elif battery < 80 or has_damage:
            tier_probs = [0.5, 0.4, 0.1]  # More competitive for damaged/low battery
        else:
            tier_probs = [0.2, 0.5, 0.3]  # More premium for good condition
        
        pricing_tier = np.random.choice(pricing_tiers, p=tier_probs)
        
        # Calculate base selling price
        base_price = base_prices[model]
        market_adjusted_price = base_price * market_multipliers[market]
        
        # Apply pricing tier
        selling_price_eur = market_adjusted_price * pricing_tier
        
        # Apply condition adjustments
        if battery < 80:
            selling_price_eur *= 0.95  # 5% reduction for low battery
        if has_damage:
            selling_price_eur *= 0.88  # 12% reduction for damage (will be refurbished)
        
        # Apply market dynamics
        if new_model_imminent:
            selling_price_eur *= 0.92  # 8% reduction when new model coming
        
        if inventory_level == 'high':
            selling_price_eur *= 0.96  # 4% reduction when high inventory
        
        selling_price_eur = round(selling_price_eur, 2)
        
        # Calculate costs
        acquisition_cost_eur = selling_price_eur * 0.70  # 70% of selling price
        
        # Refurbishing costs
        if has_damage:
            if has_screen_damage and has_backglass_damage:
                refurbishing_cost_eur = selling_price_eur * 0.15  # Major refurbishing
                refurbishing_tier = 'major'
            else:
                refurbishing_cost_eur = selling_price_eur * 0.08  # Minor refurbishing
                refurbishing_tier = 'minor'
        else:
            refurbishing_cost_eur = selling_price_eur * 0.04  # Cleaning only
            refurbishing_tier = 'cleaning'
        
        logistics_cost_eur = logistics_costs[market]
        operational_cost_eur = selling_price_eur * 0.10  # 10% operational
        
        total_costs = acquisition_cost_eur + refurbishing_cost_eur + logistics_cost_eur + operational_cost_eur
        profit_eur = selling_price_eur - total_costs
        profit_margin = profit_eur / selling_price_eur if selling_price_eur > 0 else 0
        
        # Revenue is same as selling price (simplified)
        revenue_eur = selling_price_eur
        
        # Days to sell (influenced by pricing strategy and condition)
        base_days = 14  # Base 2 weeks
        if pricing_tier == 1.1:  # Premium pricing takes longer
            base_days += 7
        elif pricing_tier == 0.9:  # Competitive pricing sells faster
            base_days -= 5
            
        if has_damage:
            base_days += 3  # Damage increases time to sell
            
        days_to_sell = max(1, int(np.random.normal(base_days, 5)))  # Add some variation
        
        # Generate random transaction date
        random_days = random.randint(0, 540)
        transaction_date = start_date + timedelta(days=random_days)
        
        # Create vanilla profit for baseline comparison (simple market rate pricing)
        vanilla_selling_price = base_price * market_multipliers[market] * 1.0  # Always market rate
        vanilla_costs = (vanilla_selling_price * 0.70) + (vanilla_selling_price * 0.06) + logistics_cost_eur + (vanilla_selling_price * 0.10)
        vanilla_profit_eur = vanilla_selling_price - vanilla_costs
        
        transaction = {
            'date': transaction_date.strftime('%Y-%m-%d'),
            'model': model,
            'market': market,
            'battery_health': battery,
            'has_screen_damage': has_screen_damage,
            'has_backglass_damage': has_backglass_damage,
            'has_damage': has_damage,
            'inventory_level': inventory_level,
            'new_model_imminent': new_model_imminent,
            'pricing_tier': pricing_tier,
            'selling_price_eur': selling_price_eur,
            'acquisition_cost_eur': round(acquisition_cost_eur, 2),
            'refurbishing_cost_eur': round(refurbishing_cost_eur, 2),
            'refurbishing_tier': refurbishing_tier,
            'logistics_cost_eur': logistics_cost_eur,
            'operational_cost_eur': round(operational_cost_eur, 2),
            'total_costs_eur': round(total_costs, 2),
            'profit_eur': round(profit_eur, 2),
            'profit_margin': round(profit_margin, 4),
            'revenue_eur': revenue_eur,
            'days_to_sell': days_to_sell,
            'vanilla_profit_eur': round(vanilla_profit_eur, 2)  # Baseline comparison
        }
        
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

def get_model_weights():
    """Get probability weights for iPhone model selection (newer models more common)"""
    # Weight newer models higher (ensure they sum to 1.0)
    weights = [
        0.05, 0.04, 0.04,  # iPhone 11 series = 0.13
        0.08, 0.06, 0.09, 0.08,  # iPhone 12 series = 0.31
        0.12, 0.08, 0.14, 0.12,  # iPhone 13 series = 0.46
        0.05, 0.04, 0.07, 0.06,  # iPhone 14 series = 0.22
        0.03, 0.02, 0.04, 0.03   # iPhone 15 series = 0.12
    ]
    # Normalize to ensure sum = 1.0
    weights = [w / sum(weights) for w in weights]
    return weights

def generate_ai_feedback_simulation(num_decisions=500):
    """Generate simulated AI bandit feedback history"""
    
    feedback_history = []
    
    # Start date for feedback (last 6 months)
    start_date = datetime.now() - timedelta(days=180)
    
    for i in range(num_decisions):
        # Generate realistic decision parameters
        model = random.choice([
            'iPhone 13', 'iPhone 13 Pro', 'iPhone 14', 'iPhone 14 Pro', 'iPhone 15'
        ])
        
        market = random.choice(['romania', 'bulgaria', 'greece', 'poland', 'finland'])
        
        # AI learning simulation - gets better over time
        learning_factor = min(1.0, i / 200)  # Improves over first 200 decisions
        
        # Tier selection gets smarter over time
        if learning_factor < 0.3:
            # Early decisions - more random
            tier = random.choice([0.9, 1.0, 1.1])
            base_reward = random.uniform(20, 80)
        else:
            # Later decisions - more intelligent
            tier_probs = [0.25, 0.45, 0.30]  # Balanced but slightly favors market rate
            tier = np.random.choice([0.9, 1.0, 1.1], p=tier_probs)
            base_reward = random.uniform(40, 120)  # Better rewards as AI learns
        
        # Apply learning bonus
        reward = base_reward * (1 + learning_factor * 0.5)  # Up to 50% improvement
        
        # Add some noise
        reward += random.uniform(-15, 15)
        reward = max(5, reward)  # Minimum reward
        
        # Generate timestamp
        days_offset = int(i * (180 / num_decisions))  # Spread evenly over 6 months
        timestamp = (start_date + timedelta(days=days_offset)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Sale outcome distribution
        outcomes = ['Device Sold', 'Device Sold', 'Device Sold', 'Still in Inventory', 'Price Reduced']
        sale_outcome = random.choice(outcomes)
        
        feedback = {
            'timestamp': timestamp,
            'decision_id': f'sim_decision_{i+1:04d}',
            'tier': tier,
            'reward': round(reward, 2),
            'sale_outcome': sale_outcome,
            'features': {
                'Model': model,
                'market': market,
                'Battery': random.randint(75, 100),
                'inventory_level': random.choice(['low', 'decent', 'high']),
                'Backglass_Damage': random.choice([0, 1]),
                'Screen_Damage': random.choice([0, 1]),
                'new_model_imminent': random.choice([True, False])
            }
        }
        
        feedback_history.append(feedback)
    
    return feedback_history

def main():
    """Generate and save simulated data"""
    print("🔄 Generating simulated transaction data...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate 10k transaction records
    df = generate_simulated_transactions(10000)
    
    # Save to CSV
    output_file = 'data/analytics_data.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Generated {len(df):,} transaction records -> {output_file}")
    
    # Generate AI feedback simulation
    feedback_data = generate_ai_feedback_simulation(500)
    
    # Save feedback history
    feedback_file = 'data/ai_feedback_history.json'
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    print(f"✅ Generated {len(feedback_data):,} AI feedback decisions -> {feedback_file}")
    
    # Print summary statistics
    print("\n📊 Data Summary:")
    print(f"📈 Total Revenue: €{df['revenue_eur'].sum():,.0f}")
    print(f"💰 Total Profit: €{df['profit_eur'].sum():,.0f}")
    print(f"📱 Units Sold: {len(df):,}")
    print(f"💹 Avg Profit/Unit: €{df['profit_eur'].mean():.0f}")
    print(f"⏱️ Avg Days to Sell: {df['days_to_sell'].mean():.1f}")
    print(f"📊 Avg Profit Margin: {df['profit_margin'].mean():.1%}")
    
    # Model distribution
    print(f"\n🏆 Top 5 Models by Volume:")
    model_counts = df['model'].value_counts().head()
    for model, count in model_counts.items():
        print(f"  {model}: {count:,} units")
    
    # Market distribution
    print(f"\n🌍 Market Distribution:")
    market_counts = df['market'].value_counts()
    for market, count in market_counts.items():
        print(f"  {market.title()}: {count:,} units")

if __name__ == "__main__":
    main()
