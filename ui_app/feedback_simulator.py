#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def create_realistic_feedback_history(num_records=50):
    """
    Create a realistic feedback history that simulates the user having 
    used the system over the past few weeks with improving performance
    """
    
    np.random.seed(42)
    random.seed(42)
    
    feedback_records = []
    
    # Simulate learning curve - performance improves over time
    for i in range(num_records):
        # Time progression (older records first)
        days_ago = num_records - i
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        # Learning improvement factor - better performance over time
        learning_factor = min(1.2, 1.0 + (i / num_records) * 0.4)  # 0-40% improvement
        
        # Realistic business outcomes with evolution
        if i < num_records * 0.2:  # Early period - more mixed results
            sale_outcomes = ['Device Sold', 'Still in Inventory', 'Price Reduced', 'Returned/Exchanged']
            outcome_weights = [0.60, 0.25, 0.12, 0.03]
        elif i < num_records * 0.6:  # Middle period - getting better
            outcome_weights = [0.70, 0.18, 0.10, 0.02]
        else:  # Recent period - much better performance
            outcome_weights = [0.80, 0.12, 0.06, 0.02]
        
        sale_outcome = random.choices(sale_outcomes, weights=outcome_weights)[0]
        
        # Simulate reward based on outcome and learning
        base_reward_by_outcome = {
            'Device Sold': np.random.normal(40, 18),
            'Price Reduced': np.random.normal(20, 12),
            'Still in Inventory': np.random.normal(-8, 4),
            'Returned/Exchanged': np.random.normal(-18, 6)
        }
        
        base_reward = base_reward_by_outcome[sale_outcome]
        
        # Apply learning factor
        if sale_outcome == 'Device Sold':
            reward = base_reward * learning_factor
            # Add time penalty occasionally
            if random.random() < 0.3:
                time_penalty = random.randint(2, 8)
                reward -= time_penalty
        else:
            reward = base_reward  # Negative outcomes don't benefit as much from learning
        
        # Pricing tier distribution that evolves over time
        if i < num_records * 0.3:  # Early: more random
            tier_weights = [0.4, 0.3, 0.3]
        elif i < num_records * 0.7:  # Middle: learning optimal balance
            tier_weights = [0.35, 0.4, 0.25]
        else:  # Recent: more strategic
            tier_weights = [0.3, 0.45, 0.25]
        
        tiers = [0.9, 1.0, 1.1]
        tier = random.choices(tiers, weights=tier_weights)[0]
        
        # Simulate realistic device features for context
        features = {
            'Model': random.choice([
                'iPhone 13', 'iPhone 14', 'iPhone 14 Pro', 'iPhone 15', 
                'iPhone 15 Pro', 'iPhone 12', 'iPhone 13 Pro'
            ]),
            'Battery': random.randint(75, 98),
            'inventory_level': random.choice(['low', 'decent', 'high']),
            'market': random.choice(['romania', 'poland', 'finland', 'greece', 'bulgaria'])
        }
        
        feedback_record = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'decision_id': f'sim_{i+1:04d}',
            'tier': tier,
            'reward': round(reward, 2),
            'sale_outcome': sale_outcome,
            'features': features
        }
        
        feedback_records.append(feedback_record)
    
    return feedback_records

def load_or_create_feedback_history():
    """
    Load existing feedback history or create a new realistic one for demonstration
    """
    import json
    
    # First, try to load the comprehensive AI feedback history (500 records)
    comprehensive_feedback_file = 'data/ai_feedback_history.json'
    if os.path.exists(comprehensive_feedback_file):
        with open(comprehensive_feedback_file, 'r') as f:
            feedback_history = json.load(f)
        print(f"📚 Loaded comprehensive AI feedback history: {len(feedback_history)} records")
        return feedback_history
    
    # Fallback to smaller demo feedback
    demo_feedback_file = 'data/demo_feedback_history.json'
    if os.path.exists(demo_feedback_file):
        with open(demo_feedback_file, 'r') as f:
            feedback_history = json.load(f)
        print(f"📚 Loaded demo feedback history: {len(feedback_history)} records")
        return feedback_history
    
    # Create new realistic history
    feedback_history = create_realistic_feedback_history(50)
    
    # Save for future use
    os.makedirs('data', exist_ok=True)
    with open(demo_feedback_file, 'w') as f:
        json.dump(feedback_history, f, indent=2)
    print(f"✨ Created new demo feedback history: {len(feedback_history)} records")
    
    return feedback_history

def enhance_streamlit_session_with_demo_data(st):
    """
    Enhance the Streamlit session state with realistic demo feedback data
    This can be called from the Streamlit app to populate Tab 3 with demo data
    """
    
    # Only create demo data if no real feedback exists
    if 'feedback_history' not in st.session_state or len(st.session_state.get('feedback_history', [])) < 5:
        
        demo_feedback = load_or_create_feedback_history()
        
        # Add to session state
        st.session_state['feedback_history'] = demo_feedback
        
        return True, len(demo_feedback)
    
    return False, len(st.session_state.get('feedback_history', []))

if __name__ == "__main__":
    # Test the feedback simulator
    print("🧪 Testing Feedback Simulator")
    print("=" * 40)
    
    # Create demo feedback history
    feedback_history = create_realistic_feedback_history(50)
    
    # Analyze the generated data
    df = pd.DataFrame(feedback_history)
    
    print(f"📊 Generated {len(df)} feedback records")
    print(f"💰 Average reward: €{df['reward'].mean():.1f}")
    print(f"📈 Reward range: €{df['reward'].min():.1f} to €{df['reward'].max():.1f}")
    print(f"🎯 Total cumulative profit: €{df['reward'].sum():.1f}")
    
    # Show outcome distribution
    outcome_dist = df['sale_outcome'].value_counts(normalize=True)
    print(f"\n📦 Sale Outcome Distribution:")
    for outcome, pct in outcome_dist.items():
        print(f"  {outcome}: {pct:.1%}")
    
    # Show strategy distribution  
    strategy_dist = df['tier'].value_counts(normalize=True)
    print(f"\n⚖️ Pricing Strategy Distribution:")
    tier_names = {0.9: 'Competitive', 1.0: 'Market Rate', 1.1: 'Premium'}
    for tier, pct in strategy_dist.items():
        print(f"  {tier_names[tier]} ({tier}): {pct:.1%}")
    
    # Show learning progression
    print(f"\n📈 Learning Progression:")
    early_avg = df.head(15)['reward'].mean()
    recent_avg = df.tail(15)['reward'].mean()
    improvement = ((recent_avg - early_avg) / early_avg) * 100
    print(f"  Early performance (first 15): €{early_avg:.1f}")
    print(f"  Recent performance (last 15): €{recent_avg:.1f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    print(f"\n✅ Feedback simulator working correctly!")
    print(f"📁 Demo data will be saved to: data/demo_feedback_history.json")
