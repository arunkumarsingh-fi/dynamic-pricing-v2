import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import numpy as np

# Initialize API base URL for V2 (separate ML backend)
api_base = os.getenv('ML_API_URL', 'http://localhost:5003')

# Configure page
st.set_page_config(
    page_title="Optimized Business - Full Circle Exchange V2",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS for progressive maturity levels
st.markdown("""
<style>
    .maturity-level {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        margin: 2rem 0 1rem 0;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    
    .price-output {
        background: #e8f5e8;
        border: 2px solid #27ae60;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #27ae60;
        margin: 10px 0;
    }
    
    .plus-button {
        background: #3498db;
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 2rem;
        font-weight: bold;
        cursor: pointer;
        margin: 2rem auto;
        display: block;
    }
    
    .plus-button:hover {
        background: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🔄 Navigation")
page = st.sidebar.selectbox(
    "Select Page:",
    ["Page 1: Story of a Unit", "Page 2: Optimized Business"],
    index=0
)

# Initialize session state
if 'show_level2' not in st.session_state:
    st.session_state.show_level2 = False
if 'show_level3' not in st.session_state:
    st.session_state.show_level3 = False  
if 'show_full_matured' not in st.session_state:
    st.session_state.show_full_matured = False
if 'current_payload' not in st.session_state:
    st.session_state.current_payload = {}
if 'show_enhanced_analytics' not in st.session_state:
    st.session_state.show_enhanced_analytics = False

# =================================================================
# PAGE 1: STORY OF A UNIT
# =================================================================
if page == "Page 1: Story of a Unit":
    # Main header
    st.title('🔄 Story of a Unit')
    st.markdown("### 📱 From Simple to Sophisticated")
    
    # ================================
    # MATURITY LEVEL 1
    # ================================
    st.markdown('<div class="maturity-level">Maturity Level 1</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # iPhone model selection
        iphone_models = [
            'iPhone 11', 'iPhone 11 Pro', 'iPhone 11 Pro Max',
            'iPhone 12', 'iPhone 12 Mini', 'iPhone 12 Pro', 'iPhone 12 Pro Max',
            'iPhone 13', 'iPhone 13 Mini', 'iPhone 13 Pro', 'iPhone 13 Pro Max',
            'iPhone 14', 'iPhone 14 Plus', 'iPhone 14 Pro', 'iPhone 14 Pro Max',
            'iPhone 15', 'iPhone 15 Plus', 'iPhone 15 Pro', 'iPhone 15 Pro Max'
        ]
        
        selected_model_l1 = st.selectbox(
            'iPhone model',
            iphone_models,
            index=iphone_models.index('iPhone 13 Pro'),
            key="model_l1"
        )
        
        # Battery Health slider
        battery_health_l1 = st.slider(
            'Battery Health',
            min_value=60,
            max_value=100,
            value=95,
            key="battery_l1",
            help="Current battery health percentage"
        )
    
    with col2:
        st.markdown("### Level 1 Predictions")
        
        # Store Level 1 data
        st.session_state.current_payload = {
            'Model': selected_model_l1,
            'Battery': battery_health_l1,
            'Screen_Damage': 0,
            'Backglass_Damage': 0,
            'market': 'poland',
            'inventory_level': 'decent',
            'new_model_imminent': False
        }
        
        # Predict Buying Price button
        if st.button("Predict Buying Price", key="buy_l1", use_container_width=True):
            try:
                response = requests.post(f'{api_base}/recommend_price', json=st.session_state.current_payload, timeout=10)
                result = response.json()
                buying_price = result.get('target_acquisition_cost', {}).get('eur', 0)
                st.session_state['buy_price_l1'] = buying_price
                st.rerun()
            except Exception as e:
                st.error(f"API Error: {str(e)}")
        
        if 'buy_price_l1' in st.session_state:
            st.markdown(f'<div class="price-output">€{st.session_state.buy_price_l1:.2f}</div>', unsafe_allow_html=True)
        
        # Predict Selling Price button
        if st.button("Predict Selling Price", key="sell_l1", use_container_width=True):
            try:
                response = requests.post(f'{api_base}/recommend_price', json=st.session_state.current_payload, timeout=10)
                result = response.json()
                selling_price = result.get('recommended_price_eur', 0)
                st.session_state['sell_price_l1'] = selling_price
                st.rerun()
            except Exception as e:
                st.error(f"API Error: {str(e)}")
        
        if 'sell_price_l1' in st.session_state:
            st.markdown(f'<div class="price-output">€{st.session_state.sell_price_l1:.2f}</div>', unsafe_allow_html=True)
    
    # Plus button for Level 2
    if 'sell_price_l1' in st.session_state and 'buy_price_l1' in st.session_state:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("➕", key="expand_l2", help="Expand to Maturity Level 2"):
                st.session_state.show_level2 = True
                st.rerun()
    
    # ================================
    # MATURITY LEVEL 2
    # ================================
    if st.session_state.show_level2:
        st.markdown('<div class="maturity-level">Maturity Level 2</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"Building on: {selected_model_l1} with {battery_health_l1}% battery")
            
            # Screen condition
            screen_condition_l2 = st.selectbox(
                'Screen condition',
                ['Undamaged', 'Minor Scratches', 'Cracked', 'Severely Damaged'],
                key="screen_l2"
            )
            
            # Back glass condition  
            back_condition_l2 = st.selectbox(
                'Back glass condition',
                ['No Damage', 'Minor Wear', 'Scratched', 'Damaged'],
                key="back_l2"
            )
            
            # Current Inventory Level
            inventory_l2 = st.selectbox(
                'Current Inventory Level',
                ['low', 'decent', 'high'],
                index=1,
                key="inventory_l2"
            )
            
            # New Release imminent
            new_release_l2 = st.toggle(
                'New Release imminent?',
                help="Is a new iPhone model expected to be released soon?",
                key="release_l2"
            )
        
        with col2:
            st.markdown("### Level 2 Predictions")
            
            # Convert conditions for API
            screen_damage = 1 if screen_condition_l2 in ['Cracked', 'Severely Damaged'] else 0
            back_damage = 1 if back_condition_l2 in ['Damaged'] else 0
            
            # Update payload with Level 2 data
            st.session_state.current_payload.update({
                'Screen_Damage': screen_damage,
                'Backglass_Damage': back_damage,
                'inventory_level': inventory_l2,
                'new_model_imminent': new_release_l2
            })
            
            # Predict Buying Price L2
            if st.button("Predict Buying Price", key="buy_l2", use_container_width=True):
                try:
                    response = requests.post(f'{api_base}/recommend_price', json=st.session_state.current_payload, timeout=10)
                    result = response.json()
                    buying_price = result.get('target_acquisition_cost', {}).get('eur', 0)
                    st.session_state['buy_price_l2'] = buying_price
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
            
            if 'buy_price_l2' in st.session_state:
                st.markdown(f'<div class="price-output">€{st.session_state.buy_price_l2:.2f}</div>', unsafe_allow_html=True)
            
            # Predict Selling Price L2
            if st.button("Predict Selling Price", key="sell_l2", use_container_width=True):
                try:
                    response = requests.post(f'{api_base}/recommend_price', json=st.session_state.current_payload, timeout=10)
                    result = response.json()
                    selling_price = result.get('recommended_price_eur', 0)
                    st.session_state['sell_price_l2'] = selling_price
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
            
            if 'sell_price_l2' in st.session_state:
                st.markdown(f'<div class="price-output">€{st.session_state.sell_price_l2:.2f}</div>', unsafe_allow_html=True)
        
        # Plus button for Level 3
        if 'sell_price_l2' in st.session_state and 'buy_price_l2' in st.session_state:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("➕", key="expand_l3", help="Expand to Maturity Level 3"):
                    st.session_state.show_level3 = True
                    st.rerun()
    
    # ================================
    # MATURITY LEVEL 3  
    # ================================
    if st.session_state.show_level3:
        st.markdown('<div class="maturity-level">Maturity Level 3</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Action A: Single Market Analysis")
            
            # Target Market
            target_market_l3 = st.selectbox(
                'Target Market',
                ['Romania', 'Bulgaria', 'Greece', 'Poland', 'Finland'],
                index=3,
                key="market_l3"
            )
            
            # Update payload for specific market
            market_payload = st.session_state.current_payload.copy()
            market_payload['market'] = target_market_l3.lower()
            
            # Generate price for selected market
            if st.button("Generate price for selected market", key="single_market", use_container_width=True):
                try:
                    response = requests.post(f'{api_base}/recommend_price', json=market_payload, timeout=10)
                    result = response.json()
                    st.session_state['single_market_result'] = result
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
        
        with col2:
            st.markdown("### Action B: Multi Market Optimization")
            st.markdown("Find best market and Price")
            
            # Multi-market optimization
            if st.button("Find Best Market & Price", key="multi_market", use_container_width=True, type="primary"):
                try:
                    response = requests.post(f'{api_base}/optimize_market_and_price', json=st.session_state.current_payload, timeout=15)
                    result = response.json()
                    st.session_state['multi_market_result'] = result
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
        
        # Display results (same format as V1)
        if 'single_market_result' in st.session_state or 'multi_market_result' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Show single market result
            if 'single_market_result' in st.session_state:
                result = st.session_state['single_market_result']
                
                # Calculate total costs properly for single market
                price_eur = result.get('recommended_price_eur', 0)
                acquisition_cost = result.get('target_acquisition_cost', {}).get('eur', 0)
                refurb_cost = result.get('cost_breakdown', {}).get('refurbishing_cost_eur', price_eur * 0.05)
                logistics_cost = 0.0  # Single market has no logistics cost
                operational_cost = price_eur * 0.10  # 10%
                total_costs = acquisition_cost + refurb_cost + logistics_cost + operational_cost
                correct_profit = price_eur - total_costs
                
                # Store for consistent use
                st.session_state['single_market_costs'] = {
                    'acquisition_cost': acquisition_cost,
                    'refurb_cost': refurb_cost,
                    'logistics_cost': logistics_cost,
                    'operational_cost': operational_cost,
                    'total_costs': total_costs
                }
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Selling Price", f"€{price_eur:.2f}")
                with col2:
                    st.metric("Total Costs", f"€{total_costs:.2f}")
                with col3:
                    st.metric("Net Profit", f"€{correct_profit:.2f}")
                with col4:
                    strategy = result.get('pricing_strategy', 'Market Rate')
                    st.metric("Strategy", strategy)
            
            # Show multi-market result
            if 'multi_market_result' in st.session_state:
                result = st.session_state['multi_market_result']
                
                if 'best_option' in result:
                    best = result['best_option']
                    st.success(f"🎯 Best Market: {best['market']} - €{best['selling_price_eur']:.2f}")
                    
                    # Show Analysis Results for multi-market with correct costs
                    price_eur = best.get('selling_price_eur', 0)
                    costs = best.get('cost_breakdown', {})
                    total_costs = costs.get('total_costs_eur', 0)
                    correct_profit = best.get('net_profit_eur', 0)
                    
                    # Store multi-market costs
                    st.session_state['multi_market_costs'] = {
                        'acquisition_cost': costs.get('acquisition_cost_eur', 0),
                        'refurb_cost': costs.get('refurbishing_cost_eur', 0),
                        'logistics_cost': costs.get('logistics_cost_eur', 0),
                        'operational_cost': costs.get('operational_cost_eur', 0),
                        'total_costs': total_costs
                    }
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Selling Price", f"€{price_eur:.2f}")
                    with col2:
                        st.metric("Total Costs", f"€{total_costs:.2f}")
                    with col3:
                        st.metric("Net Profit", f"€{correct_profit:.2f}")
                    with col4:
                        strategy = best.get('pricing_strategy', 'Market Rate')
                        st.metric("Strategy", strategy)
                    
                    # Show cost breakdown for best market
                    st.subheader("💸 Cost Breakdown (Best Market)")
                    costs = best.get('cost_breakdown', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🛒 Acquisition", f"€{costs.get('acquisition_cost_eur', 0):.2f}")
                    with col2:
                        st.metric("🔧 Refurbishing", f"€{costs.get('refurbishing_cost_eur', 0):.2f}")
                    with col3:
                        st.metric("🚚 Logistics", f"€{costs.get('logistics_cost_eur', 0):.2f}")
                    with col4:
                        st.metric("⚙️ Operational", f"€{costs.get('operational_cost_eur', 0):.2f}")
                    
                    # Markets comparison table
                    st.subheader("🌍 Complete Market Analysis")
                    markets_data = []
                    for market_info in result['market_analysis']:
                        cost_breakdown = market_info.get('cost_breakdown', {})
                        acquisition_cost = cost_breakdown.get('acquisition_cost_eur', 0)
                        refurbishing_cost = cost_breakdown.get('refurbishing_cost_eur', 0)
                        logistics_cost = cost_breakdown.get('logistics_cost_eur', 0)
                        operational_cost = cost_breakdown.get('operational_cost_eur', 0)
                        total_costs = cost_breakdown.get('total_costs_eur', acquisition_cost + refurbishing_cost + logistics_cost + operational_cost)
                        
                        markets_data.append({
                            'Market': market_info['market'],
                            'Selling Price': f"€{market_info['selling_price_eur']:.2f}",
                            'Acquisition': f"€{acquisition_cost:.2f}",
                            'Refurb': f"€{refurbishing_cost:.2f}",
                            'Logistics': f"€{logistics_cost:.2f}",
                            'Operational': f"€{operational_cost:.2f}",
                            'Total Costs': f"€{total_costs:.2f}",
                            'Net Profit': f"€{market_info['net_profit_eur']:.2f}",
                            'Strategy': market_info['pricing_strategy']
                        })
                    
                    df_markets = pd.DataFrame(markets_data)
                    st.dataframe(df_markets, use_container_width=True)
            
            # Plus button for Full Matured Flow
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("➕", key="expand_full", help="Expand to Full Matured Flow"):
                    st.session_state.show_full_matured = True
                    st.rerun()
    
    # ================================
    # FULL MATURED FLOW
    # ================================
    if st.session_state.show_full_matured:
        st.markdown('<div class="maturity-level">Full Matured Flow</div>', unsafe_allow_html=True)
        
        # Get the most recent result for business analysis
        if 'multi_market_result' in st.session_state:
            result = st.session_state['multi_market_result']['best_option']
            price_eur = result.get('selling_price_eur', 0)
            cost_breakdown = result.get('cost_breakdown', {})
            total_costs = cost_breakdown.get('total_costs_eur', 0)
            acquisition_cost = cost_breakdown.get('acquisition_cost_eur', 0)
            refurb_cost = cost_breakdown.get('refurbishing_cost_eur', 0)
            logistics_cost = cost_breakdown.get('logistics_cost_eur', 0)
            operational_cost = cost_breakdown.get('operational_cost_eur', 0)
            target_cost = acquisition_cost
        elif 'single_market_result' in st.session_state:
            result = st.session_state['single_market_result']
            price_eur = result.get('recommended_price_eur', 0)
            target_cost = result.get('target_acquisition_cost', {}).get('eur', 0)
            single_costs = st.session_state.get('single_market_costs', {})
            if single_costs:
                total_costs = single_costs.get('total_costs', 0)
                acquisition_cost = single_costs.get('acquisition_cost', 0)
                refurb_cost = single_costs.get('refurb_cost', 0)
                logistics_cost = single_costs.get('logistics_cost', 0.0)  # Single market has no logistics
                operational_cost = single_costs.get('operational_cost', 0)
            else:
                acquisition_cost = target_cost
                refurb_cost = price_eur * 0.05
                logistics_cost = 0.0  # Single market has no logistics cost
                operational_cost = price_eur * 0.10
                total_costs = acquisition_cost + refurb_cost + logistics_cost + operational_cost
        else:
            st.warning("⚠️ Please complete Level 3 analysis first.")
            st.stop()
        
        # Store consistent results
        tier = result.get('recommended_tier', 1.0)
        strategy = result.get('pricing_strategy', 'Market Rate')
        condition_score = 95.0
        
        # Main recommendation display
        col1, col2 = st.columns(2)
        
        with col1:
            if tier == 0.9:
                st.error(f"🔻 **{strategy} Pricing** (Tier {tier})")
            elif tier == 1.0:
                st.success(f"🎯 **{strategy} Pricing** (Tier {tier})")
            else:
                st.warning(f"🔺 **{strategy} Pricing** (Tier {tier})")
            
            st.markdown(f"**Recommended Selling Price:** €{price_eur:.2f}")
            st.markdown(f"**Market Segment:** High-end")
            st.markdown(f"**Condition Score:** {condition_score:.1f}%")
        
        with col2:
            # All pricing options display
            st.markdown("**All Pricing Options:**")
            pricing_options = result.get('all_pricing_options', {})
            
            if not pricing_options:
                pricing_options = {
                    '0.9': {'price_eur': price_eur * 0.9, 'strategy': 'Competitive', 'recommended': tier == 0.9},
                    '1.0': {'price_eur': price_eur, 'strategy': 'Market Rate', 'recommended': tier == 1.0},
                    '1.1': {'price_eur': price_eur * 1.1, 'strategy': 'Premium', 'recommended': tier == 1.1}
                }
            
            for tier_val, details in pricing_options.items():
                recommended_icon = "🎯" if details.get('recommended', False) else ""
                st.markdown(f"{recommended_icon} **{details['strategy']}**: €{details['price_eur']:.2f}")
        
        st.markdown("---")
        
        # Target Acquisition Cost Section
        st.subheader("🎯 Target Acquisition Cost")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Costs", f"€{total_costs:.2f}")
        with col2:
            st.metric("Expected Selling Price", f"€{price_eur:.2f}")
        with col3:
            correct_profit = price_eur - total_costs
            st.metric("Expected Profit", f"€{correct_profit:.2f}")
        
        st.markdown("---")
        
        # Time Value Analysis Section
        st.subheader("📈 Time Value Analysis")
        st.markdown("**Profit decay over time if device remains unsold:**")
        
        days = np.arange(1, 91)  # 90 days
        time_decay = np.where(days <= 7, 1.0, 1.0 - (days - 7) * 0.005)
        time_decay = np.maximum(time_decay, 0.7)
        
        adjusted_prices = price_eur * time_decay
        adjusted_profits = adjusted_prices - total_costs
        
        # Create the decay chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days,
            y=adjusted_prices,
            name='Selling Price (€)',
            line=dict(color='blue', width=3),
            hovertemplate='Day %{x}<br>Price: €%{y:.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=days,
            y=adjusted_profits,
            name='Net Profit (€)',
            line=dict(color='green', width=3),
            hovertemplate='Day %{x}<br>Profit: €%{y:.0f}<extra></extra>'
        ))
        
        fig.add_hline(y=total_costs, line_dash="dash", 
                     line_color="red", annotation_text="Break-even")
        
        fig.update_layout(
            title='Price & Profit Decay Over Time',
            xaxis_title='Days to Sell',
            yaxis_title='Amount (€)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time-based recommendations
        st.markdown("**📅 Time-Based Pricing Strategy**")
        quick_sale_days = 7
        medium_sale_days = 21
        slow_sale_days = 60
        
        quick_price = price_eur * time_decay[quick_sale_days - 1]
        medium_price = price_eur * time_decay[medium_sale_days - 1] 
        slow_price = price_eur * time_decay[slow_sale_days - 1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quick_profit = quick_price - total_costs
            st.success(f"🚀 **Quick Sale (1-7 days)**")
            st.write(f"Price: €{quick_price:.2f}")
            st.write(f"Profit: €{quick_profit:.2f}")
            
        with col2:
            medium_profit = medium_price - total_costs
            st.info(f"🔄 **Standard Sale (~{medium_sale_days} days)**")
            st.write(f"Price: €{medium_price:.2f}")
            st.write(f"Profit: €{medium_profit:.2f}")
            
        with col3:
            slow_profit = slow_price - total_costs
            st.warning(f"🐌 **Slow Sale (~{slow_sale_days} days)**")
            st.write(f"Price: €{slow_price:.2f}")
            st.write(f"Profit: €{slow_profit:.2f}")
        
        st.info("💡 **Key Insight**: Every day after the first week reduces your profit by approximately €1-2 due to market depreciation and holding costs.")
        
        st.markdown("---")
        
        # Manual Price Override Section
        st.subheader("💰 Price Override (Optional)")
        col1, col2 = st.columns(2)
        
        with col1:
            custom_buying_price = st.number_input(
                '🛒 Custom Buying Price (EUR)', 
                min_value=0.0, 
                value=0.0, 
                step=10.0,
                help="Override the recommended buying price (0 = use AI recommendation)"
            )
        
        with col2:
            custom_selling_price = st.number_input(
                '💵 Custom Selling Price (EUR)', 
                min_value=0.0, 
                value=0.0, 
                step=10.0,
                help="Override the recommended selling price (0 = use AI recommendation)"
            )
        
        # Apply manual overrides if set
        final_selling_price = custom_selling_price if custom_selling_price > 0 else price_eur
        final_buying_price = custom_buying_price if custom_buying_price > 0 else target_cost
        
        # Store for session compatibility
        st.session_state['decision_id'] = result.get('decision_id')
        st.session_state['tier'] = tier
        st.session_state['final_prices'] = {
            'selling': final_selling_price,
            'buying': final_buying_price
        }
        
        st.markdown("---")
        
        # Send Feedback to Advanced AI
        st.subheader("🤖 Send Feedback to Advanced AI")
        st.markdown("**Help the AI learn by reporting the actual outcome of this pricing decision.**")
        
        # Calculate costs based on available data for consistent use
        try:
            if 'multi_market_result' in st.session_state:
                result_data = st.session_state['multi_market_result']['best_option']
                cost_breakdown = result_data.get('cost_breakdown', {})
                breakdown_acquisition = cost_breakdown.get('acquisition_cost_eur', target_cost if 'target_cost' in locals() else 150)
                breakdown_refurb = cost_breakdown.get('refurbishing_cost_eur', 15)
                breakdown_logistics = cost_breakdown.get('logistics_cost_eur', 25)
                breakdown_operational = cost_breakdown.get('operational_cost_eur', 35)
            elif 'single_market_result' in st.session_state:
                breakdown_acquisition = target_cost if 'target_cost' in locals() else 150
                breakdown_refurb = price_eur * 0.05 if 'price_eur' in locals() else 15
                breakdown_logistics = 0.0  # Single market
                breakdown_operational = price_eur * 0.10 if 'price_eur' in locals() else 35
            else:
                # Fallback sample values
                breakdown_acquisition = 158.31
                breakdown_refurb = 14.04
                breakdown_logistics = 0.0
                breakdown_operational = 28.09
        except:
            # If any error, use sample values
            breakdown_acquisition = 158.31
            breakdown_refurb = 14.04
            breakdown_logistics = 0.0
            breakdown_operational = 28.09
        
        # Total costs for calculations
        total_costs_complete = breakdown_acquisition + breakdown_refurb + breakdown_logistics + breakdown_operational
        selling_price_for_calc = final_selling_price if 'final_selling_price' in locals() else price_eur
        true_expected_profit = selling_price_for_calc - total_costs_complete
        
        # Cost breakdown right before feedback for accurate profit calculation
        st.subheader("💸 Complete Cost Analysis for Feedback")
        st.markdown("**Review all costs before reporting actual outcomes:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🛒 Acquisition", f"€{breakdown_acquisition:.2f}")
        with col2:
            st.metric("🔧 Refurbishing", f"€{breakdown_refurb:.2f}")
        with col3:
            st.metric("🚚 Logistics", f"€{breakdown_logistics:.2f}")
        with col4:
            st.metric("⚙️ Operational", f"€{breakdown_operational:.2f}")
        
        # Show the true profit calculation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Expected Selling Price", f"€{selling_price_for_calc:.2f}")
        with col2:
            st.metric("📊 Total All Costs", f"€{total_costs_complete:.2f}")
        with col3:
            st.metric("🎯 True Expected Profit", f"€{true_expected_profit:.2f}", 
                     help="Selling Price - (Acquisition + Refurbishing + Logistics + Operational)")
        
        st.warning("⚠️ **Important**: The 'True Expected Profit' above accounts for ALL costs. Use this for accurate reward calculations.")
        st.markdown("---")
        
        # Feedback inputs
        col1, col2 = st.columns(2)
        
        with col1:
            actual_sale_outcome = st.selectbox(
                "📈 Actual Sale Outcome",
                ["Device Sold", "Still in Inventory", "Price Reduced", "Returned to Supplier"],
                help="What actually happened with this device?"
            )
            
            actual_selling_price = st.number_input(
                "💰 Actual Selling Price (EUR)",
                min_value=0.0,
                value=final_selling_price,
                step=5.0,
                help="The actual price you sold the device for"
            )
        
        with col2:
            actual_buying_price = st.number_input(
                "🛒 Actual Buying Price (EUR)",
                min_value=0.0, 
                value=final_buying_price,
                step=5.0,
                help="The actual price you bought the device for"
            )
            
            days_to_sell = st.number_input(
                "📅 Days to Sell",
                min_value=0,
                value=14,
                step=1,
                help="How many days did it take to sell the device?"
            )
        
        # Calculate CORRECT actual profit using all costs
        if actual_sale_outcome == "Device Sold":
            # Use the actual breakdown costs calculated above, but replace acquisition with actual buying price
            actual_total_costs = actual_buying_price + breakdown_refurb + breakdown_logistics + breakdown_operational
            actual_profit_complete = actual_selling_price - actual_total_costs
        else:
            actual_profit_complete = 0
        
        # Show calculation breakdown for transparency
        st.info(f"💡 **Actual Profit Calculation**: €{actual_selling_price:.2f} (selling) - €{actual_buying_price:.2f} (buying) - €{breakdown_refurb:.2f} (refurb) - €{breakdown_logistics:.2f} (logistics) - €{breakdown_operational:.2f} (operational) = €{actual_profit_complete:.2f}")
        
        # Smart reward calculation using complete profit
        time_penalty = max(0, (days_to_sell - 14) * 2)
        smart_reward = actual_profit_complete - time_penalty
        
        # Manual reward override
        manual_reward = st.number_input(
            "🎯 Manual Reward Override (EUR)",
            value=smart_reward,
            step=1.0,
            help="Override the calculated reward if needed"
        )
        
        st.markdown("---")
        
        # Send feedback button
        if st.button('🚀 **SEND FEEDBACK TO AI BANDIT**', type="primary", use_container_width=True):
            if st.session_state.get('decision_id'):
                feedback_payload = {
                    'decision_id': st.session_state['decision_id'],
                    'actual_reward': manual_reward,
                    'sale_outcome': actual_sale_outcome
                }
                
                try:
                    response = requests.post(f'{api_base}/report_outcome', json=feedback_payload, timeout=10)
                    if response.status_code == 200:
                        st.success("✅ Feedback sent! The AI bandit has learned from this outcome.")
                        
                        # Store feedback in history
                        if 'feedback_history' not in st.session_state:
                            st.session_state['feedback_history'] = []
                        st.session_state['feedback_history'].append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'decision_id': st.session_state['decision_id'],
                            'tier': st.session_state['tier'],
                            'reward': manual_reward,
                            'sale_outcome': actual_sale_outcome,
                            'actual_profit': actual_profit_complete,
                            'days_to_sell': days_to_sell
                        })
                    else:
                        st.error("❌ Failed to send feedback. Please check the API connection.")
                except Exception as e:
                    st.error(f"❌ Feedback error: {str(e)}")
            else:
                st.warning("⚠️ No pricing decision available to provide feedback on.")

# =================================================================
# PAGE 2: OPTIMIZED BUSINESS
# =================================================================
elif page == "Page 2: Optimized Business":
    # Main header
    st.title('🚀 Optimized Business')
    st.markdown("### 📊 Enhanced Analytics Dashboard")
    
    # Create enhanced business analytics with CEO improvements
    try:
        # Generate 5000 synthetic records for demo if not exists
        analytics_data = []
        from datetime import datetime, timedelta
        import random
        
        if 'enhanced_analytics_data' not in st.session_state:
            with st.spinner("📊 Generating 5,000 synthetic business records..."):
                base_date = datetime.now() - timedelta(days=500)
                for i in range(5000):
                    date = base_date + timedelta(days=random.randint(0, 365))
                    
                    # iPhone models with realistic distribution
                    models = ['iPhone 12', 'iPhone 13', 'iPhone 13 Pro', 'iPhone 14', 'iPhone 14 Pro', 'iPhone 15']
                    model = random.choice(models)
                    
                    # Battery health
                    battery_health = random.randint(70, 100)
                    
                    # Market distribution
                    markets = ['romania', 'poland', 'bulgaria', 'greece', 'finland']
                    market = random.choice(markets)
                    
                    # Pricing based on model and condition
                    base_prices = {
                        'iPhone 12': 300, 'iPhone 13': 400, 'iPhone 13 Pro': 500,
                        'iPhone 14': 600, 'iPhone 14 Pro': 700, 'iPhone 15': 800
                    }
                    
                    base_price = base_prices.get(model, 400)
                    battery_factor = battery_health / 100.0
                    market_factor = random.uniform(0.8, 1.2)
                    
                    selling_price = base_price * battery_factor * market_factor
                    acquisition_cost = selling_price * 0.7  # 70% acquisition cost
                    refurb_cost = selling_price * random.uniform(0.02, 0.08)
                    operational_cost = selling_price * 0.1
                    
                    total_costs = acquisition_cost + refurb_cost + operational_cost
                    profit = selling_price - total_costs
                    
                    # Days to sell based on pricing strategy
                    days_to_sell = random.randint(1, 60)
                    
                    # Damage probability
                    has_damage = random.random() < 0.3
                    
                    analytics_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'model': model,
                        'battery_health': battery_health,
                        'market': market,
                        'selling_price_eur': round(selling_price, 2),
                        'revenue_eur': round(selling_price, 2),
                        'profit_eur': round(profit, 2),
                        'profit_margin': round(profit / selling_price, 3) if selling_price > 0 else 0,
                        'days_to_sell': days_to_sell,
                        'has_damage': has_damage,
                        'vanilla_profit_eur': round(profit * 0.6, 2)  # Baseline comparison
                    })
                
                st.session_state['enhanced_analytics_data'] = analytics_data
        
        # Load data
        df = pd.DataFrame(st.session_state['enhanced_analytics_data'])
        df['date'] = pd.to_datetime(df['date'])
        
        st.success(f"✅ Generated {len(df):,} synthetic transaction records")
        
        # ENHANCED BUSINESS DASHBOARD
        st.subheader("🏆 Executive Summary - AI vs Baseline Performance")
        
        # Generate synthetic AI feedback data
        if 'ai_feedback_data' not in st.session_state:
            synthetic_feedback = []
            base_date = datetime.now() - timedelta(days=500)
            
            for i in range(5000):  # Match the 5000 business records
                date = base_date + timedelta(days=random.randint(0, 365))
                
                # Simulate AI learning curve - better performance over time
                learning_factor = min(1.3, 1.0 + (i / 4000))  # Max 30% improvement over 5000 decisions
                base_reward = random.uniform(30, 120)
                ai_reward = base_reward * learning_factor + random.uniform(-15, 15)
                
                synthetic_feedback.append({
                    'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'decision_id': f'ai_{i}',
                    'tier': random.choice([0.9, 1.0, 1.1]),
                    'reward': max(0, ai_reward),
                    'sale_outcome': random.choice(['Device Sold', 'Still in Inventory', 'Price Reduced']),
                    'features': {}
                })
                
            st.session_state['ai_feedback_data'] = synthetic_feedback
        
        feedback_df = pd.DataFrame(st.session_state['ai_feedback_data'])
        baseline_df = df.copy()
        
        # ENHANCEMENT 2: Executive Summary with EUR comma formatting
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # AI Model performance
            ai_total_profit = feedback_df['reward'].sum()
            st.metric("🤖 Total Profit (AI Model)", f"€{ai_total_profit:,.0f}")
            
        with col2:
            # ENHANCEMENT 3: Change "Simple Model" to "Manual Sale"
            baseline_sample = baseline_df.sample(n=min(len(feedback_df), len(baseline_df)))
            baseline_total_profit = baseline_sample['vanilla_profit_eur'].sum()
            st.metric("👤 Total Profit (Manual Sale)", f"€{baseline_total_profit:,.0f}")
            
        with col3:
            # Profit uplift calculation with comma formatting
            if baseline_total_profit > 0:
                profit_uplift = ((ai_total_profit - baseline_total_profit) / baseline_total_profit) * 100
                profit_diff = ai_total_profit - baseline_total_profit
                st.metric("📈 Profit Uplift", f"{profit_uplift:,.1f}%", delta=f"€{profit_diff:,.0f}")
            else:
                st.metric("📈 Profit Uplift", "N/A")
            
        with col4:
            # Average profit per unit comparison with comma formatting
            ai_avg_profit = feedback_df['reward'].mean()
            baseline_avg_profit = baseline_sample['vanilla_profit_eur'].mean()
            avg_profit_diff = ai_avg_profit - baseline_avg_profit
            st.metric("💹 Avg Profit/Unit (AI)", f"€{ai_avg_profit:,.0f}", 
                     delta=f"€{avg_profit_diff:,.0f}" if avg_profit_diff != 0 else None)
        
        st.markdown("---")
        
        # ENHANCEMENT 3: Change "Simple Model" to "Manual Sale" in the chart
        st.subheader("📈 Cumulative Gains Chart - AI Learning Over Time")
        
        # Create cumulative profit data
        feedback_df_sorted = feedback_df.sort_values('timestamp').reset_index(drop=True)
        feedback_df_sorted['cumulative_ai_profit'] = feedback_df_sorted['reward'].cumsum()
        
        # Create baseline cumulative (assuming fixed performance)
        baseline_avg = baseline_df['vanilla_profit_eur'].mean()
        feedback_df_sorted['cumulative_manual_profit'] = [baseline_avg * (i + 1) for i in range(len(feedback_df_sorted))]
        feedback_df_sorted['decision_number'] = range(1, len(feedback_df_sorted) + 1)
        
        # ENHANCEMENT 5: Interactive slider for progress visualization
        st.markdown("### 🎛️ Interactive Progress Visualization")
        
        # Slider to control how many decisions to show
        max_decisions = len(feedback_df_sorted)
        selected_decisions = st.slider(
            "Show progress up to decision number:",
            min_value=50,
            max_value=max_decisions,
            value=max_decisions,
            step=10,
            help="Move the slider to see how AI performance evolved over time"
        )
        
        # Filter data based on slider selection
        display_df = feedback_df_sorted[feedback_df_sorted['decision_number'] <= selected_decisions].copy()
        
        # Create the chart with filtered data
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=display_df['decision_number'],
            y=display_df['cumulative_ai_profit'],
            name='Intelligent Model',
            line=dict(color='#1f77b4', width=4),
            hovertemplate='Decision: %{x}<br>AI Profit: €%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=display_df['decision_number'],
            y=display_df['cumulative_manual_profit'],
            name='Manual Sale',  # Changed from "Simple Model"
            line=dict(color='#808080', width=3, dash='dash'),
            hovertemplate='Decision: %{x}<br>Manual Profit: €%{y:,.0f}<extra></extra>'
        ))
        
        # Add current position indicator
        if selected_decisions < max_decisions:
            fig.add_vline(x=selected_decisions, line_dash="dot", line_color="red", 
                         annotation_text=f"Decision {selected_decisions}")
        
        fig.update_layout(
            title=f'Cumulative Profit: AI vs Manual Sale (Up to Decision {selected_decisions})',
            xaxis_title='Number of Sales Decisions',
            yaxis_title='Cumulative Profit (€)',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show current performance metrics at the selected point
        if selected_decisions < max_decisions:
            current_ai_profit = display_df['cumulative_ai_profit'].iloc[-1]
            current_manual_profit = display_df['cumulative_manual_profit'].iloc[-1]
            gap = current_ai_profit - current_manual_profit
            
            st.info(f"💡 **At Decision {selected_decisions}**: AI has generated €{gap:,.0f} more profit than manual pricing ({((gap/current_manual_profit)*100):,.1f}% improvement)")
        else:
            st.info("💡 **The widening gap shows the AI's learning capability and compounding financial impact**")
        
        st.markdown("---")
        
        # Strategic Gains Breakdown (added from enhanced analytics)
        st.subheader("🎯 Strategic Gains Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pricing Strategy Performance
            st.subheader("⚖️ Dynamic Pricing Gain")
            
            if 'tier' in feedback_df.columns:
                strategy_performance = feedback_df.groupby('tier').agg({
                    'reward': ['mean', 'count', 'sum']
                }).round(2)
                strategy_performance.columns = ['Avg Profit (€)', 'Count', 'Total Profit (€)']
                
                # Add strategy names
                tier_names = {0.9: 'Competitive', 1.0: 'Market Rate', 1.1: 'Premium'}
                strategy_performance.index = [tier_names.get(idx, f'Tier {idx}') for idx in strategy_performance.index]
                
                # Format numbers with EUR and commas
                strategy_display = strategy_performance.copy()
                strategy_display['Total Profit (€)'] = strategy_display['Total Profit (€)'].apply(lambda x: f"€{x:,.0f}")
                strategy_display['Avg Profit (€)'] = strategy_display['Avg Profit (€)'].apply(lambda x: f"€{x:,.0f}")
                
                st.dataframe(strategy_display, use_container_width=True)
                
                # Visualize strategy performance
                fig = px.bar(strategy_performance.reset_index(), 
                            x='index', y='Avg Profit (€)',
                            title='Average Profit by Pricing Strategy',
                            labels={'index': 'Strategy'})
                st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Model Learning Progress
            st.subheader("📈 Learning Progress Analysis")
            
            # Show recent vs early performance
            if len(feedback_df) >= 6:
                early_decisions = feedback_df.head(len(feedback_df)//2)
                recent_decisions = feedback_df.tail(len(feedback_df)//2)
                
                early_avg = early_decisions['reward'].mean()
                recent_avg = recent_decisions['reward'].mean()
                improvement = recent_avg - early_avg
                
                st.metric("Early Performance (Avg)", f"€{early_avg:,.0f}")
                st.metric("Recent Performance (Avg)", f"€{recent_avg:,.0f}", 
                         delta=f"€{improvement:,.0f}" if improvement != 0 else None)
                
                improvement_pct = (improvement / early_avg * 100) if early_avg != 0 else 0
                if improvement_pct > 5:
                    st.success(f"📈 **Learning Detected**: {improvement_pct:.1f}% improvement over time")
                elif improvement_pct < -5:
                    st.warning(f"📊 Performance declined by {abs(improvement_pct):.1f}% - may need more diverse feedback")
                else:
                    st.info("➡️ Performance stable - model is consistent")
            
            # Show decision distribution over time
            if len(feedback_df) > 3:
                fig = px.line(feedback_df.reset_index(), x='index', y='reward',
                             title='Individual Decision Performance Over Time',
                             labels={'index': 'Decision Number', 'reward': 'Profit (€)'})
                fig.add_hline(y=feedback_df['reward'].mean(), line_dash="dash", 
                             annotation_text="Average Performance")
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ENHANCEMENT 4: CEO-friendly explanation written like for a 5-year-old but with CEO tone
        st.subheader("🧠 How the Enhanced AI System Works")
        
        st.markdown("""
        **Strategic Multi-Market Pricing Intelligence: Executive Overview**
        
        **🎯 The Business Challenge We Solve**:
        Imagine you're selling lemonade, but instead of one fixed price, you could magically know the perfect price for each customer, each day, and each location. That's exactly what our AI does with phones—but with mathematical precision that learns and improves every single day.
        
        **🤖 How Our AI Brain Makes Money**:
        Think of our system like having the world's smartest pricing expert who never sleeps:
        - **Three Smart Choices**: For every phone, it picks between "Low Price for Fast Sale," "Normal Market Price," or "High Price for Maximum Profit"
        - **Learning from Every Deal**: When you tell it "this worked" or "this didn't work," it gets smarter about future decisions
        - **Global Market Intelligence**: It simultaneously checks 5 different countries to find where your phone will make the most money
        
        **🌍 The Strategic Advantage**:
        While your competitors use the same price everywhere, our AI is like having local experts in Romania, Poland, Greece, Bulgaria, and Finland—all working 24/7 to maximize your profits.
        
        **📊 Why This Creates Exponential Value**:
        - **Compound Learning**: Every feedback makes the next 1,000 decisions better
        - **Market Arbitrage**: Automatically finds profit opportunities across international markets  
        - **Risk-Adjusted Pricing**: Factors in inventory levels and market timing to optimize cash flow
        - **Acquisition Intelligence**: Tells you the maximum price to pay when buying devices
        
        **💡 Executive Bottom Line**:
        This isn't just "smart pricing"—it's a self-improving profit optimization engine that turns every transaction into business intelligence for future growth.
        """)
        
        # Business Health KPIs from synthetic data
        st.subheader("📊 Business Health KPIs (5,000 Records)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_revenue = df['revenue_eur'].sum()
            st.metric("💵 Total Revenue", f"€{total_revenue:,.0f}")
            
        with col2:
            total_profit = df['profit_eur'].sum()
            st.metric("💰 Total Profit", f"€{total_profit:,.0f}")
            
        with col3:
            units_sold = len(df)
            st.metric("📱 Units Sold", f"{units_sold:,}")
            
        with col4:
            avg_profit = df['profit_eur'].mean() if len(df) > 0 else 0
            st.metric("💹 Avg Profit/Unit", f"€{avg_profit:.0f}")
            
        with col5:
            avg_days = df['days_to_sell'].mean() if len(df) > 0 else 0
            st.metric("⏱️ Avg Days to Sell", f"{avg_days:.1f}")
            
    except Exception as e:
        st.error(f"Error loading enhanced analytics: {str(e)}")
        st.info("Using basic demo mode instead.")

# Footer
st.markdown("---")
st.markdown("**🔄 Full Circle Exchange V2** | Progressive Maturity Design")
