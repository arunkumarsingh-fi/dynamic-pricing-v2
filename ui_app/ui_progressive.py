import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Initialize API base URL for V2 (separate ML backend)
api_base = os.getenv('ML_API_URL', 'http://localhost:5003')

# Configure page
st.set_page_config(
    page_title="Full Circle Exchange V2 - Progressive Maturity",
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
    
    .price-button {
        background: #3498db;
        color: white;
        padding: 15px 30px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        margin: 10px;
        cursor: pointer;
        width: 100%;
        text-align: center;
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
    
    .plus-sign {
        text-align: center;
        font-size: 3rem;
        color: #3498db;
        margin: 2rem 0;
        font-weight: bold;
    }
    
    .section-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.title('🔄 Full Circle Exchange V2: Progressive Maturity Flow')
st.markdown("### 📱 Story of a Unit - From Simple to Sophisticated")

# Initialize session state
if 'show_level2' not in st.session_state:
    st.session_state.show_level2 = False
if 'show_level3' not in st.session_state:
    st.session_state.show_level3 = False
if 'show_full_matured' not in st.session_state:
    st.session_state.show_full_matured = False
if 'level1_data' not in st.session_state:
    st.session_state.level1_data = {}
if 'level2_data' not in st.session_state:
    st.session_state.level2_data = {}
if 'level3_data' not in st.session_state:
    st.session_state.level3_data = {}

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
    
    # Predict Buying Price button
    if st.button("Predict Buying Price", key="buy_l1", use_container_width=True):
        payload = {
            'Model': selected_model_l1,
            'Battery': battery_health_l1,
            'Screen_Damage': 0,
            'Backglass_Damage': 0,
            'market': 'poland',
            'inventory_level': 'medium'
        }
        
        try:
            response = requests.post(f'{api_base}/recommend_price', json=payload, timeout=10)
            result = response.json()
            buying_price = result.get('target_acquisition_cost', {}).get('eur', 0)
            st.session_state['buy_price_l1'] = buying_price
            st.rerun()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
    
    # Display buying price if available
    if 'buy_price_l1' in st.session_state:
        st.markdown(f'<div class="price-output">€{st.session_state.buy_price_l1:.2f}</div>', unsafe_allow_html=True)
    
    # Predict Selling Price button
    if st.button("Predict Selling Price", key="sell_l1", use_container_width=True):
        payload = {
            'Model': selected_model_l1,
            'Battery': battery_health_l1,
            'Screen_Damage': 0,
            'Backglass_Damage': 0,
            'market': 'poland',
            'inventory_level': 'medium'
        }
        
        try:
            response = requests.post(f'{api_base}/recommend_price', json=payload, timeout=10)
            result = response.json()
            selling_price = result.get('recommended_price_eur', 0)
            st.session_state['sell_price_l1'] = selling_price
            st.session_state.level1_data = {'model': selected_model_l1, 'battery': battery_health_l1}
            st.rerun()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
    
    # Display selling price if available
    if 'sell_price_l1' in st.session_state:
        st.markdown(f'<div class="price-output">€{st.session_state.sell_price_l1:.2f}</div>', unsafe_allow_html=True)

# Plus sign to expand to level 2
if 'sell_price_l1' in st.session_state and 'buy_price_l1' in st.session_state:
    st.markdown('<div class="plus-sign">+</div>', unsafe_allow_html=True)
    
    # ================================
    # MATURITY LEVEL 2
    # ================================
    st.markdown('<div class="maturity-level">Maturity Level 2</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use data from Level 1
        st.info(f"Building on: {st.session_state.level1_data.get('model', 'iPhone 13 Pro')} with {st.session_state.level1_data.get('battery', 95)}% battery")
        
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
        
        # Predict Buying Price L2
        if st.button("Predict Buying Price", key="buy_l2", use_container_width=True):
            payload = {
                'Model': st.session_state.level1_data.get('model', 'iPhone 13 Pro'),
                'Battery': st.session_state.level1_data.get('battery', 95),
                'Screen_Damage': screen_damage,
                'Backglass_Damage': back_damage,
                'inventory_level': inventory_l2,
                'new_model_imminent': new_release_l2,
                'market': 'poland'
            }
            
            try:
                response = requests.post(f'{api_base}/recommend_price', json=payload, timeout=10)
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
            payload = {
                'Model': st.session_state.level1_data.get('model', 'iPhone 13 Pro'),
                'Battery': st.session_state.level1_data.get('battery', 95),
                'Screen_Damage': screen_damage,
                'Backglass_Damage': back_damage,
                'inventory_level': inventory_l2,
                'new_model_imminent': new_release_l2,
                'market': 'poland'
            }
            
            try:
                response = requests.post(f'{api_base}/recommend_price', json=payload, timeout=10)
                result = response.json()
                selling_price = result.get('recommended_price_eur', 0)
                st.session_state['sell_price_l2'] = selling_price
                st.session_state.level2_data = {
                    'screen_damage': screen_damage,
                    'back_damage': back_damage,
                    'inventory': inventory_l2,
                    'new_release': new_release_l2
                }
                st.rerun()
            except Exception as e:
                st.error(f"API Error: {str(e)}")
        
        if 'sell_price_l2' in st.session_state:
            st.markdown(f'<div class="price-output">€{st.session_state.sell_price_l2:.2f}</div>', unsafe_allow_html=True)

    # Plus sign to expand to level 3
    if 'sell_price_l2' in st.session_state and 'buy_price_l2' in st.session_state:
        st.markdown('<div class="plus-sign">+</div>', unsafe_allow_html=True)
        
        # ================================
        # MATURITY LEVEL 3
        # ================================
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
            
            # Generate price for selected market
            if st.button("Generate price for selected market", key="single_market", use_container_width=True):
                payload = {
                    'Model': st.session_state.level1_data.get('model', 'iPhone 13 Pro'),
                    'Battery': st.session_state.level1_data.get('battery', 95),
                    'Screen_Damage': st.session_state.level2_data.get('screen_damage', 0),
                    'Backglass_Damage': st.session_state.level2_data.get('back_damage', 0),
                    'inventory_level': st.session_state.level2_data.get('inventory', 'decent'),
                    'new_model_imminent': st.session_state.level2_data.get('new_release', False),
                    'market': target_market_l3.lower()
                }
                
                try:
                    response = requests.post(f'{api_base}/recommend_price', json=payload, timeout=10)
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
                payload = {
                    'Model': st.session_state.level1_data.get('model', 'iPhone 13 Pro'),
                    'Battery': st.session_state.level1_data.get('battery', 95),
                    'Screen_Damage': st.session_state.level2_data.get('screen_damage', 0),
                    'Backglass_Damage': st.session_state.level2_data.get('back_damage', 0),
                    'inventory_level': st.session_state.level2_data.get('inventory', 'decent'),
                    'new_model_imminent': st.session_state.level2_data.get('new_release', False)
                }
                
                try:
                    response = requests.post(f'{api_base}/optimize_market_and_price', json=payload, timeout=15)
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
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    price_eur = result.get('recommended_price_eur', 0)
                    st.metric("Recommended Price", f"€{price_eur:.2f}")
                with col2:
                    strategy = result.get('pricing_strategy', 'Market Rate')
                    st.metric("Strategy", strategy)
                with col3:
                    target_cost = result.get('target_acquisition_cost', {}).get('eur', 0)
                    st.metric("Max Buy Price", f"€{target_cost:.2f}")
            
            # Show multi-market result
            if 'multi_market_result' in st.session_state:
                result = st.session_state['multi_market_result']
                
                if 'best_option' in result:
                    best = result['best_option']
                    st.success(f"🎯 Best Market: {best['market']} - €{best['selling_price_eur']:.2f}")
                    
                    # Markets comparison table
                    markets_data = []
                    for market_info in result['market_analysis']:
                        markets_data.append({
                            'Market': market_info['market'],
                            'Price (EUR)': f"€{market_info['selling_price_eur']:.2f}",
                            'Profit (EUR)': f"€{market_info['net_profit_eur']:.2f}",
                            'Strategy': market_info['pricing_strategy']
                        })
                    
                    df_markets = pd.DataFrame(markets_data)
                    st.dataframe(df_markets, use_container_width=True)
            
            # Plus sign to expand to full matured flow
            st.markdown('<div class="plus-sign">+</div>', unsafe_allow_html=True)
            
            # ================================
            # FULL MATURED FLOW
            # ================================
            st.markdown('<div class="maturity-level">Full Matured Flow</div>', unsafe_allow_html=True)
            
            st.markdown("### Business Decision and Financial Analysis")
            st.info("This section maintains the same design and functionality as Version 1")
            
            # Price override section (same as V1)
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
            
            # Business performance metrics
            st.markdown("### Profit Analysis and Business Performance")
            
            # Calculate profit metrics
            if 'single_market_result' in st.session_state:
                result = st.session_state['single_market_result']
                selling_price = custom_selling_price if custom_selling_price > 0 else result.get('recommended_price_eur', 0)
                buying_price = custom_buying_price if custom_buying_price > 0 else result.get('target_acquisition_cost', {}).get('eur', 0)
                
                if selling_price > 0 and buying_price > 0:
                    profit = selling_price - buying_price
                    profit_margin = (profit / selling_price) * 100 if selling_price > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Selling Price", f"€{selling_price:.2f}")
                    with col2:
                        st.metric("Buying Price", f"€{buying_price:.2f}")
                    with col3:
                        st.metric("Expected Profit", f"€{profit:.2f}", f"{profit_margin:.1f}%")
            
            # Send Feedback to Advanced AI
            st.markdown("### Send Feedback to Advanced AI")
            
            col1, col2 = st.columns(2)
            with col1:
                actual_sale_outcome = st.selectbox(
                    "Sale Outcome",
                    ["Device Sold", "Still in Inventory", "Price Reduced", "Returned to Supplier"],
                    help="What actually happened with this device?"
                )
            with col2:
                actual_profit = st.number_input(
                    "Actual Profit (EUR)",
                    value=0.0,
                    step=10.0,
                    help="Actual profit achieved from this transaction"
                )
            
            # Feedback button (same design as V1)
            if st.button("Send Feedback to Advanced AI", key="feedback", use_container_width=True, type="primary"):
                if 'single_market_result' in st.session_state and 'decision_id' in st.session_state['single_market_result']:
                    feedback_payload = {
                        'decision_id': st.session_state['single_market_result']['decision_id'],
                        'actual_reward': actual_profit,
                        'sale_outcome': actual_sale_outcome
                    }
                    
                    try:
                        response = requests.post(f'{api_base}/report_outcome', json=feedback_payload, timeout=10)
                        if response.status_code == 200:
                            st.success("✅ Feedback sent! The AI has learned from this outcome.")
                        else:
                            st.error("❌ Failed to send feedback")
                    except Exception as e:
                        st.error(f"❌ Feedback error: {str(e)}")
                else:
                    st.warning("⚠️ No pricing decision available to provide feedback on")

# Footer
st.markdown("---")
st.markdown("**🔄 Full Circle Exchange V2** | Progressive Maturity Design")
st.markdown(f"*ML API: {api_base}*")
