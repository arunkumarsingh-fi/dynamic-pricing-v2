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
    page_title="Full Circle Exchange V2",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .section-header {
        font-size: 1.3rem;
        color: #2c3e50;
        margin: 1rem 0;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 5px;
        border-left: 3px solid #3498db;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #27ae60;
        margin: 0.5rem 0;
    }
    .price-display {
        font-size: 1.8rem;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
        padding: 1rem;
        background: #f8fff8;
        border-radius: 8px;
        border: 2px solid #27ae60;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🔄 Full Circle Exchange V2</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">End-to-End Asset Optimization Platform</div>', unsafe_allow_html=True)

# Sidebar for navigation and settings
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Page selection
    page_selection = st.radio(
        "Navigate to:",
        ["📱 Pricing Calculator", "📊 Analytics Dashboard", "⚙️ Settings"],
        index=0
    )
    
    st.markdown("---")
    
    # Quick settings
    st.markdown("### ⚙️ Quick Settings")
    debug_mode = st.checkbox("🔍 Debug Mode")
    show_advanced = st.checkbox("🔧 Advanced Options")

# Main content area
if page_selection == "📱 Pricing Calculator":
    # Device Information Section
    st.markdown('<div class="section-header">📱 Device Information</div>', unsafe_allow_html=True)
    
    # Create three columns for device inputs
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**Basic Details**")
        
        # iPhone model selection
        iphone_models = [
            'iPhone 11', 'iPhone 11 Pro', 'iPhone 11 Pro Max',
            'iPhone 12', 'iPhone 12 Mini', 'iPhone 12 Pro', 'iPhone 12 Pro Max',
            'iPhone 13', 'iPhone 13 Mini', 'iPhone 13 Pro', 'iPhone 13 Pro Max',
            'iPhone 14', 'iPhone 14 Plus', 'iPhone 14 Pro', 'iPhone 14 Pro Max',
            'iPhone 15', 'iPhone 15 Plus', 'iPhone 15 Pro', 'iPhone 15 Pro Max'
        ]
        
        selected_model = st.selectbox(
            "📱 iPhone Model",
            iphone_models,
            index=iphone_models.index('iPhone 13 Pro')
        )
        
        # Storage capacity (simplified)
        storage_options = ['64GB', '128GB', '256GB', '512GB', '1TB']
        selected_storage = st.selectbox("💾 Storage", storage_options, index=1)
        
        # Color (for market appeal)
        color_options = ['Space Gray', 'Silver', 'Gold', 'Midnight', 'Blue', 'Purple', 'Red', 'Green']
        selected_color = st.selectbox("🎨 Color", color_options)
    
    with col2:
        st.markdown("**Condition Assessment**")
        
        # Battery health with visual indicator
        battery_health = st.slider("🔋 Battery Health", 60, 100, 95, help="Current maximum capacity")
        
        # Create battery health indicator
        if battery_health >= 90:
            battery_status = "🟢 Excellent"
        elif battery_health >= 80:
            battery_status = "🟡 Good"
        else:
            battery_status = "🔴 Needs Service"
        st.write(f"Status: {battery_status}")
        
        # Physical condition
        screen_condition = st.selectbox("📱 Screen", ["Perfect", "Minor Scratches", "Cracked", "Severely Damaged"])
        back_condition = st.selectbox("🔙 Back/Frame", ["Perfect", "Minor Wear", "Scratched", "Damaged"])
        
        # Functionality check
        all_functions_work = st.checkbox("✅ All Functions Working", value=True)
    
    with col3:
        st.markdown("**Quick Info**")
        
        # Visual condition score
        condition_factors = {
            "Perfect": 100, "Minor Scratches": 85, "Minor Wear": 85,
            "Cracked": 70, "Scratched": 75, "Damaged": 60, "Severely Damaged": 40
        }
        
        screen_score = condition_factors.get(screen_condition, 100)
        back_score = condition_factors.get(back_condition, 100)
        function_score = 100 if all_functions_work else 50
        
        overall_condition = (screen_score + back_score + function_score + battery_health) / 4
        
        # Condition gauge
        if overall_condition >= 90:
            condition_color = "🟢"
            condition_text = "Excellent"
        elif overall_condition >= 75:
            condition_color = "🟡" 
            condition_text = "Good"
        elif overall_condition >= 60:
            condition_color = "🟠"
            condition_text = "Fair"
        else:
            condition_color = "🔴"
            condition_text = "Poor"
        
        st.metric("Condition Score", f"{overall_condition:.0f}%", delta=None)
        st.write(f"{condition_color} {condition_text}")
        
        # Market readiness
        if overall_condition >= 85 and all_functions_work:
            st.success("✅ Market Ready")
        else:
            st.warning("⚠️ May Need Refurb")

    # Market Context Section
    st.markdown('<div class="section-header">🌍 Market Context</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        target_market = st.selectbox(
            "🎯 Target Market",
            ['Romania', 'Bulgaria', 'Greece', 'Poland', 'Finland'],
            index=3
        )
    
    with col2:
        inventory_level = st.selectbox(
            "📦 Inventory Level",
            ['Low', 'Medium', 'High'],
            index=1
        )
    
    with col3:
        urgency = st.selectbox(
            "⏱️ Sale Urgency",
            ['No Rush', 'Moderate', 'Urgent'],
            index=0
        )
    
    with col4:
        new_model_soon = st.checkbox("📅 New Model Soon", help="Is a new iPhone expected soon?")

    # AI Analysis Section
    st.markdown('<div class="section-header">🤖 AI Price Analysis</div>', unsafe_allow_html=True)
    
    # Analysis buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quick_price_btn = st.button("⚡ Quick Price", use_container_width=True, type="secondary")
    
    with col2:
        detailed_analysis_btn = st.button("🔬 Detailed Analysis", use_container_width=True, type="primary")
    
    with col3:
        market_compare_btn = st.button("🌍 Compare Markets", use_container_width=True)

    # Handle button actions
    if quick_price_btn or detailed_analysis_btn or market_compare_btn:
        # Convert inputs for API
        screen_damage = 1 if screen_condition in ["Cracked", "Severely Damaged"] else 0
        back_damage = 1 if back_condition in ["Damaged"] else 0
        
        payload = {
            'Model': selected_model,
            'Battery': battery_health,
            'Screen_Damage': screen_damage,
            'Backglass_Damage': back_damage,
            'market': target_market.lower(),
            'inventory_level': inventory_level.lower(),
            'new_model_imminent': new_model_soon
        }
        
        try:
            if market_compare_btn:
                # Multi-market analysis
                api_url = f'{api_base}/optimize_market_and_price'
                response = requests.post(api_url, json=payload, timeout=15)
                result = response.json()
                
                # Display results
                if 'best_option' in result:
                    best = result['best_option']
                    
                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Best Market Recommendation")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Market", best['market'])
                    with col2:
                        st.metric("Selling Price", f"€{best['selling_price_eur']:.2f}")
                    with col3:
                        st.metric("Expected Profit", f"€{best['net_profit_eur']:.2f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show all markets comparison
                    st.markdown("### 📊 All Markets Comparison")
                    
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
            
            else:
                # Single market price analysis
                api_url = f'{api_base}/recommend_price'
                response = requests.post(api_url, json=payload, timeout=10)
                result = response.json()
                
                # Display results
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown(f"### 💰 Pricing Recommendation for {target_market}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    price_eur = result.get('recommended_price_eur', 0)
                    st.markdown(f'<div class="price-display">€{price_eur:.2f}</div>', unsafe_allow_html=True)
                    st.markdown("**Recommended Selling Price**")
                
                with col2:
                    strategy = result.get('pricing_strategy', 'Market Rate')
                    tier = result.get('recommended_tier', 1.0)
                    st.metric("Strategy", strategy)
                    st.metric("Tier Multiplier", f"{tier}x")
                
                with col3:
                    target_cost = result.get('target_acquisition_cost', {}).get('eur', 0)
                    st.metric("Max Buy Price", f"€{target_cost:.2f}")
                    
                    if price_eur > target_cost:
                        profit = price_eur - target_cost
                        st.metric("Expected Profit", f"€{profit:.2f}", delta=f"{(profit/target_cost)*100:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional details for detailed analysis
                if detailed_analysis_btn:
                    st.markdown("### 📋 Detailed Breakdown")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Pricing Options**")
                        pricing_options = result.get('all_pricing_options', {})
                        
                        options_data = []
                        for tier_val, details in pricing_options.items():
                            options_data.append({
                                'Strategy': details['strategy'],
                                'Price (EUR)': f"€{details['price_eur']:.2f}",
                                'Recommended': '✅' if details['recommended'] else ''
                            })
                        
                        if options_data:
                            df_options = pd.DataFrame(options_data)
                            st.dataframe(df_options, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Cost Breakdown**")
                        cost_info = result.get('cost_breakdown', {})
                        
                        if cost_info:
                            st.write(f"Refurbishing: €{cost_info.get('refurbishing_cost_eur', 0):.2f}")
                            st.write(f"Refurb Type: {cost_info.get('refurbishing_tier', 'N/A')}")
                            st.write(f"Cost %: {cost_info.get('cost_percentage', 0):.1f}%")
                
                # Store in session state for feedback
                st.session_state['last_recommendation'] = result
                st.session_state['device_info'] = {
                    'model': selected_model,
                    'storage': selected_storage,
                    'color': selected_color,
                    'battery': battery_health,
                    'condition': condition_text
                }
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("Please ensure the ML API is running on http://localhost:5002")

elif page_selection == "📊 Analytics Dashboard":
    st.markdown('<div class="section-header">📊 Business Analytics</div>', unsafe_allow_html=True)
    st.info("Analytics dashboard will be implemented based on your Page 2 design...")
    
    # Placeholder for analytics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", "€125,450", "12.5%")
    
    with col2:
        st.metric("Units Sold", "342", "8.2%")
    
    with col3:
        st.metric("Avg Profit", "€89.50", "4.7%")
    
    with col4:
        st.metric("Success Rate", "94.2%", "2.1%")

elif page_selection == "⚙️ Settings":
    st.markdown('<div class="section-header">⚙️ System Settings</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**API Configuration**")
        api_url_input = st.text_input("ML API URL", value=api_base)
        
        # Test API connection
        if st.button("🔍 Test Connection"):
            try:
                response = requests.get(f"{api_url_input}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API Connection Successful")
                    health_data = response.json()
                    st.json(health_data)
                else:
                    st.error("❌ API Connection Failed")
            except Exception as e:
                st.error(f"❌ Connection Error: {str(e)}")
    
    with col2:
        st.markdown("**Display Preferences**")
        currency_preference = st.selectbox("Currency Display", ["EUR", "USD", "GBP"])
        theme_preference = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        show_debug_info = st.checkbox("Show Debug Information")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.8rem;">'
    '🔄 Full Circle Exchange V2 | AI-Powered Asset Optimization'
    '</div>', 
    unsafe_allow_html=True
)
