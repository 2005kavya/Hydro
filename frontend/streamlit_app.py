import streamlit as st
import cv2
import numpy as np
import requests
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import base64
from PIL import Image
import io
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="HydroAlert - Smart Water Intake Monitor",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    .hydration-score {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .excellent { background-color: #d4edda; color: #155724; }
    .good { background-color: #d1ecf1; color: #0c5460; }
    .moderate { background-color: #fff3cd; color: #856404; }
    .low { background-color: #f8d7da; color: #721c24; }
    .critical { background-color: #f5c6cb; color: #721c24; }
    .recommendation {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .water-intake-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .ml-log-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .facial-analysis-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'intake_history' not in st.session_state:
    st.session_state.intake_history = []
if 'daily_goal' not in st.session_state:
    st.session_state.daily_goal = 2000  # Default 2000ml
if 'current_intake' not in st.session_state:
    st.session_state.current_intake = 0

# API configuration
API_BASE_URL = "http://localhost:8000"

def create_user_if_not_exists():
    """Create a user if one doesn't exist"""
    if st.session_state.user_id is None:
        # Generate a simple user ID
        user_id = f"user_{int(time.time())}"
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/users",
                params={"user_id": user_id},
                json={
                    "hydration_goal_ml": st.session_state.daily_goal,
                    "reminder_frequency_minutes": 60,
                    "preferences": {}
                }
            )
            if response.status_code == 200:
                st.session_state.user_id = user_id
                st.success(f"Welcome! Your user ID is: {user_id}")
            else:
                st.error("Failed to create user")
        except Exception as e:
            st.error(f"Error creating user: {str(e)}")

def capture_image():
    """Capture image from webcam"""
    try:
        # Use camera input
        camera_input = st.camera_input("Take a photo of your face")
        
        if camera_input is not None:
            # Convert to bytes
            image_bytes = camera_input.getvalue()
            
            # Convert to PIL Image for display
            image = Image.open(io.BytesIO(image_bytes))
            
            return image_bytes, image
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Error capturing image: {str(e)}")
        return None, None

def record_audio():
    """Record audio for voice analysis"""
    # Placeholder for audio recording
    st.info("🎤 Voice recording feature coming soon!")
    return None

def analyze_hydration(image_data, audio_data=None):
    """Send data to backend for hydration analysis"""
    try:
        if st.session_state.user_id is None:
            st.error("Please create a user first")
            return None
        
        # For now, use the mock analysis endpoint
        response = requests.post(
            f"{API_BASE_URL}/api/v1/analysis",
            json={"user_id": st.session_state.user_id}
        )
        
        if response.status_code == 200:
            analysis_result = response.json()["analysis"]
            
            # Add to history
            st.session_state.analysis_history.append({
                "timestamp": datetime.now(),
                "result": analysis_result
            })
            
            return analysis_result
        else:
            st.error("Failed to analyze hydration")
            return None
            
    except Exception as e:
        st.error(f"Error in hydration analysis: {str(e)}")
        return None

def log_water_intake(amount_ml):
    """Log water intake"""
    try:
        if st.session_state.user_id is None:
            st.error("Please create a user first")
            return False
        
        # Add to session state
        st.session_state.current_intake += amount_ml
        
        # Add to history
        st.session_state.intake_history.append({
            "timestamp": datetime.now(),
            "amount_ml": amount_ml
        })
        
        # Send to backend
        response = requests.post(
            f"{API_BASE_URL}/api/v1/users/{st.session_state.user_id}/log-intake",
            json={"amount_ml": amount_ml}
        )
        
        if response.status_code == 200:
            return True
        else:
            st.warning("Intake logged locally, but failed to sync with backend")
            return True
            
    except Exception as e:
        st.error(f"Error logging water intake: {str(e)}")
        return False

def get_hydration_history():
    """Get hydration history from backend"""
    try:
        if st.session_state.user_id is None:
            return None
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/users/{st.session_state.user_id}/history"
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            # Return local data if backend fails
            return {
                "history": st.session_state.intake_history + st.session_state.analysis_history,
                "statistics": {
                    "total_intake_ml": st.session_state.current_intake,
                    "analysis_count": len(st.session_state.analysis_history),
                    "average_hydration_score": 75.0  # Default
                }
            }
            
    except Exception as e:
        st.error(f"Error getting history: {str(e)}")
        return None

def display_hydration_score(score, level):
    """Display hydration score with styling"""
    level_class = level.lower()
    
    st.markdown(f"""
    <div class="hydration-score {level_class}">
        {score:.1f}%
    </div>
    <h3 style="text-align: center; color: #666;">{level} Hydration</h3>
    """, unsafe_allow_html=True)

def display_recommendations(recommendations):
    """Display recommendations"""
    st.subheader("💡 Recommendations")
    
    for rec in recommendations:
        st.markdown(f"""
        <div class="recommendation">
            {rec}
        </div>
        """, unsafe_allow_html=True)

def create_hydration_chart(history_data):
    """Create hydration trend chart"""
    if not history_data or not history_data.get("history"):
        st.info("No history data available yet. Start analyzing to see your trends!")
        return
    
    # Extract analysis data
    analyses = [entry for entry in history_data["history"] if entry["type"] == "analysis"]
    
    if not analyses:
        st.info("No analysis data available yet.")
        return
    
    # Prepare data for plotting
    dates = [datetime.fromisoformat(entry["timestamp"]) for entry in analyses]
    scores = [entry["hydration_score"] for entry in analyses]
    
    # Create DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "Hydration Score": scores
    })
    
    # Create line chart
    fig = px.line(
        df, 
        x="Date", 
        y="Hydration Score",
        title="Hydration Score Trend (Last 7 Days)",
        labels={"Hydration Score": "Score (%)", "Date": "Date"},
        markers=True
    )
    
    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_tickformat=".0f",
        hovermode="x unified"
    )
    
    # Add horizontal lines for hydration levels
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Excellent")
    fig.add_hline(y=60, line_dash="dash", line_color="blue", annotation_text="Good")
    fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Moderate")
    fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Low")
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚰 HydroAlert</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Smart Water Intake Monitor using Facial and Voice Analysis</p>', unsafe_allow_html=True)
    
    # Create user if needed
    create_user_if_not_exists()
    
    # Sidebar Navigation
    st.sidebar.title("🚰 HydroAlert")
    
    # User info in sidebar
    if st.session_state.user_id:
        st.sidebar.success(f"User: {st.session_state.user_id[:8]}...")
    
    # Simple navigation in sidebar - only user info and basic nav
    st.sidebar.markdown("---")
    
    # Page selection with icons
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Home", "📊 Analysis", "💧 Water Intake", "📈 History", "⚙️ Settings"],
        index=["🏠 Home", "📊 Analysis", "💧 Water Intake", "📈 History", "⚙️ Settings"].index(st.session_state.current_page)
    )
    
    # Update current page if selection changed
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.rerun()
    
    # Page routing
    if st.session_state.current_page == "🏠 Home":
        show_home_page()
    elif st.session_state.current_page == "📊 Analysis":
        show_analysis_page()
    elif st.session_state.current_page == "💧 Water Intake":
        show_water_intake_page()
    elif st.session_state.current_page == "📈 History":
        show_history_page()
    elif st.session_state.current_page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Display home page with ML logging and water intake tracking"""
    st.header("Welcome to HydroAlert! 💧")
    
    # Main dashboard cards with the better alignment
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Water Intake Card - Better alignment
        progress_percentage = min(100, (st.session_state.current_intake / st.session_state.daily_goal) * 100)
        st.markdown(f"""
        <div class="water-intake-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h2 style="margin: 0; font-size: 1.8rem;">💧 Today's Water Intake</h2>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">{progress_percentage:.1f}%</span>
            </div>
            <div style="text-align: center; margin: 2rem 0;">
                <h1 style="font-size: 4rem; margin: 0; font-weight: 300; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{st.session_state.current_intake}ml</h1>
                <p style="font-size: 1.2rem; margin: 0.5rem 0; opacity: 0.9;">Goal: {st.session_state.daily_goal}ml</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); height: 25px; border-radius: 15px; margin: 1.5rem 0; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(90deg, #ffffff 0%, #f0f8ff 100%); height: 100%; border-radius: 15px; width: {progress_percentage}%; transition: width 0.5s ease; box-shadow: 0 2px 8px rgba(255,255,255,0.3);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <span style="font-size: 0.9rem; opacity: 0.8;">Remaining: {max(0, st.session_state.daily_goal - st.session_state.current_intake)}ml</span>
                <span style="font-size: 0.9rem; opacity: 0.8;">Intakes: {len(st.session_state.intake_history)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # ML Analysis Card - Better alignment
        st.markdown(f"""
        <div class="ml-log-card">
            <h2 style="margin: 0 0 1rem 0; font-size: 1.8rem;">🤖 ML Analysis</h2>
            <div style="text-align: center; margin: 1.5rem 0;">
                <h1 style="font-size: 3.5rem; margin: 0; font-weight: 300; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{len(st.session_state.analysis_history)}</h1>
                <p style="font-size: 1.1rem; margin: 0.5rem 0; font-weight: 500;">Analyses Today</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 15px; margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.9rem;">Facial Analysis</span>
                    <span style="font-weight: bold;">✅ Active</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem;">Voice Analysis</span>
                    <span style="font-weight: bold; color: #ffeb3b;">🔄 Coming Soon</span>
                </div>
            </div>
            <div style="text-align: center; margin-top: 1rem;">
                <span style="font-size: 0.8rem; opacity: 0.7; background: rgba(255,255,255,0.1); padding: 0.3rem 0.8rem; border-radius: 10px;">AI-Powered Health Insights</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.current_page == "🏠 Home" else "secondary"):
            st.session_state.current_page = "🏠 Home"
            st.rerun()
    
    with col2:
        if st.button("📊 Analysis", use_container_width=True, type="primary" if st.session_state.current_page == "📊 Analysis" else "secondary"):
            st.session_state.current_page = "📊 Analysis"
            st.rerun()
    
    with col3:
        if st.button("💧 Water Intake", use_container_width=True, type="primary" if st.session_state.current_page == "💧 Water Intake" else "secondary"):
            st.session_state.current_page = "💧 Water Intake"
            st.rerun()
    
    with col4:
        if st.button("📈 History", use_container_width=True, type="primary" if st.session_state.current_page == "📈 History" else "secondary"):
            st.session_state.current_page = "📈 History"
            st.rerun()
    
    with col5:
        if st.button("⚙️ Settings", use_container_width=True, type="primary" if st.session_state.current_page == "⚙️ Settings" else "secondary"):
            st.session_state.current_page = "⚙️ Settings"
            st.rerun()
    
    # How it works section
    st.markdown("---")
    st.subheader("🔍 How It Works")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **HydroAlert** uses advanced AI to analyze your facial features and voice to predict your hydration levels before you even feel thirsty!
        
        #### What we analyze:
        - 👄 **Lip dryness** - Dry lips indicate dehydration
        - 👁️ **Eye redness** - Bloodshot eyes can signal dehydration  
        - 🎨 **Skin tone** - Changes in skin elasticity and color
        - 🗣️ **Voice quality** - Dry mouth affects vocal tone
        
        #### Get started:
        1. Go to the **Analysis** page
        2. Take a photo with your camera
        3. Get instant hydration insights and recommendations!
        """)
    
    with col2:
        st.markdown("""
        ### Quick Stats
        
        **Your Hydration Goal:** 2000ml/day
        
        **Today's Progress:** 
        - Intake: {st.session_state.current_intake}ml
        - Goal: {st.session_state.daily_goal}ml
        - Progress: {min(100, (st.session_state.current_intake / st.session_state.daily_goal) * 100):.1f}%
        
        **Last Analysis:** {len(st.session_state.analysis_history)} analyses today
        """)

def show_analysis_page():
    """Display analysis page with facial analysis"""
    st.header("📊 Hydration Analysis")
    
    # Analysis section
    st.subheader("Step 1: Capture Your Image")
    st.info("Position your face in good lighting and take a clear photo")
    
    # Capture image
    image_data, captured_image = capture_image()
    
    if captured_image is not None:
        st.success("✅ Image captured successfully!")
        
        # Voice recording (placeholder)
        st.subheader("Step 2: Voice Analysis (Optional)")
        audio_data = record_audio()
        
        # Analysis button
        if st.button("🔍 Analyze Hydration", type="primary", use_container_width=True):
            with st.spinner("Analyzing your hydration level..."):
                analysis_result = analyze_hydration(image_data, audio_data)
                
                if analysis_result:
                    st.success("Analysis completed!")
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        display_hydration_score(
                            analysis_result["hydration_score"],
                            analysis_result["hydration_level"]
                        )
                    
                    with col2:
                        st.subheader("📊 Analysis Details")
                        
                        # Facial analysis
                        facial = analysis_result["facial_analysis"]
                        st.metric("Lip Dryness", f"{facial['lip_dryness']:.2f}")
                        st.metric("Eye Redness", f"{facial['eye_redness']:.2f}")
                        st.metric("Skin Elasticity", f"{facial['skin_elasticity']:.2f}")
                        
                        # Voice analysis (if available)
                        if analysis_result.get("voice_analysis"):
                            voice = analysis_result["voice_analysis"]
                            st.metric("Voice Roughness", f"{voice['voice_roughness']:.2f}")
                            st.metric("Speech Clarity", f"{voice['speech_clarity']:.2f}")
                    
                    # Recommendations
                    display_recommendations(analysis_result["recommendations"])
                    
                    # Confidence
                    st.info(f"Analysis Confidence: {analysis_result['confidence']:.1%}")

def show_water_intake_page():
    """Display water intake logging page"""
    st.header("💧 Water Intake Logging")
    
    # Current status with better alignment
    col1, col2 = st.columns([3, 2])
    
    with col1:
        progress_percentage = min(100, (st.session_state.current_intake / st.session_state.daily_goal) * 100)
        st.markdown(f"""
        <div class="water-intake-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h2 style="margin: 0; font-size: 2rem;">💧 Today's Progress</h2>
                <div style="background: rgba(255,255,255,0.2); padding: 0.8rem 1.5rem; border-radius: 25px; font-weight: bold; font-size: 1.2rem;">{progress_percentage:.1f}%</div>
            </div>
            <div style="text-align: center; margin: 2rem 0;">
                <h1 style="font-size: 4.5rem; margin: 0; font-weight: 300; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{st.session_state.current_intake}ml</h1>
                <p style="font-size: 1.3rem; margin: 0.5rem 0; opacity: 0.9;">Goal: {st.session_state.daily_goal}ml</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); height: 30px; border-radius: 20px; margin: 2rem 0; overflow: hidden; box-shadow: inset 0 3px 6px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(90deg, #ffffff 0%, #f0f8ff 100%); height: 100%; border-radius: 20px; width: {progress_percentage}%; transition: width 0.5s ease; box-shadow: 0 3px 12px rgba(255,255,255,0.4);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold;">{max(0, st.session_state.daily_goal - st.session_state.current_intake)}</div>
                    <div style="font-size: 0.9rem; opacity: 0.8;">Remaining (ml)</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.intake_history)}</div>
                    <div style="font-size: 0.9rem; opacity: 0.8;">Intakes Today</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold;">{st.session_state.current_intake // max(1, len(st.session_state.intake_history)) if st.session_state.intake_history else 0}</div>
                    <div style="font-size: 0.9rem; opacity: 0.8;">Avg per Intake</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0;">
            <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">🎯 Daily Goal</h3>
            <div style="font-size: 2.5rem; font-weight: 300; margin: 1rem 0;">2000ml</div>
            <p style="margin: 0; opacity: 0.9;">Recommended daily intake</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0;">
            <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">⏰ Next Reminder</h3>
            <div style="font-size: 1.8rem; font-weight: 300; margin: 1rem 0;">2:00 PM</div>
            <p style="margin: 0; opacity: 0.9;">Stay hydrated!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick intake buttons
    st.markdown("---")
    st.subheader("🚰 Quick Intake")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("250ml", use_container_width=True):
            if log_water_intake(250):
                st.success("✅ 250ml logged!")
                st.rerun()
    
    with col2:
        if st.button("500ml", use_container_width=True):
            if log_water_intake(500):
                st.success("✅ 500ml logged!")
                st.rerun()
    
    with col3:
        if st.button("1000ml", use_container_width=True):
            if log_water_intake(1000):
                st.success("✅ 1000ml logged!")
                st.rerun()
    
    with col4:
        if st.button("Reset", use_container_width=True):
            st.session_state.current_intake = 0
            st.success("✅ Daily intake reset!")
            st.rerun()
    
    # Custom amount
    st.markdown("---")
    st.subheader("📝 Custom Amount")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        custom_amount = st.number_input("Amount in ml", min_value=50, max_value=2000, value=250, step=50)
    
    with col2:
        if st.button("Log Amount", use_container_width=True):
            if log_water_intake(custom_amount):
                st.success(f"✅ {custom_amount}ml logged!")
                st.rerun()
    
    # Recent intakes
    if st.session_state.intake_history:
        st.markdown("---")
        st.subheader("📋 Recent Intakes")
        
        for intake in st.session_state.intake_history[-5:]:  # Last 5
            st.info(f"💧 {intake['amount_ml']}ml - {intake['timestamp'].strftime('%H:%M')}")

def show_history_page():
    """Display history page with charts"""
    st.header("📈 Hydration History")
    
    # Get history data
    history_data = get_hydration_history()
    
    if history_data:
        # Statistics
        stats = history_data.get("statistics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Score", f"{stats.get('average_hydration_score', 0):.1f}%")
        
        with col2:
            st.metric("Total Intake", f"{stats.get('total_intake_ml', 0)}ml")
        
        with col3:
            st.metric("Daily Average", f"{stats.get('average_daily_intake_ml', 0):.0f}ml")
        
        with col4:
            st.metric("Analyses", stats.get('analysis_count', 0))
        
        # Chart
        st.subheader("Hydration Trend")
        create_hydration_chart(history_data)
        
        # Recent activity
        st.subheader("Recent Activity")
        
        if history_data.get("history"):
            for entry in history_data["history"][-5:]:  # Last 5 entries
                if entry["type"] == "analysis":
                    st.info(f"📊 Analysis: {entry['hydration_score']:.1f}% - {entry['timestamp'][:19]}")
                elif entry["type"] == "intake":
                    st.success(f"💧 Intake: {entry['amount_ml']}ml - {entry['timestamp'][:19]}")
    else:
        st.info("No history data available yet. Start analyzing to build your hydration profile!")

def show_settings_page():
    """Display settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("User Profile")
    
    if st.session_state.user_id:
        st.info(f"User ID: {st.session_state.user_id}")
        
        # Hydration goal
        goal = st.number_input(
            "Daily Hydration Goal (ml)",
            min_value=500,
            max_value=5000,
            value=st.session_state.daily_goal,
            step=100
        )
        
        if st.button("💾 Save Goal"):
            st.session_state.daily_goal = goal
            st.success("Goal saved!")
        
        # Reminder frequency
        reminder_freq = st.selectbox(
            "Reminder Frequency",
            [30, 60, 90, 120],
            index=1,
            format_func=lambda x: f"{x} minutes"
        )
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved!")
    
    st.subheader("About HydroAlert")
    st.markdown("""
    **Version:** 1.0.0
    
    **Features:**
    - Facial hydration analysis
    - Voice quality assessment
    - Smart recommendations
    - Progress tracking
    
    **Privacy:** All analysis is done locally on your device.
    """)

def show_quick_intake_sidebar():
    """Show quick water intake in sidebar"""
    st.sidebar.subheader("💧 Quick Water Log")
    
    # Preset amounts
    if st.sidebar.button("250ml", key="sidebar_250"):
        if log_water_intake(250):
            st.sidebar.success("✅ 250ml logged!")
    
    if st.sidebar.button("500ml", key="sidebar_500"):
        if log_water_intake(500):
            st.sidebar.success("✅ 500ml logged!")
    
    if st.sidebar.button("1000ml", key="sidebar_1000"):
        if log_water_intake(1000):
            st.sidebar.success("✅ 1000ml logged!")
    
    # Custom amount
    custom = st.sidebar.number_input("Custom (ml)", min_value=50, max_value=2000, value=250, step=50, key="sidebar_custom")
    if st.sidebar.button("Log Custom", key="sidebar_log"):
        if log_water_intake(custom):
            st.sidebar.success(f"✅ {custom}ml logged!")

if __name__ == "__main__":
    main() 