from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import logging
from typing import Dict, Any
import json
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HydroAlert API",
    description="Smart Water Intake Monitor using Facial and Voice Analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class WaterIntakeRequest(BaseModel):
    amount_ml: int

# Simple in-memory storage for demo
users_db = {}
hydration_history = {}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HydroAlert API is running!",
        "version": "1.0.0",
        "description": "Smart Water Intake Monitor using Facial and Voice Analysis"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "hydroalert"}

@app.get("/api/v1/users")
async def get_or_create_user(user_id: str):
    """Get or create a user"""
    if user_id not in users_db:
        users_db[user_id] = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "daily_goal_ml": 2000,
            "current_intake_ml": 0,
            "last_reset": datetime.now().date().isoformat()
        }
        hydration_history[user_id] = []
    
    return JSONResponse(content=users_db[user_id])

@app.post("/api/v1/users")
async def create_user(user_id: str, hydration_goal_ml: int = 2000, reminder_frequency_minutes: int = 60, preferences: dict = {}):
    """Create a new user"""
    if user_id in users_db:
        return JSONResponse(content={"message": "User already exists", "user_id": user_id})
    
    users_db[user_id] = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "daily_goal_ml": hydration_goal_ml,
        "current_intake_ml": 0,
        "last_reset": datetime.now().date().isoformat(),
        "reminder_frequency_minutes": reminder_frequency_minutes,
        "preferences": preferences
    }
    hydration_history[user_id] = []
    
    return JSONResponse(content={"message": "User created successfully", "user_id": user_id})

@app.post("/api/v1/users/{user_id}/log-intake")
async def log_water_intake(user_id: str, request: WaterIntakeRequest):
    """Log water intake for a user"""
    amount_ml = request.amount_ml
    
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Reset daily intake if it's a new day
    today = datetime.now().date().isoformat()
    if users_db[user_id]["last_reset"] != today:
        users_db[user_id]["current_intake_ml"] = 0
        users_db[user_id]["last_reset"] = today
    
    # Update intake
    users_db[user_id]["current_intake_ml"] += amount_ml
    
    # Log to history
    entry = {
        "timestamp": datetime.now().isoformat(),
        "amount_ml": amount_ml,
        "type": "intake",
        "total_daily_ml": users_db[user_id]["current_intake_ml"]
    }
    hydration_history[user_id].append(entry)
    
    return JSONResponse(content={
        "message": "Intake logged successfully",
        "current_daily_intake": users_db[user_id]["current_intake_ml"],
        "daily_goal": users_db[user_id]["daily_goal_ml"]
    })

@app.get("/api/v1/users/{user_id}/log-intake")
async def get_log_intake_info(user_id: str, amount_ml: int = 250):
    """Get log intake endpoint info and test with default amount"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    return JSONResponse(content={
        "message": "This is a POST endpoint. Use POST with JSON body: {'amount_ml': 250}",
        "user_id": user_id,
        "current_daily_intake": users_db[user_id]["current_intake_ml"],
        "daily_goal": users_db[user_id]["daily_goal_ml"],
        "example_post_request": f"POST /api/v1/users/{user_id}/log-intake with body: {{'amount_ml': {amount_ml}}}"
    })

@app.get("/api/v1/users/{user_id}/history")
async def get_user_history(user_id: str):
    """Get hydration history for a specific user"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate statistics
    history = hydration_history[user_id]
    intakes = [entry for entry in history if entry.get("type") == "intake"]
    analyses = [entry for entry in history if entry.get("type") == "analysis"]
    
    stats = {
        "total_intake_ml": sum(entry["amount_ml"] for entry in intakes),
        "analysis_count": len(analyses),
        "average_hydration_score": 75.0,  # Default
        "average_daily_intake_ml": sum(entry["amount_ml"] for entry in intakes) / max(len(intakes), 1)
    }
    
    return JSONResponse(content={
        "history": history,
        "statistics": stats
    })

@app.post("/api/v1/analysis")
async def analyze_hydration(request: dict):
    """Analyze hydration based on user data"""
    user_id = request.get("user_id")
    
    # Simulate analysis results
    analysis_result = {
        "hydration_score": random.uniform(60, 90),
        "hydration_level": random.choice(["Low", "Moderate", "Good", "Excellent"]),
        "facial_analysis": {
            "lip_dryness": random.uniform(0.1, 0.8),
            "eye_redness": random.uniform(0.1, 0.7),
            "skin_elasticity": random.uniform(0.4, 0.9),
            "skin_tone": random.uniform(0.5, 0.8)
        },
        "voice_analysis": {
            "voice_roughness": random.uniform(0.1, 0.6),
            "speech_clarity": random.uniform(0.6, 0.95)
        },
        "recommendations": [
            "👍 Your hydration looks good!",
            "💧 Consider drinking 150-200ml more water",
            "🔄 Remember to stay hydrated throughout the day"
        ],
        "confidence": random.uniform(0.8, 0.95),
        "timestamp": datetime.now().isoformat()
    }
    
    # Log analysis to history if user_id provided
    if user_id and user_id in hydration_history:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "analysis",
            "hydration_score": analysis_result["hydration_score"],
            "hydration_level": analysis_result["hydration_level"]
        }
        hydration_history[user_id].append(entry)
    
    return JSONResponse(content={"analysis": analysis_result})

@app.get("/api/v1/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user statistics"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    history = hydration_history[user_id]
    
    # Calculate stats
    intakes = [entry for entry in history if entry.get("type") == "intake"]
    analyses = [entry for entry in history if entry.get("type") == "analysis"]
    
    total_intake = sum(entry["amount_ml"] for entry in intakes)
    avg_daily = total_intake / max(len(intakes), 1)
    
    stats = {
        "user_id": user_id,
        "total_lifetime_intake_ml": total_intake,
        "average_daily_intake_ml": avg_daily,
        "current_daily_intake_ml": user["current_intake_ml"],
        "daily_goal_ml": user["daily_goal_ml"],
        "goal_completion_percentage": (user["current_intake_ml"] / user["daily_goal_ml"]) * 100,
        "total_entries": len(history),
        "analysis_count": len(analyses)
    }
    
    return JSONResponse(content=stats)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
