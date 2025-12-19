from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

@app.post("/api/v1/users/{user_id}/log-intake")
async def log_water_intake(user_id: str, amount_ml: int):
    """Log water intake for a user"""
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
        "total_daily_ml": users_db[user_id]["current_intake_ml"]
    }
    hydration_history[user_id].append(entry)
    
    return JSONResponse(content={
        "message": "Intake logged successfully",
        "current_daily_intake": users_db[user_id]["current_intake_ml"],
        "daily_goal": users_db[user_id]["daily_goal_ml"]
    })

@app.get("/api/v1/users/{user_id}/history")
async def get_user_history(user_id: str):
    """Get hydration history for a specific user"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    return JSONResponse(content=hydration_history[user_id])

@app.post("/api/v1/analyze-hydration")
async def analyze_hydration_demo():
    """Demo hydration analysis endpoint"""
    # Simulate analysis results
    analysis_result = {
        "hydration_level": random.choice(["low", "moderate", "good"]),
        "confidence": random.uniform(0.7, 0.95),
        "recommendations": [
            "Drink more water",
            "Consider electrolyte supplements",
            "Monitor your intake throughout the day"
        ],
        "estimated_dehydration_percentage": random.uniform(0, 15),
        "timestamp": datetime.now().isoformat()
    }
    
    return JSONResponse(content=analysis_result)

@app.get("/api/v1/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user statistics"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    history = hydration_history[user_id]
    
    # Calculate stats
    total_intake = sum(entry["amount_ml"] for entry in history)
    avg_daily = total_intake / max(len(history), 1)
    
    stats = {
        "user_id": user_id,
        "total_lifetime_intake_ml": total_intake,
        "average_daily_intake_ml": avg_daily,
        "current_daily_intake_ml": user["current_intake_ml"],
        "daily_goal_ml": user["daily_goal_ml"],
        "goal_completion_percentage": (user["current_intake_ml"] / user["daily_goal_ml"]) * 100,
        "total_entries": len(history)
    }
    
    return JSONResponse(content=stats)

if __name__ == "__main__":
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 