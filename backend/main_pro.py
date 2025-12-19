from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import json
import os
from datetime import datetime, timedelta
import random
import base64
from typing import Dict, Any, Optional
from collections import Counter
from math import log2
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HydroAlert Pro API",
    description="Professional Water Intake Monitor with AI Analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_PATH = "hydroalert.db"

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            email TEXT,
            daily_goal_ml INTEGER DEFAULT 2000,
            weight_kg REAL,
            activity_level TEXT DEFAULT 'moderate',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Hydration logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hydration_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            amount_ml INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT DEFAULT 'manual',
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            hydration_level TEXT,
            confidence REAL,
            recommendations TEXT,
            image_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            alert_type TEXT,
            alert_message TEXT,
            alert_level TEXT,
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

class HydrationAnalyzer:
    """Professional hydration analysis with AI simulation"""
    
    def analyze_facial_features(self, image_data: str) -> Dict[str, Any]:
        """Analyze facial features for hydration signs"""
        # Simulate AI analysis
        features = {
            "skin_moisture": random.uniform(0.3, 0.9),
            "eye_brightness": random.uniform(0.4, 0.95),
            "lip_moisture": random.uniform(0.2, 0.8),
            "face_elasticity": random.uniform(0.5, 0.9)
        }
        
        # Calculate hydration score
        avg_score = sum(features.values()) / len(features)
        
        if avg_score > 0.7:
            level = "good"
            recommendations = ["Maintain current hydration routine", "Continue drinking water regularly"]
        elif avg_score > 0.5:
            level = "moderate"
            recommendations = ["Increase water intake", "Consider electrolyte supplements"]
        else:
            level = "low"
            recommendations = ["Drink water immediately", "Consider sports drinks", "Monitor symptoms"]
        
        return {
            "hydration_level": level,
            "confidence": random.uniform(0.75, 0.95),
            "features": features,
            "recommendations": recommendations,
            "estimated_dehydration_percentage": max(0, (0.7 - avg_score) * 100)
        }
    
    def analyze_voice(self, audio_data: str) -> Dict[str, Any]:
        """Analyze voice for hydration indicators"""
        # Simulate voice analysis
        voice_indicators = {
            "voice_clarity": random.uniform(0.4, 0.9),
            "throat_moisture": random.uniform(0.3, 0.8),
            "speech_fluency": random.uniform(0.5, 0.95)
        }
        
        avg_voice_score = sum(voice_indicators.values()) / len(voice_indicators)
        
        return {
            "voice_analysis": voice_indicators,
            "voice_hydration_score": avg_voice_score,
            "voice_recommendations": ["Stay hydrated", "Avoid caffeine"] if avg_voice_score < 0.6 else ["Voice sounds healthy"]
        }

class DataManager:
    """Professional data management with SQLite"""
    
    def create_user(self, user_id: str, username: str = None, email: str = None, 
                   daily_goal_ml: int = 2000, weight_kg: float = None, 
                   activity_level: str = "moderate") -> Dict[str, Any]:
        """Create or update user"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, username, email, daily_goal_ml, weight_kg, activity_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, username, email, daily_goal_ml, weight_kg, activity_level))
        
        conn.commit()
        conn.close()
        
        return self.get_user(user_id)
    
    def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user data"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        user_data = cursor.fetchone()
        
        if not user_data:
            conn.close()
            return None
        
        columns = [description[0] for description in cursor.description]
        user_dict = dict(zip(columns, user_data))
        
        # Get today's intake
        today = datetime.now().date()
        cursor.execute('''
            SELECT SUM(amount_ml) FROM hydration_logs 
            WHERE user_id = ? AND DATE(timestamp) = ?
        ''', (user_id, today))
        
        today_intake = cursor.fetchone()[0] or 0
        user_dict['current_daily_intake_ml'] = today_intake
        
        conn.close()
        return user_dict
    
    def log_intake(self, user_id: str, amount_ml: int, source: str = "manual", 
                  notes: str = None) -> Dict[str, Any]:
        """Log water intake"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Use explicit datetime to ensure correct local time storage
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.execute('''
            INSERT INTO hydration_logs (user_id, amount_ml, source, notes, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, amount_ml, source, notes, current_time))
        
        conn.commit()
        conn.close()
        
        return {"message": "Intake logged successfully", "amount_ml": amount_ml}
    
    def get_user_history(self, user_id: str, days: int = 7) -> list:
        """Get user hydration history"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM hydration_logs 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, days * 10))  # Assume max 10 entries per day
        
        history = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        result = []
        for row in history:
            entry = dict(zip(columns, row))
            entry['timestamp'] = entry['timestamp']
            result.append(entry)
        
        conn.close()
        return result
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total lifetime intake
        cursor.execute('SELECT SUM(amount_ml) FROM hydration_logs WHERE user_id = ?', (user_id,))
        total_intake = cursor.fetchone()[0] or 0
        
        # Average daily intake (last 30 days)
        cursor.execute('''
            SELECT AVG(daily_total) FROM (
                SELECT DATE(timestamp) as date, SUM(amount_ml) as daily_total
                FROM hydration_logs 
                WHERE user_id = ? AND timestamp >= date('now', '-30 days')
                GROUP BY DATE(timestamp)
            )
        ''', (user_id,))
        avg_daily = cursor.fetchone()[0] or 0
        
        # Today's intake
        today = datetime.now().date()
        cursor.execute('''
            SELECT SUM(amount_ml) FROM hydration_logs 
            WHERE user_id = ? AND DATE(timestamp) = ?
        ''', (user_id, today))
        today_intake = cursor.fetchone()[0] or 0

        # Goal completion
        user = self.get_user(user_id)
        daily_goal = user['daily_goal_ml'] if user else 2000
        goal_completion = (today_intake / daily_goal) * 100 if daily_goal > 0 else 0
        
        # Streak calculation
        cursor.execute('''
            SELECT COUNT(*) FROM (
                SELECT DISTINCT DATE(timestamp) as date
                FROM hydration_logs 
                WHERE user_id = ? AND timestamp >= date('now', '-7 days')
                GROUP BY DATE(timestamp)
                HAVING SUM(amount_ml) >= ?
            )
        ''', (user_id, daily_goal * 0.8))  # 80% of goal
        streak = cursor.fetchone()[0] or 0

        # Per‑day achievement history for the last 7 days
        cursor.execute('''
            SELECT DATE(timestamp) as date, SUM(amount_ml) as total_ml
            FROM hydration_logs
            WHERE user_id = ? AND timestamp >= date('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        ''', (user_id,))
        rows = cursor.fetchall()

        daily_goal_history = []
        for date_str, total_ml in rows:
            total_ml = total_ml or 0
            completion_pct = (total_ml / daily_goal) * 100 if daily_goal > 0 else 0
            if completion_pct >= 100:
                status = "achieved"
            elif completion_pct >= 80:
                status = "near_goal"
            else:
                status = "below_goal"
            daily_goal_history.append({
                "date": date_str,
                "total_intake_ml": int(total_ml),
                "completion_percentage": round(completion_pct, 2),
                "status": status
            })

        # Hydration Window Score: how evenly intake is spread over morning/afternoon/evening (today only)
        hydration_window_score = 0.0
        if today_intake > 0:
            cursor.execute('''
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour, SUM(amount_ml) as total_ml
                FROM hydration_logs
                WHERE user_id = ? AND DATE(timestamp) = ?
                GROUP BY hour
            ''', (user_id, today))
            per_hour_rows = cursor.fetchall()

            morning_ml = 0.0  # 05:00–11:59
            afternoon_ml = 0.0  # 12:00–17:59
            evening_ml = 0.0  # 18:00–23:59

            for hour, total_ml in per_hour_rows:
                h = int(hour)
                total_ml = float(total_ml or 0)
                if 5 <= h < 12:
                    morning_ml += total_ml
                elif 12 <= h < 18:
                    afternoon_ml += total_ml
                elif 18 <= h < 24:
                    evening_ml += total_ml

            # Fractions of today's intake in each window
            m_frac = morning_ml / today_intake if today_intake > 0 else 0.0
            a_frac = afternoon_ml / today_intake if today_intake > 0 else 0.0
            e_frac = evening_ml / today_intake if today_intake > 0 else 0.0

            # Ideal is roughly 1/3 in each window; measure deviation
            ideal = 1.0 / 3.0
            total_deviation = abs(m_frac - ideal) + abs(a_frac - ideal) + abs(e_frac - ideal)

            # Normalize deviation (max ~2 when all water is in one window)
            normalized = min(total_deviation / 2.0, 1.0)
            base_score = (1.0 - normalized) * 100.0

            # Small bonus if all three windows have at least some intake
            windows_with_intake = sum(1 for v in [morning_ml, afternoon_ml, evening_ml] if v > 0)
            if windows_with_intake == 3:
                base_score += 5.0

            hydration_window_score = max(0.0, min(base_score, 100.0))

        conn.close()
        
        return {
            "user_id": user_id,
            "total_lifetime_intake_ml": total_intake,
            "average_daily_intake_ml": round(avg_daily, 2),
            "current_daily_intake_ml": today_intake,
            "daily_goal_ml": daily_goal,
            "goal_completion_percentage": round(goal_completion, 2),
            "current_streak_days": streak,
            "total_entries": len(self.get_user_history(user_id, 365)),
            # NEW: last 7 days achievement breakdown (UI can use this in future without breaking)
            "daily_goal_history": daily_goal_history,
            # NEW: Hydration Window Score (0–100) for today's timing quality
            "hydration_window_score": round(hydration_window_score, 2)
        }
    
    def create_alert(self, user_id: str, alert_type: str, alert_message: str, alert_level: str = "info"):
        """Create an alert for the user"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (user_id, alert_type, alert_message, alert_level)
            VALUES (?, ?, ?, ?)
        ''', (user_id, alert_type, alert_message, alert_level))
        
        conn.commit()
        conn.close()
        
        return {"message": "Alert created successfully"}
    
    def get_alerts(self, user_id: str, unread_only: bool = False) -> list:
        """Get alerts for user"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if unread_only:
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE user_id = ? AND is_read = 0
                ORDER BY created_at DESC
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 50
            ''', (user_id,))
        
        alerts = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        result = []
        for row in alerts:
            entry = dict(zip(columns, row))
            result.append(entry)
        
        conn.close()
        return result
    
    def mark_alert_read(self, alert_id: int):
        """Mark an alert as read"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts SET is_read = 1 WHERE id = ?
        ''', (alert_id,))
        
        conn.commit()
        conn.close()
        
        return {"message": "Alert marked as read"}
    
    def check_and_create_alerts(self, user_id: str):
        """Check user status and create alerts if goal cannot be reached"""
        stats = self.get_user_stats(user_id)
        today_intake = stats['current_daily_intake_ml']
        daily_goal = stats['daily_goal_ml']
        completion = stats['goal_completion_percentage']
        streak = stats['current_streak_days']
        
        # Get current time
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute
        
        # Calculate time remaining in the day (hours)
        hours_remaining = (24 - current_hour) - (current_minute / 60.0)
        
        # Calculate remaining amount needed
        remaining_ml = daily_goal - today_intake
        
        # Only check if goal is not already reached
        if completion < 100:
            # Calculate if it's possible to reach goal
            # Assume reasonable drinking rate: max 500ml per hour (safe drinking rate)
            max_possible_intake = hours_remaining * 500
            
            # Check if goal cannot be reached
            if remaining_ml > max_possible_intake and hours_remaining < 6:
                # Goal cannot be reached - create alert (only once per day)
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                today = datetime.now().date()
                cursor.execute('''
                    SELECT COUNT(*) FROM alerts 
                    WHERE user_id = ? AND alert_type = 'goal_unreachable' 
                    AND DATE(created_at) = ?
                ''', (user_id, today))
                existing = cursor.fetchone()[0]
                conn.close()
                
                if existing == 0:
                    self.create_alert(
                        user_id,
                        "goal_unreachable",
                        f"⚠️ Goal Alert: You're at {today_intake}ml ({completion:.0f}% of {daily_goal}ml goal). With {hours_remaining:.1f} hours remaining, it may be difficult to reach your goal. Try to drink more water!",
                        "warning"
                    )
        
        # Check for goal achievement alert (only when reached)
        if completion >= 100:
            # Check if alert already exists today
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            today = datetime.now().date()
            cursor.execute('''
                SELECT COUNT(*) FROM alerts 
                WHERE user_id = ? AND alert_type = 'goal_achieved' 
                AND DATE(created_at) = ?
            ''', (user_id, today))
            existing = cursor.fetchone()[0]
            conn.close()
            
            if existing == 0:
                self.create_alert(
                    user_id,
                    "goal_achieved",
                    f"🎉 Congratulations! You've reached your daily hydration goal of {daily_goal}ml! Keep up the great work!",
                    "success"
                )
        
        # Check for streak milestones (only on milestone days)
        if streak == 3 or streak == 7 or streak == 14 or streak == 30:
            # Check if alert already exists for this streak
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM alerts 
                WHERE user_id = ? AND alert_type = 'streak_milestone' 
                AND alert_message LIKE ?
            ''', (user_id, f"%{streak} days%"))
            existing = cursor.fetchone()[0]
            conn.close()
            
            if existing == 0:
                self.create_alert(
                    user_id,
                    "streak_milestone",
                    f"🔥 Amazing! You've maintained a {streak}-day hydration streak! Keep it up!",
                    "success"
                )

# Initialize components
analyzer = HydrationAnalyzer()
data_manager = DataManager()


# ------------------------------------------------------------
# Minimal ML model (pure Python logistic regression) for demo
# ------------------------------------------------------------
class SimpleLogisticModel:
    """
    Tiny logistic regression implemented with numpy-like math using lists.
    Trains on synthetic data at startup; used for demo ML classification.
    """

    def __init__(self, num_features: int):
        # include bias as last weight
        self.weights = [0.0 for _ in range(num_features + 1)]

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Stable sigmoid
        if x >= 0:
            z = pow(2.718281828, -x)
            return 1 / (1 + z)
        else:
            z = pow(2.718281828, x)
            return z / (1 + z)

    def predict_proba(self, features: list[float]) -> float:
        # append bias
        x = features + [1.0]
        dot = sum(w * xi for w, xi in zip(self.weights, x))
        return self._sigmoid(dot)

    def train(self, X: list[list[float]], y: list[int], lr: float = 0.05, epochs: int = 300):
        for _ in range(epochs):
            # simple batch gradient descent
            grad = [0.0 for _ in self.weights]
            for features, target in zip(X, y):
                x = features + [1.0]
                p = self._sigmoid(sum(w * xi for w, xi in zip(self.weights, x)))
                error = p - target
                for i in range(len(grad)):
                    grad[i] += error * x[i]
            n = max(len(X), 1)
            for i in range(len(self.weights)):
                self.weights[i] -= lr * (grad[i] / n)


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * (log2(p) if p > 0 else 0.0)
    # Normalize roughly to [0,1] since max entropy for bytes is 8 bits
    return min(entropy / 8.0, 1.0)


def _feature_vector(image_bytes: Optional[bytes], audio_bytes: Optional[bytes]) -> list[float]:
    img_kb = (len(image_bytes) / 1024.0) if image_bytes else 0.0
    img_entropy = _byte_entropy(image_bytes) if image_bytes else 0.0
    aud_kb = (len(audio_bytes) / 1024.0) if audio_bytes else 0.0
    aud_entropy = _byte_entropy(audio_bytes) if audio_bytes else 0.0
    # Basic scaling to a reasonable range
    return [
        min(img_kb / 150.0, 1.0),      # typical selfie ~100-300KB
        img_entropy,                    # [0,1]
        min(aud_kb / 500.0, 1.0),      # short audio clip ~100KB-1MB
        aud_entropy                     # [0,1]
    ]


# Initialize and train the demo ML model on synthetic data
ml_model = SimpleLogisticModel(num_features=4)
synthetic_X = []
synthetic_y = []
random.seed(42)
for _ in range(300):
    # simulate plausible ranges
    img_kb = random.uniform(20, 300) / 150.0
    img_ent = random.uniform(0.3, 0.95)
    aud_kb = random.uniform(0, 1000) / 500.0
    aud_ent = random.uniform(0.2, 0.95)
    features = [min(img_kb, 1.0), img_ent, min(aud_kb, 1.0), aud_ent]
    synthetic_X.append(features)
    # define a synthetic rule: higher entropy + moderate size -> more likely hydrated
    score = 0.4 * img_ent + 0.3 * aud_ent + 0.2 * (1 - abs(img_kb - 0.6)) + 0.1 * (1 - abs(aud_kb - 0.5))
    label = 1 if score > 0.6 else 0
    synthetic_y.append(label)
ml_model.train(synthetic_X, synthetic_y, lr=0.08, epochs=400)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HydroAlert Pro API is running!",
        "version": "2.0.0",
        "description": "Professional Water Intake Monitor with AI Analysis",
        "endpoints": {
            "health": "/health",
            "users": "/api/v1/users",
            "analysis": "/api/v1/analyze-hydration",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "hydroalert-pro",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }

@app.get("/api/v1/users")
async def get_or_create_user(
    user_id: str,
    username: Optional[str] = None,
    email: Optional[str] = None,
    daily_goal_ml: int = 2000,
    weight_kg: Optional[float] = None,
    activity_level: str = "moderate"
):
    """Get or create a user with professional data management"""
    try:
        user = data_manager.get_user(user_id)
        
        if not user:
            # Create new user
            user = data_manager.create_user(
                user_id=user_id,
                username=username,
                email=email,
                daily_goal_ml=daily_goal_ml,
                weight_kg=weight_kg,
                activity_level=activity_level
            )
        
        return JSONResponse(content=user)
    except Exception as e:
        logger.error(f"Error in user management: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User management failed: {str(e)}")

@app.post("/api/v1/users/{user_id}/log-intake")
async def log_water_intake(
    user_id: str, 
    amount_ml: str = Form(...),
    source: str = Form("manual"),
    notes: Optional[str] = Form(None)
):
    """Log water intake with professional tracking"""
    try:
        # Convert string to int and validate
        try:
            amount_ml_int = int(amount_ml)
        except ValueError:
            raise HTTPException(status_code=400, detail="Amount must be a valid number")
        
        if amount_ml_int <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")
        
        result = data_manager.log_intake(user_id, amount_ml_int, source, notes)
        
        # Get updated user stats
        user_stats = data_manager.get_user_stats(user_id)
        
        return JSONResponse(content={
            **result,
            "user_stats": user_stats
        })
    except Exception as e:
        logger.error(f"Error logging intake: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to log intake: {str(e)}")

@app.get("/api/v1/users/{user_id}/history")
async def get_user_history(user_id: str, days: int = 7):
    """Get user hydration history with professional data retrieval"""
    try:
        history = data_manager.get_user_history(user_id, days)
        return JSONResponse(content=history)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.get("/api/v1/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get comprehensive user statistics"""
    try:
        # Check and create alerts when getting stats
        data_manager.check_and_create_alerts(user_id)
        stats = data_manager.get_user_stats(user_id)
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@app.get("/api/v1/users/{user_id}/alerts")
async def get_alerts(user_id: str, unread_only: bool = False):
    """Get alerts for user"""
    try:
        alerts = data_manager.get_alerts(user_id, unread_only=unread_only)
        return JSONResponse(content=alerts)
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch alerts: {str(e)}")

@app.post("/api/v1/alerts/{alert_id}/read")
async def mark_alert_read(alert_id: int):
    """Mark an alert as read"""
    try:
        result = data_manager.mark_alert_read(alert_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error marking alert as read: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to mark alert as read: {str(e)}")

@app.post("/api/v1/analyze-hydration")
async def analyze_hydration(
    user_id: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """Professional hydration analysis with AI simulation"""
    try:
        analysis_result = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "comprehensive-ml"
        }
        
        # Read raw bytes for feature extraction (no heavy deps)
        image_bytes: Optional[bytes] = None
        audio_bytes: Optional[bytes] = None

        if image:
            image_bytes = await image.read()
            analysis_result["image_processed"] = True
        else:
            analysis_result["image_processed"] = False

        if audio:
            audio_bytes = await audio.read()
            analysis_result["audio_processed"] = True
        else:
            analysis_result["audio_processed"] = False

        # Extract simple byte-level features and run ML model
        features = _feature_vector(image_bytes, audio_bytes)
        proba = ml_model.predict_proba(features)
        # Map probability to multi-class hydration level
        if proba > 0.72:
            overall_level = "good"
        elif proba > 0.45:
            overall_level = "moderate"
        else:
            overall_level = "low"

        analysis_result["overall_assessment"] = {
            "hydration_level": overall_level,
            "confidence": round(float(proba), 3),
            "features_used": {
                "image_kb_scaled": features[0],
                "image_entropy": features[1],
                "audio_kb_scaled": features[2],
                "audio_entropy": features[3],
            },
            "recommendations": [
                "Sip water regularly across the day" if overall_level != "good" else "Maintain your routine",
                "Include electrolytes during workouts" if overall_level == "low" else "Set gentle reminders",
            ],
        }
        
        # Store analysis result
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analysis_results 
            (user_id, hydration_level, confidence, recommendations, image_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            analysis_result.get("overall_assessment", {}).get("hydration_level", "unknown"),
            analysis_result.get("overall_assessment", {}).get("confidence", 0.0),
            json.dumps(analysis_result.get("overall_assessment", {}).get("recommendations", [])),
            ""  # no raw image persisted in demo
        ))
        conn.commit()
        conn.close()
        
        # Create alert if hydration is low
        if overall_level == "low":
            data_manager.create_alert(
                user_id,
                "dehydration_warning",
                "⚠️ Dehydration Warning: Your facial analysis indicates low hydration levels. Please drink water immediately!",
                "danger"
            )
        
        return JSONResponse(content=analysis_result)
        
    except Exception as e:
        logger.error(f"Error in hydration analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/v1/dashboard/{user_id}")
async def get_dashboard_data(user_id: str):
    """Get comprehensive dashboard data"""
    try:
        # Get user stats
        stats = data_manager.get_user_stats(user_id)
        
        # Get recent history
        history = data_manager.get_user_history(user_id, 7)
        
        # Get recent analysis
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM analysis_results 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''', (user_id,))
        
        recent_analysis = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        analysis_list = []
        for row in recent_analysis:
            analysis_list.append(dict(zip(columns, row)))
        
        conn.close()
        
        # Calculate trends
        daily_totals = {}
        for entry in history:
            date = entry['timestamp'].split(' ')[0]
            daily_totals[date] = daily_totals.get(date, 0) + entry['amount_ml']
        
        return JSONResponse(content={
            "user_stats": stats,
            "recent_history": history[:10],
            "recent_analysis": analysis_list,
            "daily_trends": daily_totals,
            "recommendations": [
                "Drink water regularly throughout the day",
                "Set hydration reminders",
                "Monitor your progress in the dashboard"
            ]
        })
        
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main_pro:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 