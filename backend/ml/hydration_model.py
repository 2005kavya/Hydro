import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class HydrationModel:
    """
    Simple hydration prediction model
    """
    
    def __init__(self):
        """Initialize the hydration model"""
        logger.info("New hydration model initialized")
        logger.info("HydrationModel initialized")
    
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict hydration score based on features
        
        Args:
            features: Dictionary containing facial and voice features
            
        Returns:
            Hydration score (0-100)
        """
        try:
            # Extract facial features
            facial = features.get("facial_analysis", {})
            
            # Get individual scores
            lip_dryness = facial.get("lip_dryness", 0.5)
            eye_redness = facial.get("eye_redness", 0.5)
            skin_elasticity = facial.get("skin_elasticity", 0.5)
            skin_tone = facial.get("skin_tone", 0.5)
            
            # Simple weighted scoring system
            # Lower dryness/redness = higher hydration
            # Higher elasticity = higher hydration
            
            lip_score = (1 - lip_dryness) * 100
            eye_score = (1 - eye_redness) * 100
            skin_score = skin_elasticity * 100
            tone_score = skin_tone * 100
            
            # Weighted average
            hydration_score = (
                lip_score * 0.3 +
                eye_score * 0.25 +
                skin_score * 0.3 +
                tone_score * 0.15
            )
            
            # Ensure score is between 0-100
            hydration_score = max(0, min(100, hydration_score))
            
            logger.info(f"Predicted hydration score: {hydration_score:.2f}")
            return hydration_score
            
        except Exception as e:
            logger.error(f"Error in hydration prediction: {str(e)}")
            # Return default score if prediction fails
            return 75.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance weights"""
        return {
            "lip_dryness": 0.3,
            "eye_redness": 0.25,
            "skin_elasticity": 0.3,
            "skin_tone": 0.15
        }
    
    def explain_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Explain the prediction with feature contributions"""
        try:
            facial = features.get("facial_analysis", {})
            
            lip_dryness = facial.get("lip_dryness", 0.5)
            eye_redness = facial.get("eye_redness", 0.5)
            skin_elasticity = facial.get("skin_elasticity", 0.5)
            skin_tone = facial.get("skin_tone", 0.5)
            
            # Calculate individual contributions
            lip_contribution = (1 - lip_dryness) * 100 * 0.3
            eye_contribution = (1 - eye_redness) * 100 * 0.25
            skin_contribution = skin_elasticity * 100 * 0.3
            tone_contribution = skin_tone * 100 * 0.15
            
            return {
                "lip_contribution": lip_contribution,
                "eye_contribution": eye_contribution,
                "skin_contribution": skin_contribution,
                "tone_contribution": tone_contribution,
                "total_score": lip_contribution + eye_contribution + skin_contribution + tone_contribution
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return {"error": "Could not explain prediction"} 