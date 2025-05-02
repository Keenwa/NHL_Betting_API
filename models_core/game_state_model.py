import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

class SituationalModel:
    """
    Model for predicting SOG adjustments based on game situation.
    This is a placeholder implementation - in a real application
    this would be a more sophisticated model.
    """
    
    def __init__(self):
        """Initialize the situational model."""
        self.trained = False
        self.adjustment_data = {}
    
    def train(self, shot_data: pd.DataFrame) -> bool:
        """
        Train the situational model on historical data.
        
        Args:
            shot_data: DataFrame with shot data including game state info
            
        Returns:
            True if training succeeded, False otherwise
        """
        required_columns = ['score_diff', 'period', 'time_remaining']
        if not all(col in shot_data.columns for col in required_columns):
            print(f"Warning: Missing required columns for training: {[col for col in required_columns if col not in shot_data.columns]}")
            return False
        
        try:
            # In a real implementation, this would actually train a model
            # For now, we'll just set up some simple heuristics
            
            # Store time remaining effects (in seconds)
            self.adjustment_data['time_remaining'] = {
                0: {'mean_adj': 0.4, 'sigma_factor': 1.3},      # Final minute: +0.4 SOG, 30% more variance
                60: {'mean_adj': 0.3, 'sigma_factor': 1.2},     # Last 2 minutes: +0.3 SOG, 20% more variance
                120: {'mean_adj': 0.2, 'sigma_factor': 1.15},   # Last 3 minutes: +0.2 SOG, 15% more variance
                300: {'mean_adj': 0.1, 'sigma_factor': 1.1},    # Last 5 minutes: +0.1 SOG, 10% more variance
                600: {'mean_adj': 0.0, 'sigma_factor': 1.0},    # Mid-period: no adjustment
                900: {'mean_adj': -0.1, 'sigma_factor': 0.95},  # First 5 minutes: -0.1 SOG, 5% less variance
                1200: {'mean_adj': -0.2, 'sigma_factor': 0.9}   # Period start: -0.2 SOG, 10% less variance
            }
            
            # Store overtime effects
            self.adjustment_data['overtime'] = {
                True: {'mean_adj': 0.3, 'sigma_factor': 1.2},   # Overtime: +0.3 SOG, 20% more variance
                False: {'mean_adj': 0.0, 'sigma_factor': 1.0}   # Regulation: no adjustment
            }
            
            # Store trailing team effects
            self.adjustment_data['trailing'] = {
                True: {'mean_adj': 0.25, 'sigma_factor': 1.1},  # Trailing: +0.25 SOG, 10% more variance
                False: {'mean_adj': 0.0, 'sigma_factor': 1.0}   # Not trailing: no adjustment
            }
            
            self.trained = True
            return True
            
        except Exception as e:
            print(f"Error training situational model: {e}")
            return False
    
    def predict_adjustment(
        self, 
        player_id: str,
        period: int,
        score_diff: int,
        time_remaining: int,
        is_overtime: bool
    ) -> Dict[str, float]:
        """
        Predict SOG adjustment based on game situation.
        
        Args:
            player_id: Player ID
            period: Current period
            score_diff: Score difference (player team - opponent)
            time_remaining: Time remaining in seconds
            is_overtime: Whether game is in overtime
            
        Returns:
            Dictionary with mean adjustment and sigma factor
        """
        if not self.trained:
            # Return no adjustment if not trained
            return {'mean_adj': 0.0, 'sigma_factor': 1.0}
        
        # Get score difference adjustment
        score_diff_capped = min(max(score_diff, -3), 3)  # Cap at -3/+3
        score_diff_adj = self.adjustment_data['score_diff'].get(
            score_diff_capped, 
            {'mean_adj': 0.0, 'sigma_factor': 1.0}
        )
        
        # Get period adjustment
        period_capped = min(max(period, 1), 5)  # Cap at periods 1-5
        period_adj = self.adjustment_data['period'].get(
            period_capped, 
            {'mean_adj': 0.0, 'sigma_factor': 1.0}
        )
        
        # Get time remaining adjustment
        time_buckets = sorted(self.adjustment_data['time_remaining'].keys())
        time_bucket = min(time_buckets, key=lambda x: abs(x - time_remaining))
        time_adj = self.adjustment_data['time_remaining'].get(
            time_bucket, 
            {'mean_adj': 0.0, 'sigma_factor': 1.0}
        )
        
        # Get overtime adjustment
        ot_adj = self.adjustment_data['overtime'].get(
            is_overtime, 
            {'mean_adj': 0.0, 'sigma_factor': 1.0}
        )
        
        # Get trailing adjustment
        trailing_adj = self.adjustment_data['trailing'].get(
            score_diff < 0, 
            {'mean_adj': 0.0, 'sigma_factor': 1.0}
        )
        
        # Combine adjustments
        mean_adj = (
            score_diff_adj['mean_adj'] + 
            period_adj['mean_adj'] + 
            time_adj['mean_adj'] + 
            ot_adj['mean_adj'] + 
            trailing_adj['mean_adj']
        )
        
        # Multiply sigma factors
        sigma_factor = (
            score_diff_adj['sigma_factor'] * 
            period_adj['sigma_factor'] * 
            time_adj['sigma_factor'] * 
            ot_adj['sigma_factor'] * 
            trailing_adj['sigma_factor']
        )
        
        return {'mean_adj': mean_adj, 'sigma_factor': sigma_factor} score difference effects
            self.adjustment_data['score_diff'] = {
                -3: {'mean_adj': 0.5, 'sigma_factor': 1.2},    # Down by 3+: +0.5 SOG, 20% more variance
                -2: {'mean_adj': 0.4, 'sigma_factor': 1.15},   # Down by 2: +0.4 SOG, 15% more variance
                -1: {'mean_adj': 0.2, 'sigma_factor': 1.1},    # Down by 1: +0.2 SOG, 10% more variance
                0: {'mean_adj': 0.0, 'sigma_factor': 1.0},     # Tied: no adjustment
                1: {'mean_adj': -0.1, 'sigma_factor': 0.95},   # Up by 1: -0.1 SOG, 5% less variance
                2: {'mean_adj': -0.3, 'sigma_factor': 0.9},    # Up by 2: -0.3 SOG, 10% less variance
                3: {'mean_adj': -0.5, 'sigma_factor': 0.9}     # Up by 3+: -0.5 SOG, 10% less variance
            }
            
            # Store period effects
            self.adjustment_data['period'] = {
                1: {'mean_adj': 0.0, 'sigma_factor': 1.0},     # 1st period: no adjustment
                2: {'mean_adj': 0.1, 'sigma_factor': 1.05},    # 2nd period: +0.1 SOG, 5% more variance
                3: {'mean_adj': 0.2, 'sigma_factor': 1.1},     # 3rd period: +0.2 SOG, 10% more variance
                4: {'mean_adj': 0.3, 'sigma_factor': 1.2},     # OT: +0.3 SOG, 20% more variance
                5: {'mean_adj': 0.4, 'sigma_factor': 1.3}      # 2OT+: +0.4 SOG, 30% more variance
            }
            
            # Store