from typing import List, Optional, Dict, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
import pandas as pd
from database.models import Shot, Game, Player

class ShotRepository:
    """Repository for Shot data access"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, shot_id: str) -> Optional[Shot]:
        """Get shot by shot_id"""
        return self.db.query(Shot).filter(Shot.shot_id == shot_id).first()
    
    def get_by_game(self, game_id: str) -> List[Shot]:
        """Get all shots for a game"""
        return self.db.query(Shot).filter(Shot.game_id == game_id).all()
    
    def get_by_player(self, player_id: str, limit: int = 100) -> List[Shot]:
        """Get shots by a player"""
        return self.db.query(Shot).filter(Shot.shooter_id == player_id).limit(limit).all()
    
    def get_by_team(self, team_id: str, limit: int = 100) -> List[Shot]:
        """Get shots by a team"""
        return self.db.query(Shot).filter(Shot.team_id == team_id).limit(limit).all()
    
    def create(self, shot_data: dict) -> Shot:
        """Create a new shot record"""
        # Derive the flags
        event = shot_data.get("event", "")
        shot_was_on_goal = shot_data.get("shot_was_on_goal", False)
        is_goal = shot_data.get("is_goal", False)
        hit_post = shot_data.get("hit_post", False)
        
        # Set the derived flags
        shot_data["is_sog"] = (event == "SHOT" and shot_was_on_goal) or (event == "GOAL")
        shot_data["is_miss"] = event == "MISS" or (event == "SHOT" and not shot_was_on_goal)
        shot_data["is_block"] = event == "BLOCK"
        shot_data["is_post"] = hit_post
        
        shot = Shot(**shot_data)
        self.db.add(shot)
        self.db.commit()
        self.db.refresh(shot)
        return shot
    
    def bulk_create(self, shots_data: List[dict]) -> List[Shot]:
        """Create multiple shot records at once"""
        shots = []
        for shot_data in shots_data:
            # Derive the flags
            event = shot_data.get("event", "")
            shot_was_on_goal = shot_data.get("shot_was_on_goal", False)
            is_goal = shot_data.get("is_goal", False)
            hit_post = shot_data.get("hit_post", False)
            
            # Set the derived flags
            shot_data["is_sog"] = (event == "SHOT" and shot_was_on_goal) or (event == "GOAL")
            shot_data["is_miss"] = event == "MISS" or (event == "SHOT" and not shot_was_on_goal)
            shot_data["is_block"] = event == "BLOCK"
            shot_data["is_post"] = hit_post
            
            shot = Shot(**shot_data)
            shots.append(shot)
            self.db.add(shot)
        
        self.db.commit()
        return shots
    
    def verify_data_integrity(self) -> Tuple[bool, Dict]:
        """
        Verify that team-SOG + MISS + BLOCK + POST = total attempts for each game/team
        
        Returns:
            Tuple containing:
            - Boolean indicating if integrity check passed
            - Dictionary with details about any failures
        """
        # Get counts by game and team
        counts = (
            self.db.query(
                Shot.game_id,
                Shot.team_id,
                func.sum(Shot.is_sog.cast(Integer)).label("sog_count"),
                func.sum(Shot.is_miss.cast(Integer)).label("miss_count"),
                func.sum(Shot.is_block.cast(Integer)).label("block_count"),
                func.sum(Shot.is_post.cast(Integer)).label("post_count"),
                func.count(Shot.id).label("total_count")
            )
            .group_by(Shot.game_id, Shot.team_id)
            .all()
        )
        
        failures = []
        for game_id, team_id, sog, miss, block, post, total in counts:
            calculated = sog + miss + block + post
            if calculated != total:
                failures.append({
                    "game_id": game_id,
                    "team_id": team_id,
                    "sog": sog,
                    "miss": miss,
                    "block": block,
                    "post": post,
                    "total": total,
                    "calculated": calculated,
                    "difference": total - calculated
                })
        
        return len(failures) == 0, {"passed": len(failures) == 0, "failures": failures}
    
    def get_player_sog_per_game(self, player_id: str) -> pd.DataFrame:
        """
        Get shots on goal per game for a player
        
        Returns:
            DataFrame with game_id, date, is_playoff, and sog_count columns
        """
        shots = (
            self.db.query(
                Shot.game_id,
                Game.date,
                Game.is_playoff,
                func.sum(Shot.is_sog.cast(Integer)).label("sog_count")
            )
            .join(Game, Shot.game_id == Game.game_id)
            .filter(Shot.shooter_id == player_id)
            .group_by(Shot.game_id, Game.date, Game.is_playoff)
            .order_by(Game.date.desc())
            .all()
        )
        
        # Convert to DataFrame
        return pd.DataFrame(shots, columns=["game_id", "date", "is_playoff", "sog_count"])
    
    def get_player_sog_stats(self, player_id: str) -> Dict:
        """
        Get SOG statistics for a player
        
        Returns:
            Dictionary with various SOG statistics
        """
        # Get player's shot data
        shots = (
            self.db.query(
                func.count(Shot.id).label("total_shots"),
                func.sum(Shot.is_sog.cast(Integer)).label("total_sog"),
                func.count(distinct(Shot.game_id)).label("games_played")
            )
            .filter(Shot.shooter_id == player_id)
            .one()
        )
        
        # Get SOG per game
        sog_per_game_df = self.get_player_sog_per_game(player_id)
        
        # Calculate statistics
        total_shots, total_sog, games_played = shots
        sog_per_game = total_sog / games_played if games_played > 0 else 0
        
        # Calculate weighted mean and std
        if not sog_per_game_df.empty:
            # Apply weighting: Last playoff game 50% → G-2 25% → older playoff 15% → full RS 10%
            playoff_games = sog_per_game_df[sog_per_game_df["is_playoff"]].copy()
            regular_season = sog_per_game_df[~sog_per_game_df["is_playoff"]].copy()
            
            weights = []
            sog_values = []
            
            # Last playoff game: 50%
            if not playoff_games.empty:
                last_playoff_sog = playoff_games.iloc[0]["sog_count"]
                weights.append(0.5)
                sog_values.append(last_playoff_sog)
                
                # G-2 playoff game: 25%
                if len(playoff_games) > 1:
                    g2_playoff_sog = playoff_games.iloc[1]["sog_count"]
                    weights.append(0.25)
                    sog_values.append(g2_playoff_sog)
                    
                # Older playoff games: 15%
                if len(playoff_games) > 2:
                    older_playoff_sog = playoff_games.iloc[2:]["sog_count"].mean()
                    weights.append(0.15)
                    sog_values.append(older_playoff_sog)
            
            # Regular season: 10%
            if not regular_season.empty:
                rs_sog = regular_season["sog_count"].mean()
                weights.append(0.10)
                sog_values.append(rs_sog)
                
            # Normalize weights
            import numpy as np
            weights = np.array(weights) / sum(weights)
            
            # Calculate weighted mean
            weighted_mean = np.sum(np.array(weights) * np.array(sog_values))
            
            # Calculate weighted variance and std
            if len(sog_values) > 1:
                weighted_var = np.sum(weights * (np.array(sog_values) - weighted_mean) ** 2)
                weighted_std = np.sqrt(weighted_var)
            else:
                # Default std if only one game
                weighted_std = 0.5
        else:
            weighted_mean = sog_per_game
            weighted_std = 0.5
        
        return {
            "player_id": player_id,
            "total_shots": total_shots,
            "total_sog": total_sog,
            "games_played": games_played,
            "sog_per_game": sog_per_game,
            "weighted_mean": weighted_mean,
            "weighted_std": weighted_std
        }
    
    def update_player_attributes(self, player_id: str, attributes: Dict) -> bool:
        """
        Update player attributes based on shot data analysis
        
        Args:
            player_id: ID of the player
            attributes: Dictionary of attributes to update
        
        Returns:
            Boolean indicating success
        """
        player = self.db.query(Player).filter(Player.player_id == player_id).first()
        if not player:
            return False
        
        for key, value in attributes.items():
            if hasattr(player, key):
                setattr(player, key, value)
        
        self.db.commit()
        return True