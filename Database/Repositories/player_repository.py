from typing import List, Optional
from sqlalchemy.orm import Session
from database.models import Player, Team, PlayerStatistics

class PlayerRepository:
    """Repository for Player data access"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[Player]:
        """Get all players with pagination"""
        return self.db.query(Player).offset(skip).limit(limit).all()
    
    def get_by_id(self, player_id: str) -> Optional[Player]:
        """Get player by player_id"""
        return self.db.query(Player).filter(Player.player_id == player_id).first()
    
    def get_by_team(self, team_id: str) -> List[Player]:
        """Get all players for a team"""
        return self.db.query(Player).filter(Player.team_id == team_id).all()
    
    def get_by_position(self, position: str) -> List[Player]:
        """Get all players by position"""
        return self.db.query(Player).filter(Player.position == position).all()
    
    def get_top_shooters(self) -> List[Player]:
        """Get all players flagged as top shooters"""
        return self.db.query(Player).filter(Player.is_top_shooter == True).all()
    
    def get_stars(self) -> List[Player]:
        """Get all players flagged as stars"""
        return self.db.query(Player).filter(Player.is_star == True).all()
    
    def create(self, player_data: dict) -> Player:
        """Create a new player"""
        player = Player(**player_data)
        self.db.add(player)
        self.db.commit()
        self.db.refresh(player)
        return player
    
    def update(self, player_id: str, player_data: dict) -> Optional[Player]:
        """Update an existing player"""
        player = self.get_by_id(player_id)
        if not player:
            return None
        
        for key, value in player_data.items():
            setattr(player, key, value)
        
        self.db.commit()
        self.db.refresh(player)
        return player
    
    def delete(self, player_id: str) -> bool:
        """Delete a player"""
        player = self.get_by_id(player_id)
        if not player:
            return False
        
        self.db.delete(player)
        self.db.commit()
        return True
    
    def bulk_create_or_update(self, players_data: List[dict]) -> List[Player]:
        """Create or update multiple players at once"""
        result = []
        for player_data in players_data:
            player_id = player_data.get("player_id")
            
            existing_player = self.get_by_id(player_id) if player_id else None
            
            if existing_player:
                # Update existing player
                for key, value in player_data.items():
                    setattr(existing_player, key, value)
                result.append(existing_player)
            else:
                # Create new player
                player = Player(**player_data)
                self.db.add(player)
                result.append(player)
        
        self.db.commit()
        for player in result:
            self.db.refresh(player)
        
        return result
    
    def get_player_statistics(self, player_id: str, season: Optional[str] = None) -> List[PlayerStatistics]:
        """Get player statistics"""
        query = self.db.query(PlayerStatistics).filter(PlayerStatistics.player_id == player_id)
        
        if season:
            query = query.filter(PlayerStatistics.season == season)
            
        return query.all()
    
    def update_player_statistics(self, player_id: str, season: str, stats_data: dict) -> PlayerStatistics:
        """Update or create player statistics for a season"""
        existing_stats = (
            self.db.query(PlayerStatistics)
            .filter(
                PlayerStatistics.player_id == player_id,
                PlayerStatistics.season == season
            )
            .first()
        )
        
        if existing_stats:
            # Update existing stats
            for key, value in stats_data.items():
                setattr(existing_stats, key, value)
            self.db.commit()
            self.db.refresh(existing_stats)
            return existing_stats
        else:
            # Create new stats
            stats_data["player_id"] = player_id
            stats_data["season"] = season
            stats = PlayerStatistics(**stats_data)
            self.db.add(stats)
            self.db.commit()
            self.db.refresh(stats)
            return stats