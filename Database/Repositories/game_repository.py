from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import date, datetime, timedelta
from database.models import Game, Team

class GameRepository:
    """Repository for Game data access"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, game_id: str) -> Optional[Game]:
        """Get game by game_id"""
        return self.db.query(Game).filter(Game.game_id == game_id).first()
    
    def get_by_date(self, game_date: date) -> List[Game]:
        """Get games by date"""
        return self.db.query(Game).filter(Game.date == game_date).all()
    
    def get_upcoming_games(self, days: int = 7) -> List[Game]:
        """Get upcoming games for the next X days"""
        today = date.today()
        end_date = today + timedelta(days=days)
        
        return (
            self.db.query(Game)
            .filter(
                Game.date >= today,
                Game.date <= end_date,
                Game.status != "final"
            )
            .order_by(Game.date)
            .all()
        )
    
    def get_by_team(self, team_id: str, limit: int = 10) -> List[Game]:
        """Get games for a specific team"""
        return (
            self.db.query(Game)
            .filter(
                or_(
                    Game.home_team_id == team_id,
                    Game.away_team_id == team_id
                )
            )
            .order_by(Game.date.desc())
            .limit(limit)
            .all()
        )
    
    def get_playoff_games(self, season: str = None) -> List[Game]:
        """Get playoff games, optionally filtering by season"""
        query = self.db.query(Game).filter(Game.is_playoff == True)
        
        if season:
            query = query.filter(Game.season == season)
            
        return query.order_by(Game.date.desc()).all()
    
    def create(self, game_data: dict) -> Game:
        """Create a new game"""
        game = Game(**game_data)
        self.db.add(game)
        self.db.commit()
        self.db.refresh(game)
        return game
    
    def update(self, game_id: str, game_data: dict) -> Optional[Game]:
        """Update an existing game"""
        game = self.get_by_id(game_id)
        if not game:
            return None
        
        for key, value in game_data.items():
            setattr(game, key, value)
        
        self.db.commit()
        self.db.refresh(game)
        return game
    
    def bulk_create_or_update(self, games_data: List[dict]) -> List[Game]:
        """Create or update multiple games at once"""
        result = []
        for game_data in games_data:
            game_id = game_data.get("game_id")
            
            existing_game = self.get_by_id(game_id) if game_id else None
            
            if existing_game:
                # Update existing game
                for key, value in game_data.items():
                    setattr(existing_game, key, value)
                result.append(existing_game)
            else:
                # Create new game
                game = Game(**game_data)
                self.db.add(game)
                result.append(game)
        
        self.db.commit()
        for game in result:
            self.db.refresh(game)
        
        return result
    
    def get_team_standings(self, season: str) -> List[Dict]:
        """Get team standings for a season"""
        # Get all completed games for the season
        games = (
            self.db.query(Game)
            .filter(
                Game.season == season,
                Game.status == "final"
            )
            .all()
        )
        
        # Calculate standings
        standings = {}
        
        for game in games:
            # Process home team
            if game.home_team_id not in standings:
                standings[game.home_team_id] = {
                    "team_id": game.home_team_id,
                    "games_played": 0,
                    "wins": 0,
                    "losses": 0,
                    "ot_losses": 0,
                    "points": 0,
                    "goals_for": 0,
                    "goals_against": 0
                }
            
            # Process away team
            if game.away_team_id not in standings:
                standings[game.away_team_id] = {
                    "team_id": game.away_team_id,
                    "games_played": 0,
                    "wins": 0,
                    "losses": 0,
                    "ot_losses": 0,
                    "points": 0,
                    "goals_for": 0,
                    "goals_against": 0
                }
            
            # Update standings based on game result
            home_team = standings[game.home_team_id]
            away_team = standings[game.away_team_id]
            
            home_team["games_played"] += 1
            away_team["games_played"] += 1
            
            home_team["goals_for"] += game.home_score
            home_team["goals_against"] += game.away_score
            
            away_team["goals_for"] += game.away_score
            away_team["goals_against"] += game.home_score
            
            if game.home_score > game.away_score:
                # Home team won
                home_team["wins"] += 1
                home_team["points"] += 2
                
                if game.period > 3:  # Overtime/shootout
                    away_team["ot_losses"] += 1
                    away_team["points"] += 1
                else:
                    away_team["losses"] += 1
            else:
                # Away team won
                away_team["wins"] += 1
                away_team["points"] += 2
                
                if game.period > 3:  # Overtime/shootout
                    home_team["ot_losses"] += 1
                    home_team["points"] += 1
                else:
                    home_team["losses"] += 1
        
        # Convert dict to list and sort by points
        return sorted(
            list(standings.values()),
            key=lambda x: (x["points"], x["wins"]),
            reverse=True
        )