from typing import List, Optional, Dict, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from database.models import Projection, BettingLine, Ticket, Player, Game, Team

class ProjectionRepository:
    """Repository for Projection data access"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, projection_id: int) -> Optional[Projection]:
        """Get projection by ID"""
        return self.db.query(Projection).filter(Projection.id == projection_id).first()
    
    def get_by_player_game(self, player_id: str, game_id: str) -> Optional[Projection]:
        """Get projection for a player in a specific game"""
        return (
            self.db.query(Projection)
            .filter(
                Projection.player_id == player_id,
                Projection.game_id == game_id
            )
            .order_by(desc(Projection.projection_time))
            .first()
        )
    
    def get_by_player(self, player_id: str) -> List[Projection]:
        """Get projections for a player"""
        return (
            self.db.query(Projection)
            .filter(Projection.player_id == player_id)
            .order_by(desc(Projection.projection_time))
            .all()
        )
    
    def get_by_game(self, game_id: str) -> List[Projection]:
        """Get projections for a game"""
        return (
            self.db.query(Projection)
            .filter(Projection.game_id == game_id)
            .order_by(desc(Projection.projection_time))
            .all()
        )
    
    def create(self, projection_data: dict) -> Projection:
        """Create a new projection"""
        projection = Projection(**projection_data)
        self.db.add(projection)
        self.db.commit()
        self.db.refresh(projection)
        return projection
    
    def update(self, projection_id: int, projection_data: dict) -> Optional[Projection]:
        """Update an existing projection"""
        projection = self.get_by_id(projection_id)
        if not projection:
            return None
        
        for key, value in projection_data.items():
            setattr(projection, key, value)
        
        self.db.commit()
        self.db.refresh(projection)
        return projection
    
    def bulk_create(self, projections_data: List[dict]) -> List[Projection]:
        """Create multiple projections at once"""
        projections = []
        for proj_data in projections_data:
            projection = Projection(**proj_data)
            self.db.add(projection)
            projections.append(projection)
        
        self.db.commit()
        for projection in projections:
            self.db.refresh(projection)
        
        return projections
    
    def calculate_probability(self, mean: float, std: float, line: float) -> float:
        """Calculate P(SOG > line) using Normal distribution"""
        # Continuous approximation of discrete SOG distribution
        return stats.norm.sf(line + 0.5, loc=mean, scale=std)


class BettingLineRepository:
    """Repository for BettingLine data access"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, line_id: int) -> Optional[BettingLine]:
        """Get betting line by ID"""
        return self.db.query(BettingLine).filter(BettingLine.id == line_id).first()
    
    def get_by_player_game(self, player_id: str, game_id: str) -> List[BettingLine]:
        """Get betting lines for a player in a specific game"""
        return (
            self.db.query(BettingLine)
            .filter(
                BettingLine.player_id == player_id,
                BettingLine.game_id == game_id
            )
            .order_by(desc(BettingLine.recorded_at))
            .all()
        )
    
    def get_by_player(self, player_id: str) -> List[BettingLine]:
        """Get betting lines for a player"""
        return (
            self.db.query(BettingLine)
            .filter(BettingLine.player_id == player_id)
            .order_by(desc(BettingLine.recorded_at))
            .all()
        )
    
    def get_by_game(self, game_id: str) -> List[BettingLine]:
        """Get betting lines for a game"""
        return (
            self.db.query(BettingLine)
            .filter(BettingLine.game_id == game_id)
            .order_by(desc(BettingLine.recorded_at))
            .all()
        )
    
    def get_current_lines(self) -> List[BettingLine]:
        """Get current betting lines"""
        # Get the most recent line for each player/game combo
        subquery = (
            self.db.query(
                BettingLine.player_id,
                BettingLine.game_id,
                func.max(BettingLine.recorded_at).label("max_date")
            )
            .group_by(BettingLine.player_id, BettingLine.game_id)
            .subquery()
        )
        
        return (
            self.db.query(BettingLine)
            .join(
                subquery,
                and_(
                    BettingLine.player_id == subquery.c.player_id,
                    BettingLine.game_id == subquery.c.game_id,
                    BettingLine.recorded_at == subquery.c.max_date
                )
            )
            .all()
        )
    
    def create(self, line_data: dict) -> BettingLine:
        """Create a new betting line"""
        line = BettingLine(**line_data)
        self.db.add(line)
        self.db.commit()
        self.db.refresh(line)
        return line
    
    def bulk_create(self, lines_data: List[dict]) -> List[BettingLine]:
        """Create multiple betting lines at once"""
        lines = []
        for line_data in lines_data:
            line = BettingLine(**line_data)
            self.db.add(line)
            lines.append(line)
        
        self.db.commit()
        for line in lines:
            self.db.refresh(line)
        
        return lines
    
    def american_to_implied(self, american_odds: int) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def find_edges(self, min_edge: float = 0.05) -> List[Dict]:
        """Find all betting lines with edge >= min_edge"""
        # Get current lines
        current_lines = self.get_current_lines()
        
        # Calculate edge for each line
        edges = []
        for line in current_lines:
            # Skip lines without model probability
            if line.model_probability is None:
                continue
            
            # Calculate implied probability if not already set
            if line.implied_probability is None:
                line.implied_probability = self.american_to_implied(line.american_odds)
                self.db.commit()
            
            # Calculate edge if not already set
            if line.edge is None:
                line.edge = line.model_probability - line.implied_probability
                self.db.commit()
            
            # Check if edge meets minimum threshold
            if line.edge >= min_edge:
                edges.append({
                    "line_id": line.id,
                    "player_id": line.player_id,
                    "game_id": line.game_id,
                    "line": line.line,
                    "american_odds": line.american_odds,
                    "model_probability": line.model_probability,
                    "implied_probability": line.implied_probability,
                    "edge": line.edge
                })
        
        # Sort by edge (descending)
        return sorted(edges, key=lambda x: x["edge"], reverse=True)


class TicketRepository:
    """Repository for Ticket data access"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, ticket_id: int) -> Optional[Ticket]:
        """Get ticket by ID"""
        return self.db.query(Ticket).filter(Ticket.id == ticket_id).first()
    
    def get_all(self, limit: int = 100) -> List[Ticket]:
        """Get all tickets"""
        return self.db.query(Ticket).order_by(desc(Ticket.created_at)).limit(limit).all()
    
    def create(self, ticket_data: dict) -> Ticket:
        """Create a new ticket"""
        import json
        
        # Convert lists to JSON
        for key in ["players_json", "games_json", "lines_json"]:
            if key.replace("_json", "") in ticket_data:
                ticket_data[key] = json.dumps(ticket_data.pop(key.replace("_json", "")))
        
        ticket = Ticket(**ticket_data)
        self.db.add(ticket)
        self.db.commit()
        self.db.refresh(ticket)
        return ticket
    
    def build_tickets(self, edges: List[Dict], max_legs: int = 3) -> List[Dict]:
        """Build tickets (parlays) from edges"""
        from itertools import combinations
        
        if not edges:
            return []
        
        # Convert to DataFrame for easier manipulation
        edges_df = pd.DataFrame(edges)
        
        # Convert American odds to decimal
        edges_df["decimal_odds"] = edges_df["american_odds"].apply(
            lambda x: x / 100 + 1 if x > 0 else 100 / abs(x) + 1
        )
        
        tickets = []
        
        # Single-leg tickets
        for _, edge in edges_df.iterrows():
            ticket = {
                "legs": 1,
                "players": [edge["player_id"]],
                "games": [edge["game_id"]],
                "combined_odds": float(edge["decimal_odds"]),
                "combined_probability": float(edge["model_probability"]),
                "expected_value": float(edge["decimal_odds"] * edge["model_probability"] - 1)
            }
            tickets.append(ticket)
        
        # Multi-leg tickets (2 to max_legs)
        for r in range(2, max_legs + 1):
            for combo in combinations(range(len(edges_df)), r):
                legs = edges_df.iloc[list(combo)]
                
                # Check for one leg per game
                if legs["game_id"].nunique() < len(legs):
                    continue
                
                # Calculate combined odds
                decimal_odds_product = legs["decimal_odds"].prod()
                probability_product = legs["model_probability"].prod()
                expected_value = decimal_odds_product * probability_product - 1
                
                # Only keep positive EV tickets
                if expected_value <= 0:
                    continue
                
                ticket = {
                    "legs": r,
                    "players": legs["player_id"].tolist(),
                    "games": legs["game_id"].tolist(),
                    "combined_odds": float(decimal_odds_product),
                    "combined_probability": float(probability_product),
                    "expected_value": float(expected_value)
                }
                tickets.append(ticket)
        
        # Sort by expected value (descending)
        return sorted(tickets, key=lambda x: x["expected_value"], reverse=True)
    
    def save_tickets(self, tickets: List[Dict]) -> List[Ticket]:
        """Save multiple tickets at once"""
        saved_tickets = []
        for ticket_data in tickets:
            ticket = self.create(ticket_data)
            saved_tickets.append(ticket)
        
        return saved_tickets