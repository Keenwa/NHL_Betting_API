from typing import List, Optional
from sqlalchemy.orm import Session
from database.models import Team

class TeamRepository:
    """Repository for Team data access"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_all(self) -> List[Team]:
        """Get all teams"""
        return self.db.query(Team).all()
    
    def get_by_id(self, team_id: str) -> Optional[Team]:
        """Get team by team_id"""
        return self.db.query(Team).filter(Team.team_id == team_id).first()
    
    def get_by_code(self, code: str) -> Optional[Team]:
        """Get team by code (three-letter code)"""
        return self.db.query(Team).filter(Team.code == code).first()
    
    def get_by_conference(self, conference: str) -> List[Team]:
        """Get teams by conference"""
        return self.db.query(Team).filter(Team.conference == conference).all()
    
    def get_by_division(self, division: str) -> List[Team]:
        """Get teams by division"""
        return self.db.query(Team).filter(Team.division == division).all()
    
    def create(self, team_data: dict) -> Team:
        """Create a new team"""
        team = Team(**team_data)
        self.db.add(team)
        self.db.commit()
        self.db.refresh(team)
        return team
    
    def update(self, team_id: str, team_data: dict) -> Optional[Team]:
        """Update an existing team"""
        team = self.get_by_id(team_id)
        if not team:
            return None
        
        for key, value in team_data.items():
            setattr(team, key, value)
        
        self.db.commit()
        self.db.refresh(team)
        return team
    
    def delete(self, team_id: str) -> bool:
        """Delete a team"""
        team = self.get_by_id(team_id)
        if not team:
            return False
        
        self.db.delete(team)
        self.db.commit()
        return True
    
    def bulk_create_or_update(self, teams_data: List[dict]) -> List[Team]:
        """Create or update multiple teams at once"""
        result = []
        for team_data in teams_data:
            team_id = team_data.get("team_id")
            
            existing_team = self.get_by_id(team_id) if team_id else None
            
            if existing_team:
                # Update existing team
                for key, value in team_data.items():
                    setattr(existing_team, key, value)
                result.append(existing_team)
            else:
                # Create new team
                team = Team(**team_data)
                self.db.add(team)
                result.append(team)
        
        self.db.commit()
        for team in result:
            self.db.refresh(team)
        
        return result