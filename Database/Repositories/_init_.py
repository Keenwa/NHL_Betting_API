from sqlalchemy.orm import Session
from .team_repository import TeamRepository
from .player_repository import PlayerRepository
from .game_repository import GameRepository
from .shot_repository import ShotRepository
from .projection_repository import ProjectionRepository, BettingLineRepository, TicketRepository

class RepositoryFactory:
    """Factory for creating repositories"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_team_repository(self) -> TeamRepository:
        """Get team repository"""
        return TeamRepository(self.db)
    
    def get_player_repository(self) -> PlayerRepository:
        """Get player repository"""
        return PlayerRepository(self.db)
    
    def get_game_repository(self) -> GameRepository:
        """Get game repository"""
        return GameRepository(self.db)
    
    def get_shot_repository(self) -> ShotRepository:
        """Get shot repository"""
        return ShotRepository(self.db)
    
    def get_projection_repository(self) -> ProjectionRepository:
        """Get projection repository"""
        return ProjectionRepository(self.db)
    
    def get_betting_line_repository(self) -> BettingLineRepository:
        """Get betting line repository"""
        return BettingLineRepository(self.db)
    
    def get_ticket_repository(self) -> TicketRepository:
        """Get ticket repository"""
        return TicketRepository(self.db)