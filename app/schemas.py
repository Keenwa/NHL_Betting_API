from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import date, datetime

# Base schemas (used for shared attributes)
class TeamBase(BaseModel):
    team_id: str
    code: str
    name: str
    conference: Optional[str] = None
    division: Optional[str] = None
    tempo: Optional[float] = None
    sa_per_game: Optional[float] = None
    block_rate: Optional[float] = None

class PlayerBase(BaseModel):
    player_id: str
    name: str
    position: Optional[str] = None
    team_id: str

class GameBase(BaseModel):
    game_id: str
    season: Optional[str] = None
    date: Optional[date] = None
    is_playoff: bool = False
    home_team_id: str
    away_team_id: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: Optional[str] = None
    period: Optional[int] = None

class ShotBase(BaseModel):
    shot_id: str
    game_id: str
    shooter_id: str
    team_id: str
    event: str
    period: int
    time: str
    is_sog: bool
    is_goal: bool

class ProjectionBase(BaseModel):
    player_id: str
    game_id: str
    opponent_id: str
    final_mean: float
    final_std: float

class BettingLineBase(BaseModel):
    player_id: str
    game_id: str
    line: float
    american_odds: int

class TicketBase(BaseModel):
    legs: int
    combined_odds: float
    combined_probability: float
    expected_value: float
    players: List[str]
    games: List[str]

# Response schemas (used for API responses)
class Team(TeamBase):
    id: int
    
    class Config:
        from_attributes = True

class Player(PlayerBase):
    id: int
    is_top_shooter: Optional[bool] = None
    is_star: Optional[bool] = None
    is_top_six: Optional[bool] = None
    is_high_post_shooter: Optional[bool] = None
    avg_ev_toi: Optional[float] = None
    
    class Config:
        from_attributes = True

class PlayerWithTeam(Player):
    team: Optional[Team] = None

class Game(GameBase):
    id: int
    time_remaining: Optional[str] = None
    
    class Config:
        from_attributes = True

class GameWithTeams(Game):
    home_team: Optional[Team] = None
    away_team: Optional[Team] = None

class Shot(ShotBase):
    id: int
    shot_type: Optional[str] = None
    shot_was_on_goal: bool
    hit_post: Optional[bool] = None
    is_scoring_chance: Optional[bool] = None
    is_miss: bool
    is_block: bool
    is_post: bool
    
    class Config:
        from_attributes = True

class PlayerStatistics(BaseModel):
    id: int
    player_id: str
    season: str
    games_played: int
    total_shots: int
    total_sog: int
    sog_per_game: float
    mu: float
    sigma: float
    last_game_date: Optional[date] = None
    last_game_ev_toi: Optional[float] = None
    
    class Config:
        from_attributes = True

class Projection(ProjectionBase):
    id: int
    base_mean: float
    base_std: float
    opponent_adj_mean: float
    opponent_adj_std: float
    game_state_adj_mean: float
    game_state_adj_std: float
    period: int
    score_diff: int
    is_overtime: bool
    expected_pp: float
    actual_pp: float
    projection_time: datetime
    
    class Config:
        from_attributes = True

class BettingLine(BettingLineBase):
    id: int
    model_probability: Optional[float] = None
    implied_probability: Optional[float] = None
    edge: Optional[float] = None
    bookmaker: Optional[str] = None
    recorded_at: datetime
    
    class Config:
        from_attributes = True

class Ticket(TicketBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Request schemas (used for API requests)
class TeamCreate(TeamBase):
    pass

class TeamUpdate(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None
    conference: Optional[str] = None
    division: Optional[str] = None
    tempo: Optional[float] = None
    sa_per_game: Optional[float] = None
    block_rate: Optional[float] = None

class PlayerCreate(PlayerBase):
    is_top_shooter: Optional[bool] = None
    is_star: Optional[bool] = None
    is_top_six: Optional[bool] = None
    is_high_post_shooter: Optional[bool] = None
    avg_ev_toi: Optional[float] = None

class PlayerUpdate(BaseModel):
    name: Optional[str] = None
    position: Optional[str] = None
    team_id: Optional[str] = None
    is_top_shooter: Optional[bool] = None
    is_star: Optional[bool] = None
    is_top_six: Optional[bool] = None
    is_high_post_shooter: Optional[bool] = None
    avg_ev_toi: Optional[float] = None

class GameCreate(GameBase):
    time_remaining: Optional[str] = None

class GameUpdate(BaseModel):
    season: Optional[str] = None
    date: Optional[date] = None
    is_playoff: Optional[bool] = None
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: Optional[str] = None
    period: Optional[int] = None
    time_remaining: Optional[str] = None

class ShotCreate(ShotBase):
    shot_type: Optional[str] = None
    shot_was_on_goal: bool
    hit_post: Optional[bool] = None
    is_scoring_chance: Optional[bool] = None

class ProjectionCreate(ProjectionBase):
    base_mean: float
    base_std: float
    opponent_adj_mean: float
    opponent_adj_std: float
    game_state_adj_mean: float
    game_state_adj_std: float
    period: int
    score_diff: int
    is_overtime: bool
    expected_pp: float
    actual_pp: float

class BettingLineCreate(BettingLineBase):
    model_probability: Optional[float] = None
    implied_probability: Optional[float] = None
    edge: Optional[float] = None
    bookmaker: Optional[str] = None

class TicketCreate(TicketBase):
    pass

# Response for paginated lists
class PaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    items: List[Any]

# Error response
class ErrorResponse(BaseModel):
    detail: str

# SOG projection response
class SOGProjectionResponse(BaseModel):
    player: Player
    game: Game
    line: float
    final_mean: float
    final_std: float
    p_over: float
    model_details: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Detailed model parameters and adjustments"
    )

# Edge response
class EdgeResponse(BaseModel):
    player: Player
    game: Game
    line: float
    american_odds: int
    model_probability: float
    implied_probability: float
    edge: float
    
# Ticket response
class TicketResponse(BaseModel):
    legs: int
    players: List[Player]
    games: List[Game]
    lines: List[float]
    combined_odds: float
    combined_probability: float
    expected_value: float