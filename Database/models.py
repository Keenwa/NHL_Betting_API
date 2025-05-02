from sqlalchemy import Column, ForeignKey, Integer, Float, String, Boolean, Date, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import datetime

Base = declarative_base()

class Team(Base):
    """NHL Team model"""
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    team_id = Column(String(10), unique=True, index=True, nullable=False)
    code = Column(String(3), unique=True, index=True, nullable=False)
    name = Column(String(50), nullable=False)
    conference = Column(String(20))
    division = Column(String(20))
    
    # Relationships
    players = relationship("Player", back_populates="team")
    games_home = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    games_away = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")
    
    # Team stats
    tempo = Column(Float)  # Average pace/tempo
    sa_per_game = Column(Float)  # Shots against per game
    block_rate = Column(Float)  # Block rate percentage
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Team {self.code}>"


class Player(Base):
    """NHL Player model"""
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    player_id = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    position = Column(String(10))
    team_id = Column(String(10), ForeignKey("teams.team_id"))
    
    # Player attributes for SOG model
    is_top_shooter = Column(Boolean, default=False)
    is_star = Column(Boolean, default=False)
    is_top_six = Column(Boolean, default=False)
    is_high_post_shooter = Column(Boolean, default=False)
    avg_ev_toi = Column(Float)  # Average even-strength time on ice
    
    # Relationships
    team = relationship("Team", back_populates="players")
    shots = relationship("Shot", back_populates="shooter")
    projections = relationship("Projection", back_populates="player")
    betting_lines = relationship("BettingLine", back_populates="player")
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Player {self.name}>"


class Game(Base):
    """NHL Game model"""
    __tablename__ = "games"

    id = Column(Integer, primary_key=True)
    game_id = Column(String(20), unique=True, index=True, nullable=False)
    season = Column(String(10), index=True)
    date = Column(Date, index=True)
    is_playoff = Column(Boolean, default=False)
    
    home_team_id = Column(String(10), ForeignKey("teams.team_id"))
    away_team_id = Column(String(10), ForeignKey("teams.team_id"))
    
    home_score = Column(Integer)
    away_score = Column(Integer)
    status = Column(String(20))  # scheduled, in-progress, final
    period = Column(Integer)
    time_remaining = Column(String(10))
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="games_home")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="games_away")
    shots = relationship("Shot", back_populates="game")
    betting_lines = relationship("BettingLine", back_populates="game")
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Game {self.game_id}>"


class Shot(Base):
    """NHL Shot event model (from MoneyPuck)"""
    __tablename__ = "shots"

    id = Column(Integer, primary_key=True)
    shot_id = Column(String(50), unique=True, nullable=False)
    game_id = Column(String(20), ForeignKey("games.game_id"), index=True)
    shooter_id = Column(String(10), ForeignKey("players.player_id"), index=True)
    team_id = Column(String(10), ForeignKey("teams.team_id"))
    
    # Shot details
    event = Column(String(10))  # SHOT, GOAL, MISS, BLOCK
    period = Column(Integer)
    time = Column(String(10))
    shot_type = Column(String(20))
    shot_was_on_goal = Column(Boolean)
    is_goal = Column(Boolean)
    hit_post = Column(Boolean, default=False)
    is_scoring_chance = Column(Boolean, default=False)
    
    # Derived flags (for easier querying)
    is_sog = Column(Boolean)  # Shot on goal
    is_miss = Column(Boolean)  # Missed shot
    is_block = Column(Boolean)  # Blocked shot
    is_post = Column(Boolean)  # Hit post
    
    # Relationships
    game = relationship("Game", back_populates="shots")
    shooter = relationship("Player", back_populates="shots")
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Shot {self.shot_id}>"


class PlayerStatistics(Base):
    """Player-level statistics for SOG model"""
    __tablename__ = "player_statistics"

    id = Column(Integer, primary_key=True)
    player_id = Column(String(10), ForeignKey("players.player_id"), index=True)
    season = Column(String(10), index=True)
    
    # SOG statistics
    games_played = Column(Integer)
    total_shots = Column(Integer)
    total_sog = Column(Integer)
    sog_per_game = Column(Float)
    
    # Weighted model parameters
    mu = Column(Float)  # Mean SOG per game with historical weighting
    sigma = Column(Float)  # Standard deviation
    
    # Last game info
    last_game_date = Column(Date)
    last_game_ev_toi = Column(Float)  # Last game even-strength TOI
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<PlayerStatistics player_id={self.player_id} season={self.season}>"


class Projection(Base):
    """SOG Projection model"""
    __tablename__ = "projections"

    id = Column(Integer, primary_key=True)
    player_id = Column(String(10), ForeignKey("players.player_id"), index=True)
    game_id = Column(String(20), ForeignKey("games.game_id"), index=True)
    opponent_id = Column(String(10), ForeignKey("teams.team_id"))
    
    # Base projection
    base_mean = Column(Float)
    base_std = Column(Float)
    
    # Adjusted values after each stage
    opponent_adj_mean = Column(Float)
    opponent_adj_std = Column(Float)
    game_state_adj_mean = Column(Float)
    game_state_adj_std = Column(Float)
    final_mean = Column(Float)
    final_std = Column(Float)
    
    # Game state flags used in projection
    period = Column(Integer)
    score_diff = Column(Integer)
    is_overtime = Column(Boolean, default=False)
    expected_pp = Column(Float)
    actual_pp = Column(Float)
    
    # Timestamp for when projection was made
    projection_time = Column(DateTime, server_default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="projections")
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Projection player_id={self.player_id} game_id={self.game_id}>"


class BettingLine(Base):
    """Betting line for SOG props"""
    __tablename__ = "betting_lines"

    id = Column(Integer, primary_key=True)
    player_id = Column(String(10), ForeignKey("players.player_id"), index=True)
    game_id = Column(String(20), ForeignKey("games.game_id"), index=True)
    
    # Line details
    line = Column(Float, nullable=False)  # SOG line (e.g., 2.5)
    american_odds = Column(Integer, nullable=False)  # American odds (e.g., -110)
    
    # Probability and edge
    model_probability = Column(Float)  # Model probability P(SOG > line)
    implied_probability = Column(Float)  # Implied probability from odds
    edge = Column(Float)  # model_probability - implied_probability
    
    # Bookmaker info
    bookmaker = Column(String(50))
    recorded_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="betting_lines")
    game = relationship("Game", back_populates="betting_lines")
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<BettingLine player_id={self.player_id} line={self.line} odds={self.american_odds}>"


class Ticket(Base):
    """Recommended betting ticket (parlay)"""
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True)
    legs = Column(Integer)  # Number of legs in parlay
    combined_odds = Column(Float)  # Decimal odds
    combined_probability = Column(Float)  # Combined probability
    expected_value = Column(Float)  # EV of ticket
    
    # Ticket details as JSON
    players_json = Column(Text)  # List of player_ids
    games_json = Column(Text)  # List of game_ids
    lines_json = Column(Text)  # List of lines
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Ticket legs={self.legs} ev={self.expected_value}>"