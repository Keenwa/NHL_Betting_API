from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from typing import List, Optional

from database.database import get_db
from database.repositories import RepositoryFactory
from . import schemas

router = APIRouter()

# Teams endpoints
@router.get("/teams", response_model=List[schemas.Team], tags=["teams"])
def get_all_teams(
    db: Session = Depends(get_db)
):
    """Get all teams"""
    repo_factory = RepositoryFactory(db)
    team_repo = repo_factory.get_team_repository()
    return team_repo.get_all()

@router.get("/teams/{team_id}", response_model=schemas.Team, tags=["teams"])
def get_team(
    team_id: str = Path(..., description="Team ID or code"),
    db: Session = Depends(get_db)
):
    """Get a team by ID or code"""
    repo_factory = RepositoryFactory(db)
    team_repo = repo_factory.get_team_repository()
    
    # Try by ID first
    team = team_repo.get_by_id(team_id)
    
    # If not found, try by code
    if not team:
        team = team_repo.get_by_code(team_id)
    
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    
    return team

@router.post("/teams", response_model=schemas.Team, tags=["teams"])
def create_team(
    team: schemas.TeamCreate,
    db: Session = Depends(get_db)
):
    """Create a new team"""
    repo_factory = RepositoryFactory(db)
    team_repo = repo_factory.get_team_repository()
    
    # Check if team already exists
    existing_team = team_repo.get_by_id(team.team_id)
    if existing_team:
        raise HTTPException(status_code=400, detail=f"Team with ID {team.team_id} already exists")
    
    return team_repo.create(team.dict())

@router.put("/teams/{team_id}", response_model=schemas.Team, tags=["teams"])
def update_team(
    team: schemas.TeamUpdate,
    team_id: str = Path(..., description="Team ID"),
    db: Session = Depends(get_db)
):
    """Update a team"""
    repo_factory = RepositoryFactory(db)
    team_repo = repo_factory.get_team_repository()
    
    # Check if team exists
    existing_team = team_repo.get_by_id(team_id)
    if not existing_team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    
    updated_team = team_repo.update(team_id, team.dict(exclude_unset=True))
    return updated_team

# Players endpoints
@router.get("/players", response_model=List[schemas.Player], tags=["players"])
def get_all_players(
    skip: int = Query(0, description="Number of players to skip"),
    limit: int = Query(100, description="Maximum number of players to return"),
    team_id: Optional[str] = Query(None, description="Filter by team ID"),
    db: Session = Depends(get_db)
):
    """Get all players with pagination and optional filtering"""
    repo_factory = RepositoryFactory(db)
    player_repo = repo_factory.get_player_repository()
    
    if team_id:
        return player_repo.get_by_team(team_id)
    else:
        return player_repo.get_all(skip=skip, limit=limit)

@router.get("/players/{player_id}", response_model=schemas.PlayerWithTeam, tags=["players"])
def get_player(
    player_id: str = Path(..., description="Player ID"),
    db: Session = Depends(get_db)
):
    """Get a player by ID"""
    repo_factory = RepositoryFactory(db)
    player_repo = repo_factory.get_player_repository()
    
    player = player_repo.get_by_id(player_id)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    
    return player

@router.post("/players", response_model=schemas.Player, tags=["players"])
def create_player(
    player: schemas.PlayerCreate,
    db: Session = Depends(get_db)
):
    """Create a new player"""
    repo_factory = RepositoryFactory(db)
    player_repo = repo_factory.get_player_repository()
    
    # Check if player already exists
    existing_player = player_repo.get_by_id(player.player_id)
    if existing_player:
        raise HTTPException(status_code=400, detail=f"Player with ID {player.player_id} already exists")
    
    return player_repo.create(player.dict())

@router.put("/players/{player_id}", response_model=schemas.Player, tags=["players"])
def update_player(
    player: schemas.PlayerUpdate,
    player_id: str = Path(..., description="Player ID"),
    db: Session = Depends(get_db)
):
    """Update a player"""
    repo_factory = RepositoryFactory(db)
    player_repo = repo_factory.get_player_repository()
    
    # Check if player exists
    existing_player = player_repo.get_by_id(player_id)
    if not existing_player:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    
    updated_player = player_repo.update(player_id, player.dict(exclude_unset=True))
    return updated_player

@router.get("/players/{player_id}/statistics", response_model=List[schemas.PlayerStatistics], tags=["players"])
def get_player_statistics(
    player_id: str = Path(..., description="Player ID"),
    season: Optional[str] = Query(None, description="Filter by season"),
    db: Session = Depends(get_db)
):
    """Get player statistics"""
    repo_factory = RepositoryFactory(db)
    player_repo = repo_factory.get_player_repository()
    
    # Check if player exists
    player = player_repo.get_by_id(player_id)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    
    return player_repo.get_player_statistics(player_id, season)

# Games endpoints
@router.get("/games", response_model=List[schemas.Game], tags=["games"])
def get_games(
    team_id: Optional[str] = Query(None, description="Filter by team ID"),
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    upcoming: Optional[bool] = Query(False, description="Get upcoming games"),
    db: Session = Depends(get_db)
):
    """Get games with optional filtering"""
    repo_factory = RepositoryFactory(db)
    game_repo = repo_factory.get_game_repository()
    
    if upcoming:
        return game_repo.get_upcoming_games()
    elif team_id:
        return game_repo.get_by_team(team_id)
    elif date:
        from datetime import datetime
        game_date = datetime.strptime(date, "%Y-%m-%d").date()
        return game_repo.get_by_date(game_date)
    else:
        # Default to upcoming games
        return game_repo.get_upcoming_games()

@router.get("/games/{game_id}", response_model=schemas.GameWithTeams, tags=["games"])
def get_game(
    game_id: str = Path(..., description="Game ID"),
    db: Session = Depends(get_db)
):
    """Get a game by ID"""
    repo_factory = RepositoryFactory(db)
    game_repo = repo_factory.get_game_repository()
    
    game = game_repo.get_by_id(game_id)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    
    return game

# Projections endpoints
@router.get("/projections/sog/{player_id}", response_model=schemas.SOGProjectionResponse, tags=["projections"])
def get_player_sog_projection(
    player_id: str = Path(..., description="Player ID"),
    game_id: str = Query(..., description="Game ID"),
    line: float = Query(..., description="SOG line to exceed"),
    period: int = Query(3, description="Current period"),
    score_diff: int = Query(0, description="Score difference (player team - opponent)"),
    is_overtime: bool = Query(False, description="Whether game is in overtime"),
    expected_pp: float = Query(2.0, description="Expected power play opportunities"),
    actual_pp: float = Query(2.0, description="Actual power play opportunities"),
    last_game_ev_toi: Optional[float] = Query(None, description="Even strength TOI in last game"),
    db: Session = Depends(get_db)
):
    """Get SOG projection for a specific player"""
    from scipy import stats
    
    repo_factory = RepositoryFactory(db)
    player_repo = repo_factory.get_player_repository()
    game_repo = repo_factory.get_game_repository()
    projection_repo = repo_factory.get_projection_repository()
    
    # Get player and game
    player = player_repo.get_by_id(player_id)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    
    game = game_repo.get_by_id(game_id)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    
    # Determine opponent
    opponent_id = game.away_team_id if player.team_id == game.home_team_id else game.home_team_id
    
    # Get player statistics
    player_stats = player_repo.get_player_statistics(player_id, "current")
    if not player_stats:
        raise HTTPException(status_code=404, detail=f"No statistics found for player {player_id}")
    
    # Get base mu and sigma
    player_stat = player_stats[0]
    base_mean = player_stat.mu
    base_std = player_stat.sigma
    
    # Apply simple adjustments for demo
    # In a real implementation, this would use your full model
    adjusted_mean = base_mean
    adjusted_std = base_std
    
    # Calculate probability
    p_over = stats.norm.sf(line + 0.5, loc=adjusted_mean, scale=adjusted_std)
    
    return {
        "player": player,
        "game": game,
        "line": line,
        "final_mean": adjusted_mean,
        "final_std": adjusted_std,
        "p_over": p_over,
        "model_details": {
            "base_mean": base_mean,
            "base_std": base_std,
            "period": period,
            "score_diff": score_diff,
            "is_overtime": is_overtime,
            "expected_pp": expected_pp,
            "actual_pp": actual_pp,
            "last_game_ev_toi": last_game_ev_toi
        }
    }

@router.get("/projections/edges", response_model=List[schemas.EdgeResponse], tags=["projections"])
def get_sog_edges(
    min_edge: float = Query(0.05, description="Minimum edge threshold (decimal)"),
    db: Session = Depends(get_db)
):
    """Get all SOG edges above the threshold"""
    repo_factory = RepositoryFactory(db)
    line_repo = repo_factory.get_betting_line_repository()
    player_repo = repo_factory.get_player_repository()
    game_repo = repo_factory.get_game_repository()
    
    # Find edges
    edges = line_repo.find_edges(min_edge=min_edge)
    
    # Enrich with player and game data
    result = []
    for edge in edges:
        player = player_repo.get_by_id(edge["player_id"])
        game = game_repo.get_by_id(edge["game_id"])
        
        if player and game:
            result.append({
                "player": player,
                "game": game,
                "line": edge["line"],
                "american_odds": edge["american_odds"],
                "model_probability": edge["model_probability"],
                "implied_probability": edge["implied_probability"],
                "edge": edge["edge"]
            })
    
    return result

@router.get("/projections/tickets", response_model=List[schemas.TicketResponse], tags=["projections"])
def get_sog_tickets(
    max_legs: int = Query(3, description="Maximum number of legs in a parlay"),
    min_edge: float = Query(0.05, description="Minimum edge threshold (decimal)"),
    db: Session = Depends(get_db)
):
    """Get SOG ticket recommendations (parlays)"""
    repo_factory = RepositoryFactory(db)
    line_repo = repo_factory.get_betting_line_repository()
    ticket_repo = repo_factory.get_ticket_repository()
    player_repo = repo_factory.get_player_repository()
    game_repo = repo_factory.get_game_repository()
    
    # Find edges
    edges = line_repo.find_edges(min_edge=min_edge)
    
    # Build tickets
    tickets = ticket_repo.build_tickets(edges, max_legs=max_legs)
    
    # Enrich with player and game data
    result = []
    for ticket in tickets:
        # Get players and games
        players = [player_repo.get_by_id(player_id) for player_id in ticket["players"]]
        games = [game_repo.get_by_id(game_id) for game_id in ticket["games"]]
        
        # Filter out any missing players or games
        players = [p for p in players if p]
        games = [g for g in games if g]
        
        if len(players) == len(ticket["players"]) and len(games) == len(ticket["games"]):
            # Get lines for each player
            lines = []
            for i, player_id in enumerate(ticket["players"]):
                game_id = ticket["games"][i]
                bet_lines = line_repo.get_by_player_game(player_id, game_id)
                if bet_lines:
                    lines.append(bet_lines[0].line)
                else:
                    lines.append(0.0)
            
            result.append({
                "legs": ticket["legs"],
                "players": players,
                "games": games,
                "lines": lines,
                "combined_odds": ticket["combined_odds"],
                "combined_probability": ticket["combined_probability"],
                "expected_value": ticket["expected_value"]
            })
    
    return result

@router.post("/projections/refresh", tags=["admin"])
def refresh_projections(
    db: Session = Depends(get_db)
):
    """Refresh all projections based on latest data"""
    try:
        # In a real implementation, this would trigger a background task
        # to update all projections.
        
        return {"status": "success", "message": "Projections refresh triggered"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing projections: {str(e)}")

# Betting Lines endpoints
@router.get("/betting-lines", response_model=List[schemas.BettingLine], tags=["betting"])
def get_betting_lines(
    player_id: Optional[str] = Query(None, description="Filter by player ID"),
    game_id: Optional[str] = Query(None, description="Filter by game ID"),
    current_only: bool = Query(True, description="Get only current lines"),
    db: Session = Depends(get_db)
):
    """Get betting lines with optional filtering"""
    repo_factory = RepositoryFactory(db)
    line_repo = repo_factory.get_betting_line_repository()
    
    if current_only:
        lines = line_repo.get_current_lines()
        if player_id:
            lines = [line for line in lines if line.player_id == player_id]
        if game_id:
            lines = [line for line in lines if line.game_id == game_id]
        return lines
    elif player_id and game_id:
        return line_repo.get_by_player_game(player_id, game_id)
    elif player_id:
        return line_repo.get_by_player(player_id)
    elif game_id:
        return line_repo.get_by_game(game_id)
    else:
        # Default to current lines
        return line_repo.get_current_lines()

@router.post("/betting-lines", response_model=schemas.BettingLine, tags=["betting"])
def create_betting_line(
    line: schemas.BettingLineCreate,
    db: Session = Depends(get_db)
):
    """Create a new betting line"""
    repo_factory = RepositoryFactory(db)
    line_repo = repo_factory.get_betting_line_repository()
    player_repo = repo_factory.get_player_repository()
    game_repo = repo_factory.get_game_repository()
    
    # Check if player and game exist
    player = player_repo.get_by_id(line.player_id)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {line.player_id} not found")
    
    game = game_repo.get_by_id(line.game_id)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {line.game_id} not found")
    
    # Calculate implied probability if not provided
    if line.implied_probability is None:
        line.implied_probability = line_repo.american_to_implied(line.american_odds)
    
    # Create betting line
    return line_repo.create(line.dict())

@router.post("/betting-lines/bulk", response_model=List[schemas.BettingLine], tags=["betting"])
def create_betting_lines_bulk(
    lines: List[schemas.BettingLineCreate],
    db: Session = Depends(get_db)
):
    """Create multiple betting lines at once"""
    repo_factory = RepositoryFactory(db)
    line_repo = repo_factory.get_betting_line_repository()
    
    # Calculate implied probabilities if not provided
    for line in lines:
        if line.implied_probability is None:
            line.implied_probability = line_repo.american_to_implied(line.american_odds)
    
    # Create betting lines
    return line_repo.bulk_create([line.dict() for line in lines])

# Tickets endpoints
@router.get("/tickets", response_model=List[schemas.Ticket], tags=["tickets"])
def get_tickets(
    limit: int = Query(100, description="Maximum number of tickets to return"),
    db: Session = Depends(get_db)
):
    """Get all tickets"""
    repo_factory = RepositoryFactory(db)
    ticket_repo = repo_factory.get_ticket_repository()
    
    return ticket_repo.get_all(limit=limit)

@router.post("/tickets", response_model=schemas.Ticket, tags=["tickets"])
def create_ticket(
    ticket: schemas.TicketCreate,
    db: Session = Depends(get_db)
):
    """Create a new ticket"""
    repo_factory = RepositoryFactory(db)
    ticket_repo = repo_factory.get_ticket_repository()
    
    return ticket_repo.create(ticket.dict())

@router.get("/tickets/build", response_model=List[schemas.TicketResponse], tags=["tickets"])
def build_tickets(
    max_legs: int = Query(3, description="Maximum number of legs in a parlay"),
    min_edge: float = Query(0.05, description="Minimum edge threshold (decimal)"),
    db: Session = Depends(get_db)
):
    """Build ticket recommendations"""
    # This is the same as get_sog_tickets
    return get_sog_tickets(max_legs=max_legs, min_edge=min_edge, db=db)

@router.post("/tickets/save", response_model=List[schemas.Ticket], tags=["tickets"])
def save_tickets(
    max_legs: int = Query(3, description="Maximum number of legs in a parlay"),
    min_edge: float = Query(0.05, description="Minimum edge threshold (decimal)"),
    db: Session = Depends(get_db)
):
    """Build and save ticket recommendations"""
    repo_factory = RepositoryFactory(db)
    line_repo = repo_factory.get_betting_line_repository()
    ticket_repo = repo_factory.get_ticket_repository()
    
    # Find edges
    edges = line_repo.find_edges(min_edge=min_edge)
    
    # Build tickets
    tickets = ticket_repo.build_tickets(edges, max_legs=max_legs)
    
    # Save tickets
    return ticket_repo.save_tickets(tickets)