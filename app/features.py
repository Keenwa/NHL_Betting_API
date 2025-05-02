import pandas as pd
import os
import glob
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from shot_features import (
    load_shot_data,
    process_shots,
    compute_weighted_mu_sigma,
    apply_opponent_context,
    apply_game_state_adjustments,
    apply_player_specific_quirks,
    calculate_final_projection,
    get_nhl_averages,
    validate_shot_data_integrity
)

router = APIRouter(prefix="/features", tags=["features"])

# Initialize shot data
shot_data = None
processed_shots = None
nhl_avg_tempo = None
nhl_avg_sa_per_game = None

try:
    shot_data = load_shot_data("data")
    processed_shots = process_shots(shot_data)
    nhl_avg_tempo, nhl_avg_sa_per_game = get_nhl_averages(shot_data)
    print(f"✅ Shot data loaded: {len(shot_data)} shots, avg tempo: {nhl_avg_tempo:.2f}, avg SA/GP: {nhl_avg_sa_per_game:.2f}")
except Exception as e:
    print(f"⚠️ Error loading shot data: {e}")


# ========================
# Feature Endpoints
# ========================

@router.get("/mu-sigma")
def get_player_mu_sigma(
    player_id: Optional[str] = None,
    team_id: Optional[str] = None
):
    """
    Get weighted mu (mean) and sigma (std) for players.
    Filter by player_id or team_id if provided.
    """
    global processed_shots
    
    if processed_shots is None:
        raise HTTPException(status_code=500, detail="Shot data not loaded")
    
    try:
        # Compute mu/sigma for all players
        player_stats = compute_weighted_mu_sigma(processed_shots)
        
        # Apply filters
        if player_id:
            player_stats = player_stats[player_stats["shooterPlayerId"] == player_id]
        
        if team_id:
            player_stats = player_stats[player_stats["teamCode"] == team_id]
        
        if player_stats.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified filters")
        
        return player_stats.to_dict(orient="records")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating mu/sigma: {str(e)}")


@router.get("/opponent-context")
def get_opponent_context(
    player_id: str = Query(..., description="Player ID"),
    opponent_id: str = Query(..., description="Opponent team ID")
):
    """
    Apply opponent context to player projections:
    - Pace multiplier
    - Team SA/GP adjustment
    - Series block rate adjustment
    - Shutdown matchup flag
    """
    global processed_shots, nhl_avg_tempo, nhl_avg_sa_per_game
    
    if processed_shots is None:
        raise HTTPException(status_code=500, detail="Shot data not loaded")
    
    try:
        # Get player mu/sigma
        player_stats = compute_weighted_mu_sigma(processed_shots)
        player_stats = player_stats[player_stats["shooterPlayerId"] == player_id]
        
        if player_stats.empty:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        
        # Create simple opponent context for base
        opponent_data = pd.DataFrame([{
            "teamCode": "OPP",
            "tempo": nhl_avg_tempo,
            "sa_per_game": nhl_avg_sa_per_game,
            "block_rate": 0.25
        }])
        
        # Apply opponent context (base layer)
        with_opponent = apply_opponent_context(
            player_stats,
            opponent_data,
            nhl_avg_tempo,
            nhl_avg_sa_per_game
        )
        
        # Create game state data
        game_state_data = pd.DataFrame([{
            "shooterPlayerId": player_id,
            "teamCode": player_stats["teamCode"].iloc[0],
            "period": period,
            "score_diff": score_diff,
            "trail_probability": 1.0 if score_diff < 0 else 0.4,
            "is_overtime": is_overtime,
            "is_top_shooter": True,  # For demonstration
            "is_star": True,  # For demonstration
            "is_top_six": True,  # For demonstration
        }])
        
        # Apply game state adjustments
        with_game_state = apply_game_state_adjustments(
            with_opponent,
            game_state_data
        )
        
        return with_game_state.to_dict(orient="records")[0]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying game state adjustments: {str(e)}")


@router.get("/player-quirks")
def get_player_quirks_adjustment(
    player_id: str = Query(..., description="Player ID"),
    expected_pp: float = Query(2.0, description="Expected power play opportunities"),
    actual_pp: float = Query(2.0, description="Actual power play opportunities"),
    last_game_ev_toi: Optional[float] = Query(None, description="Even strength TOI in last game")
):
    """
    Apply player-specific quirks to projections:
    - High-post shooters adjustment
    - Power-play share adjustment
    - TOI adjustment based on last game
    """
    global processed_shots
    
    if processed_shots is None:
        raise HTTPException(status_code=500, detail="Shot data not loaded")
    
    try:
        # Get player mu/sigma
        player_stats = compute_weighted_mu_sigma(processed_shots)
        player_stats = player_stats[player_stats["shooterPlayerId"] == player_id]
        
        if player_stats.empty:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        
        # Add base opponent and game state context (simplified)
        base_df = player_stats.copy()
        base_df["mu_opponent"] = base_df["mu"]
        base_df["sigma_opponent"] = base_df["sigma"]
        base_df["mu_game_state"] = base_df["mu"]
        base_df["sigma_game_state"] = base_df["sigma"]
        
        # Add player quirk data
        player_quirk_data = pd.DataFrame([{
            "shooterPlayerId": player_id,
            "is_high_post_shooter": player_id in ["8478483", "8478402"],  # Example IDs
            "expected_pp": expected_pp,
            "actual_pp": actual_pp,
            "last_game_ev_toi": last_game_ev_toi,
            "avg_ev_toi": 15.0  # Example average TOI
        }])
        
        # Apply player quirks
        player_df = base_df.merge(player_quirk_data, on="shooterPlayerId", how="left")
        with_quirks = apply_player_specific_quirks(player_df)
        
        return with_quirks.to_dict(orient="records")[0]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying player quirks: {str(e)}")


@router.get("/full-projection")
def get_full_projection(
    player_id: str = Query(..., description="Player ID"),
    opponent_id: str = Query(..., description="Opponent team ID"),
    period: int = Query(3, description="Current period"),
    score_diff: int = Query(0, description="Score difference (player team - opponent)"),
    is_overtime: bool = Query(False, description="Whether game is in overtime"),
    expected_pp: float = Query(2.0, description="Expected power play opportunities"),
    actual_pp: float = Query(2.0, description="Actual power play opportunities"),
    last_game_ev_toi: Optional[float] = Query(None, description="Even strength TOI in last game")
):
    """
    Get complete player projection with all adjustments:
    1. Base weighted mu/sigma
    2. Opponent context
    3. Game state adjustments
    4. Player-specific quirks
    """
    global processed_shots, nhl_avg_tempo, nhl_avg_sa_per_game
    
    if processed_shots is None:
        raise HTTPException(status_code=500, detail="Shot data not loaded")
    
    try:
        # Step 1: Get player mu/sigma
        player_stats = compute_weighted_mu_sigma(processed_shots)
        player_stats = player_stats[player_stats["shooterPlayerId"] == player_id]
        
        if player_stats.empty:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        
        # Step 2: Opponent context
        opponent_data = pd.DataFrame([{
            "teamCode": opponent_id,
            "tempo": nhl_avg_tempo * 1.05,  # Example: 5% faster than average
            "sa_per_game": nhl_avg_sa_per_game * 0.95,  # Example: 5% fewer shots against
            "block_rate": 0.28  # Example block rate
        }])
        
        with_opponent = apply_opponent_context(
            player_stats,
            opponent_data,
            nhl_avg_tempo,
            nhl_avg_sa_per_game
        )
        
        # Step 3: Game state adjustments
        game_state_data = pd.DataFrame([{
            "shooterPlayerId": player_id,
            "teamCode": player_stats["teamCode"].iloc[0],
            "period": period,
            "score_diff": score_diff,
            "trail_probability": 1.0 if score_diff < 0 else 0.4,
            "is_overtime": is_overtime,
            "is_top_shooter": True,  # For demonstration
            "is_star": True,  # For demonstration
            "is_top_six": True,  # For demonstration
        }])
        
        with_game_state = apply_game_state_adjustments(
            with_opponent,
            game_state_data
        )
        
        # Step 4: Player-specific quirks
        player_quirk_data = pd.DataFrame([{
            "shooterPlayerId": player_id,
            "is_high_post_shooter": player_id in ["8478483", "8478402"],  # Example IDs
            "expected_pp": expected_pp,
            "actual_pp": actual_pp,
            "last_game_ev_toi": last_game_ev_toi,
            "avg_ev_toi": 15.0  # Example average TOI
        }])
        
        player_df = with_game_state.merge(player_quirk_data, on="shooterPlayerId", how="left")
        with_quirks = apply_player_specific_quirks(player_df)
        
        # Step 5: Final projection
        final_projection = calculate_final_projection(with_quirks)
        
        # Add SOG line probabilities for common lines
        from scipy import stats
        
        final_mu = final_projection["mu_final"].iloc[0]
        final_sigma = final_projection["sigma_final"].iloc[0]
        
        sog_lines = [0.5, 1.5, 2.5, 3.5, 4.5]
        probabilities = {}
        
        for line in sog_lines:
            # P(SOG > line) = 1 - CDF(line + 0.5)
            prob = stats.norm.sf(line + 0.5, loc=final_mu, scale=final_sigma)
            probabilities[f"p_over_{line}"] = round(prob, 3)
        
        # Add probabilities to result
        result = final_projection.to_dict(orient="records")[0]
        result.update(probabilities)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating full projection: {str(e)}")


@router.get("/shot-data-integrity")
def check_shot_data_integrity():
    """
    Verify shot data integrity:
    - team-SOG + MISS + BLOCK + POST = total attempts after every game
    """
    global shot_data
    
    if shot_data is None:
        raise HTTPException(status_code=500, detail="Shot data not loaded")
    
    try:
        valid = validate_shot_data_integrity(shot_data)
        
        return {
            "integrity_check_passed": valid,
            "shot_count": len(shot_data),
            "unique_games": shot_data["game_id"].nunique()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking shot data integrity: {str(e)}")


@router.get("/player-season")
def get_player_season_stats(
    player_id: Optional[str] = None,
    season: Optional[str] = None
):
    """Compute total shots, goals, shooting percentage per player-season."""
    global processed_shots
    
    if processed_shots is None:
        raise HTTPException(status_code=500, detail="Shot data not loaded")
    
    try:
        df = processed_shots.copy()
        
        # Ensure season column exists
        if 'season' not in df.columns and 's' in df.columns:
            df = df.rename(columns={'s': 'season'})
        
        # Filter by player if provided
        if player_id:
            df = df[df["shooterPlayerId"] == player_id]
        
        # Filter by season if provided
        if season:
            df = df[df["season"].astype(str) == season]
        
        # Aggregate stats
        agg = (
            df.groupby(['season', 'shooterPlayerId'])
            .agg(
                total_shots=('is_sog', 'sum'),
                goals=('isGoal', 'sum'),
                games=('game_id', 'nunique')
            )
            .reset_index()
        )
        
        # Calculate derived stats
        agg["shooting_percentage"] = (agg["goals"] / agg["total_shots"]).fillna(0)
        agg["shots_per_game"] = (agg["total_shots"] / agg["games"]).fillna(0)
        
        if agg.empty:
            raise HTTPException(status_code=404, detail="No player-season data found for the specified filters")
        
        return agg.to_dict(orient="records")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating player-season stats: {str(e)}")


@router.get("/reload-data")
def reload_shot_data():
    """Reload shot data and recalculate NHL averages."""
    global shot_data, processed_shots, nhl_avg_tempo, nhl_avg_sa_per_game
    
    try:
        shot_data = load_shot_data("data")
        processed_shots = process_shots(shot_data)
        nhl_avg_tempo, nhl_avg_sa_per_game = get_nhl_averages(shot_data)
        
        return {
            "status": "success",
            "message": "Shot data reloaded successfully",
            "shot_count": len(shot_data),
            "avg_tempo": nhl_avg_tempo,
            "avg_sa_per_game": nhl_avg_sa_per_game
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading shot data: {str(e)}")
weighted_mu_sigma(processed_shots)
        player_stats = player_stats[player_stats["shooterPlayerId"] == player_id]
        
        if player_stats.empty:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        
        # Load opponent data (simplified for this example)
        # In a real scenario, this would come from a database or file
        opponent_data = pd.DataFrame([{
            "teamCode": opponent_id,
            "tempo": nhl_avg_tempo * 1.05,  # Example: 5% faster than average
            "sa_per_game": nhl_avg_sa_per_game * 0.95,  # Example: 5% fewer shots against
            "block_rate": 0.28  # Example block rate
        }])
        
        # Apply opponent context
        with_opponent = apply_opponent_context(
            player_stats,
            opponent_data,
            nhl_avg_tempo,
            nhl_avg_sa_per_game
        )
        
        return with_opponent.to_dict(orient="records")[0]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying opponent context: {str(e)}")


@router.get("/game-state")
def get_game_state_adjustment(
    player_id: str = Query(..., description="Player ID"),
    period: int = Query(3, description="Current period"),
    score_diff: int = Query(0, description="Score difference (player team - opponent)"),
    is_overtime: bool = Query(False, description="Whether game is in overtime")
):
    """
    Apply game state adjustments to player projections:
    - Lead-protect taper
    - Chase-mode bonus
    - Overtime/comeback adjustment
    """
    global processed_shots
    
    if processed_shots is None:
        raise HTTPException(status_code=500, detail="Shot data not loaded")
    
    try:
        # Get player mu/sigma
        player_stats = compute_