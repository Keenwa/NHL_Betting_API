import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Set, Any
import os
import glob

def compute_dynamic_weighted_mu_sigma(
    shots_df: pd.DataFrame,
    player_col: str = "shooterPlayerId",
    game_col: str = "game_id",
    season_col: str = "season",
    playoff_col: str = "is_playoff",
    sog_col: str = "is_sog",
    team_col: str = "teamCode",
    recency_factor: float = 0.9,  # Decay factor for older games
    playoff_boost: float = 1.5    # Importance multiplier for playoff games
) -> pd.DataFrame:
    """
    Compute dynamically weighted mean (μ) and std (σ) of SOG per game with 
    exponential decay for older games and importance boost for playoff games.
    
    Args:
        shots_df: DataFrame with shot data
        player_col: Column with player IDs
        game_col: Column with game IDs
        season_col: Column with seasons
        playoff_col: Column indicating playoff games
        sog_col: Column with SOG indicator
        team_col: Column with team codes
        recency_factor: Exponential decay factor for older games (0-1)
        playoff_boost: Importance multiplier for playoff games
        
    Returns:
        DataFrame with columns [player_col, team_col, 'mu', 'sigma']
    """
    # 1) per‐game SOG counts
    counts = (
        shots_df
        .groupby([player_col, season_col, team_col, game_col, playoff_col])[sog_col]
        .sum()
        .reset_index(name="sog_count")
    )
    
    # 2) Add game sequence - assuming game_id increases chronologically
    # If not, you would need to use actual game dates
    counts = counts.sort_values([player_col, game_col])
    counts["game_seq"] = counts.groupby(player_col).cumcount()
    
    results = []
    
    # 3) Process each player
    for player, player_data in counts.groupby(player_col):
        # Get team (use most recent)
        team = player_data[team_col].iloc[-1]
        
        # Order games by sequence (newest last)
        games = player_data.sort_values("game_seq")
        
        # Apply dynamic weighting
        total_games = len(games)
        
        if total_games > 0:
            # Calculate weights with recency decay and playoff boost
            weights = np.array([
                recency_factor ** (total_games - i - 1) * 
                (playoff_boost if row[playoff_col] else 1.0)
                for i, (_, row) in enumerate(games.iterrows())
            ])
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Calculate weighted statistics
            sog_values = games["sog_count"].values
            weighted_mean = np.sum(weights * sog_values)
            
            # Calculate weighted variance and std
            if total_games > 1:
                weighted_var = np.sum(weights * (sog_values - weighted_mean) ** 2)
                weighted_std = np.sqrt(weighted_var)
            else:
                weighted_std = 0.5  # Default std if only one game
                
            results.append({
                player_col: player,
                team_col: team,
                "mu": weighted_mean,
                "sigma": weighted_std,
                "games_used": total_games,
                "most_recent_weight": weights[-1] if len(weights) > 0 else 0
            })
    
    return pd.DataFrame(results)


def apply_opponent_context_with_h2h(
    mu_df: pd.DataFrame,
    opponent_df: pd.DataFrame,
    h2h_df: pd.DataFrame,  # Head-to-head stats
    league_avg_tempo: float,
    league_avg_sa_per_game: float,
    player_col: str = "shooterPlayerId",
    team_col: str = "teamCode",
    opponent_col: str = "opponent",
    h2h_weight: float = 0.3  # Weight for head-to-head data
) -> pd.DataFrame:
    """
    Apply opponent context adjustments to player mean and std,
    incorporating head-to-head history.
    
    Args:
        mu_df: DataFrame with player mu/sigma
        opponent_df: DataFrame with opponent stats
        h2h_df: DataFrame with head-to-head statistics
        league_avg_tempo: NHL average tempo
        league_avg_sa_per_game: NHL average shots against per game
        player_col: Column with player IDs
        team_col: Column with team codes
        opponent_col: Column with opponent codes
        h2h_weight: Weight for head-to-head data (0-1)
        
    Returns:
        DataFrame with adjusted mu/sigma
    """
    # Cross join players with opponents
    players = mu_df.copy()
    opp = opponent_df.copy()
    
    # Create a key for joining
    players["key"] = 1
    opp["key"] = 1
    
    # Cross join
    df = pd.merge(players, opp, on="key", suffixes=("", "_opp"))
    df = df.drop(columns=["key"])
    
    # Rename opponent columns for clarity
    df = df.rename(columns={
        f"{team_col}_opp": opponent_col,
        "tempo_opp": "tempo",
        "sa_per_game_opp": "sa_per_game",
        "block_rate_opp": "block_rate",
    })
    
    # Apply pace multiplier
    df["mu_pace"] = df["mu"] * (df["tempo"] / league_avg_tempo)
    
    # Apply SA/GP adjustment
    sa_diff = df["sa_per_game"] - league_avg_sa_per_game
    df["mu_sa"] = df["mu_pace"] + np.where(abs(sa_diff) >= 2, 0.15 * (sa_diff / 2), 0)
    
    # Apply block rate adjustment
    df["mu_block"] = df["mu_sa"] + np.where(
        df["block_rate"] > 0.30, -0.25,
        np.where(df["block_rate"] < 0.22, 0.15, 0)
    )
    
    df["sigma_block"] = df["sigma"] * np.where(df["block_rate"] > 0.30, 1.1, 1.0)
    
    # NEW: Incorporate head-to-head data
    if not h2h_df.empty:
        # Merge h2h data
        df = df.merge(
            h2h_df[[player_col, opponent_col, 'h2h_mu', 'h2h_games']],
            on=[player_col, opponent_col],
            how='left'
        )
        
        # Apply h2h adjustment where data exists
        h2h_mask = df['h2h_mu'].notna()
        
        if h2h_mask.any():
            # Calculate confidence factor based on number of h2h games
            df.loc[h2h_mask, 'h2h_confidence'] = np.minimum(
                df.loc[h2h_mask, 'h2h_games'] / 10, 1.0
            )
            
            # Apply weighted blend
            df.loc[h2h_mask, 'mu_h2h'] = (
                (1 - (h2h_weight * df.loc[h2h_mask, 'h2h_confidence'])) * df.loc[h2h_mask, 'mu_block'] +
                (h2h_weight * df.loc[h2h_mask, 'h2h_confidence']) * df.loc[h2h_mask, 'h2h_mu']
            )
        else:
            df['mu_h2h'] = df['mu_block']
    else:
        # No h2h data available
        df['mu_h2h'] = df['mu_block']
    
    # Apply shutdown matchup flag
    if "is_shadowed" in df.columns:
        df["mu_shadow"] = df["mu_h2h"] + np.where(df["is_shadowed"], -0.25, 0)
    else:
        df["mu_shadow"] = df["mu_h2h"]
    
    # Calculate final mu/sigma with all opponent context
    df["mu_opponent"] = df["mu_shadow"]
    df["sigma_opponent"] = df["sigma_block"]
    
    # Select relevant columns
    result_df = df[[
        player_col, team_col, opponent_col, 
        "mu", "sigma", "mu_opponent", "sigma_opponent"
    ]]
    
    return result_df


def apply_game_state_adjustments(
    proj_df: pd.DataFrame,
    game_state: pd.DataFrame,
    player_col: str = "shooterPlayerId",
    team_col: str = "teamCode",
) -> pd.DataFrame:
    """
    Apply game-state adjustments to player mean and std:
    • Lead-protect taper: if team leads by ≥2 after P2, top shooters −0.3 SOG
    • Chase-mode bonus: if team is 60%+ likely to trail (or does), stars +0.4 SOG
    • Overtime/comeback for Canes: add +0.3 SOG to top six in multi-OT or big deficit
    
    Args:
        proj_df: DataFrame with player projections including opponent adjustments
        game_state: DataFrame with game state information
        player_col: Column with player IDs
        team_col: Column with team codes
        
    Returns:
        DataFrame with game-state adjusted mu/sigma
    """
    df = proj_df.merge(game_state, on=[player_col, team_col], how="left")
    
    # Lead-protect taper
    # If team leads by ≥2 after P2, top shooters −0.3 SOG
    lead_protect = (
        (df["period"] > 2) & 
        (df["score_diff"] >= 2) & 
        (df["is_top_shooter"] == True)
    )
    df["mu_lead"] = df["mu_opponent"] + np.where(lead_protect, -0.3, 0)
    
    # Chase-mode bonus
    # If team is 60%+ likely to trail (or does), stars +0.4 SOG
    chase_mode = (
        ((df["trail_probability"] >= 0.6) | (df["score_diff"] < 0)) & 
        (df["is_star"] == True)
    )
    df["mu_chase"] = df["mu_lead"] + np.where(chase_mode, 0.4, 0)
    
    # Overtime/comeback for Canes
    # Add +0.3 SOG to top six in multi-OT or big deficit
    canes_bonus = (
        (df[team_col] == "CAR") & 
        ((df["is_overtime"] == True) | (df["score_diff"] <= -2)) & 
        (df["is_top_six"] == True)
    )
    df["mu_canes"] = df["mu_chase"] + np.where(canes_bonus, 0.3, 0)
    
    # Increase variance in OT
    df["sigma_ot"] = df["sigma_opponent"] * np.where(df["is_overtime"], 1.15, 1.0)
    
    # Calculate final game-state adjusted mu/sigma
    df["mu_game_state"] = df["mu_canes"]
    df["sigma_game_state"] = df["sigma_ot"]
    
    return df


def apply_advanced_game_state_adjustments(
    proj_df: pd.DataFrame,
    game_state: pd.DataFrame,
    situational_model: Optional[Any] = None,
    player_col: str = "shooterPlayerId",
    team_col: str = "teamCode",
) -> pd.DataFrame:
    """
    Apply advanced game-state adjustments to player mean and std using
    both rules-based and model-based approaches.
    
    Args:
        proj_df: DataFrame with player projections including opponent adjustments
        game_state: DataFrame with game state information
        situational_model: Optional trained situational model
        player_col: Column with player IDs
        team_col: Column with team codes
        
    Returns:
        DataFrame with game-state adjusted mu/sigma
    """
    # Start with basic game state adjustments
    df = apply_game_state_adjustments(proj_df, game_state)
    
    # Apply model-based adjustments if model is provided
    if situational_model is not None:
        for idx, row in df.iterrows():
            player_id = row[player_col]
            period = game_state.loc[idx, 'period']
            score_diff = game_state.loc[idx, 'score_diff']
            time_remaining = game_state.loc[idx, 'time_remaining'] if 'time_remaining' in game_state.columns else 600
            is_overtime = game_state.loc[idx, 'is_overtime']
            
            # Get model prediction
            adjustment = situational_model.predict_adjustment(
                player_id=player_id,
                period=period,
                score_diff=score_diff,
                time_remaining=time_remaining,
                is_overtime=is_overtime
            )
            
            # Apply adjustments
            df.loc[idx, 'mu_situation'] = df.loc[idx, 'mu_game_state'] + adjustment['mean_adj']
            df.loc[idx, 'sigma_situation'] = df.loc[idx, 'sigma_game_state'] * adjustment['sigma_factor']
    else:
        # No model, use existing values
        df['mu_situation'] = df['mu_game_state']
        df['sigma_situation'] = df['sigma_game_state']
    
    # Advanced coach-specific adjustments
    # Add team coach tendencies if available
    if 'coach_id' in game_state.columns:
        # This would use a lookup table of coach tendencies
        coach_tendencies = {
            # Example: coach ID -> adjustment factor for 3rd period trailing
            'c001': 0.2,  # Coach pulls goalie earlier, more aggressive
            'c002': -0.1,  # Coach is more defensive when trailing
        }
        
        for idx, row in df.iterrows():
            coach_id = game_state.loc[idx, 'coach_id']
            if coach_id in coach_tendencies and game_state.loc[idx, 'period'] >= 3:
                if game_state.loc[idx, 'score_diff'] < 0:  # Trailing
                    df.loc[idx, 'mu_situation'] += coach_tendencies[coach_id]
    
    # Set final game-state adjusted mu/sigma
    df['mu_game_state_final'] = df['mu_situation']
    df['sigma_game_state_final'] = df['sigma_situation']
    
    return df


def apply_player_specific_quirks(
    proj_df: pd.DataFrame,
    player_col: str = "shooterPlayerId",
) -> pd.DataFrame:
    """
    Apply player-specific quirks to mean and std:
    • High-post shooters (Strome, McDavid): mean −0.05 SOG, σ wider
    • Power-play share: ±0.15 SOG for every PP above/below 2 expected
    • Rolling last-game EV-TOI directly scales mean
    
    Args:
        proj_df: DataFrame with player projections including game-state adjustments
        player_col: Column with player IDs
        
    Returns:
        DataFrame with player-specific adjusted mu/sigma
    """
    df = proj_df.copy()
    
    # High-post shooters
    if "is_high_post_shooter" in df.columns:
        df["mu_post"] = df["mu_game_state"] + np.where(df["is_high_post_shooter"], -0.05, 0)
        df["sigma_post"] = df["sigma_game_state"] * np.where(df["is_high_post_shooter"], 1.1, 1.0)
    else:
        # Default if column doesn't exist
        df["mu_post"] = df["mu_game_state"]
        df["sigma_post"] = df["sigma_game_state"]
    
    # Power-play share
    if "actual_pp" in df.columns and "expected_pp" in df.columns:
        pp_diff = df["actual_pp"] - df["expected_pp"]
        df["mu_pp"] = df["mu_post"] + 0.15 * (pp_diff / 2)
    else:
        # Default if columns don't exist
        df["mu_pp"] = df["mu_post"]
    
    # Rolling last-game EV-TOI
    # Default to 1.0 (no change) if no TOI data
    df["toi_factor"] = 1.0
    
    # Where we have both last game TOI and average TOI
    if "last_game_ev_toi" in df.columns and "avg_ev_toi" in df.columns:
        mask = (df["last_game_ev_toi"].notna()) & (df["avg_ev_toi"] > 0)
        df.loc[mask, "toi_factor"] = df.loc[mask, "last_game_ev_toi"] / df.loc[mask, "avg_ev_toi"]
    
    # Apply TOI factor to mean
    df["mu_toi"] = df["mu_pp"] * df["toi_factor"]
    
    # Calculate final player-specific adjusted mu/sigma
    df["mu_player"] = df["mu_toi"]
    df["sigma_player"] = df["sigma_post"]
    
    return df


def calculate_final_projection(
    player_df: pd.DataFrame,
    player_col: str = "shooterPlayerId",
) -> pd.DataFrame:
    """
    Calculate final projections for all players.
    
    Args:
        player_df: DataFrame with all player adjustments
        player_col: Column with player IDs
        
    Returns:
        DataFrame with final projections
    """
    # Ensure all adjustment columns exist
    # If not, use the previous stage's values
    result_df = player_df.copy()
    
    if "mu_opponent" not in result_df.columns:
        result_df["mu_opponent"] = result_df["mu"]
        result_df["sigma_opponent"] = result_df["sigma"]
    
    if "mu_game_state" not in result_df.columns:
        result_df["mu_game_state"] = result_df["mu_opponent"]
        result_df["sigma_game_state"] = result_df["sigma_opponent"]
    
    if "mu_player" not in result_df.columns:
        result_df["mu_player"] = result_df["mu_game_state"]
        result_df["sigma_player"] = result_df["sigma_game_state"]
    
    # Set final values
    result_df["mu_final"] = result_df["mu_player"]
    result_df["sigma_final"] = result_df["sigma_player"]
    
    # Select relevant columns for output
    cols_to_keep = [
        player_col, "teamCode"
    ]
    
    # Add opponent if it exists
    if "opponent" in result_df.columns:
        cols_to_keep.append("opponent")
    
    # Add all mu/sigma columns
    mu_sigma_cols = [
        "mu", "sigma",                          # Base weighted stats
        "mu_opponent", "sigma_opponent",        # After opponent context
        "mu_game_state", "sigma_game_state",    # After game state
        "mu_player", "sigma_player",            # After player quirks
        "mu_final", "sigma_final"               # Final values
    ]
    
    # Add columns that exist
    for col in mu_sigma_cols:
        if col in result_df.columns:
            cols_to_keep.append(col)
    
    return result_df[cols_to_keep]


def calculate_final_projection_with_confidence(
    player_df: pd.DataFrame,
    lines: List[float] = [0.5, 1.5, 2.5, 3.5, 4.5],
    confidence_levels: List[float] = [0.80, 0.90, 0.95],
    player_col: str = "shooterPlayerId"
) -> pd.DataFrame:
    """
    Calculate final projections for all players with confidence intervals.
    
    Args:
        player_df: DataFrame with all player adjustments
        lines: List of SOG lines to calculate probabilities for
        confidence_levels: List of confidence levels for intervals
        player_col: Column with player IDs
        
    Returns:
        DataFrame with final projections and confidence intervals
    """
    # Start with standard final projection
    result_df = calculate_final_projection(player_df, player_col)
    
    # Add probabilities for each line
    for line in lines:
        line_str = str(line).replace('.', '_')
        result_df[f'p_over_{line_str}'] = result_df.apply(
            lambda row: stats.norm.sf(line + 0.5, loc=row['mu_final'], scale=row['sigma_final']),
            axis=1
        )
    
    # Calculate confidence intervals
    for i, row in result_df.iterrows():
        # Store confidence intervals for a representative line (e.g., 2.5)
        key_line = 2.5
        key_line_str = '2_5'
        
        intervals = calculate_confidence_intervals(
            mu=row['mu_final'],
            sigma=row['sigma_final'],
            line=key_line,
            confidence_levels=confidence_levels
        )
        
        # Add confidence intervals to result
        for level in confidence_levels:
            level_str = str(int(level * 100))
            result_df.at[i, f'mu_lower_{level_str}'] = intervals['intervals'][level]['mean_lower']
            result_df.at[i, f'mu_upper_{level_str}'] = intervals['intervals'][level]['mean_upper']
            result_df.at[i, f'p_over_{key_line_str}_lower_{level_str}'] = intervals['intervals'][level]['prob_lower']
            result_df.at[i, f'p_over_{key_line_str}_upper_{level_str}'] = intervals['intervals'][level]['prob_upper']
    
    return result_df


def load_shot_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load all shot CSVs from the shot_data folder.
    
    Args:
        data_dir: Directory containing shot data
        
    Returns:
        DataFrame with shot data
    """
    shot_dir = os.path.join(data_dir, "shot_data")
    if not os.path.exists(shot_dir):
        print(f"Shot data directory not found: {shot_dir}")
        # Create a minimal dummy DataFrame for testing
        return pd.DataFrame({
            'game_id': [1001, 1002, 1003],
            'teamCode': ['WSH', 'EDM', 'COL'],
            'shooterPlayerId': ['8471675', '8478402', '8471214'],
            'event': ['SHOT', 'GOAL', 'MISS'],
            'shotWasOnGoal': [1, 1, 0],
            'period': [1, 2, 3]
        })
    
    csv_paths = glob.glob(os.path.join(shot_dir, "**", "*.csv"), recursive=True)
    
    if not csv_paths:
        print(f"No shot-data CSVs found in {shot_dir}")
        # Try to extract from ZIP files if they exist
        zip_paths = glob.glob(os.path.join(shot_dir, "*.zip"))
        if zip_paths:
            print(f"Found {len(zip_paths)} ZIP files. Attempting to extract...")
            import zipfile
            for zip_path in zip_paths:
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(shot_dir)
                    print(f"Extracted {zip_path}")
                except Exception as e:
                    print(f"Error extracting {zip_path}: {e}")
            
            # Try to find CSVs again
            csv_paths = glob.glob(os.path.join(shot_dir, "**", "*.csv"), recursive=True)
    
    if not csv_paths:
        print("Still no CSV files found. Using dummy data for testing.")
        return pd.DataFrame({
            'game_id': [1001, 1002, 1003],
            'teamCode': ['WSH', 'EDM', 'COL'],
            'shooterPlayerId': ['8471675', '8478402', '8471214'],
            'event': ['SHOT', 'GOAL', 'MISS'],
            'shotWasOnGoal': [1, 1, 0],
            'period': [1, 2, 3]
        })
    
    df_list = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    if not df_list:
        print("Failed to load any CSV files. Using dummy data for testing.")
        return pd.DataFrame({
            'game_id': [1001, 1002, 1003],
            'teamCode': ['WSH', 'EDM', 'COL'],
            'shooterPlayerId': ['8471675', '8478402', '8471214'],
            'event': ['SHOT', 'GOAL', 'MISS'],
            'shotWasOnGoal': [1, 1, 0],
            'period': [1, 2, 3]
        })
    
    return pd.concat(df_list, ignore_index=True)


def process_shots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process shot data to identify SOG, MISS, BLOCK, and POST.
    Also adds zone information for more detailed analysis.
    
    Args:
        df: DataFrame with shot data
        
    Returns:
        Processed DataFrame with shot classification and zones
    """
    # Create a new dataframe instead of modifying in place
    # This avoids fragmentation warnings
    result_columns = {}
    
    # Copy original columns
    for col in df.columns:
        result_columns[col] = df[col].copy()
    
    # Add shot classification columns all at once
    if 'event' in df.columns:
        shot_mask = df['event'] == 'SHOT'
        goal_mask = df['event'] == 'GOAL'
        miss_mask = df['event'] == 'MISS'
        block_mask = df['event'] == 'BLOCK'
        
        # SOG flag
        if 'shotWasOnGoal' in df.columns:
            result_columns['is_sog'] = ((shot_mask & (df['shotWasOnGoal'] == 1)) | goal_mask).astype(bool)
        else:
            result_columns['is_sog'] = (shot_mask | goal_mask).astype(bool)
        
        # MISS flag
        if 'shotWasOnGoal' in df.columns:
            result_columns['is_miss'] = (miss_mask | (shot_mask & (df['shotWasOnGoal'] == 0))).astype(bool)
        else:
            result_columns['is_miss'] = miss_mask.astype(bool)
        
        # BLOCK flag
        result_columns['is_block'] = block_mask.astype(bool)
    else:
        # Default values if event column doesn't exist
        result_columns['is_sog'] = False
        result_columns['is_miss'] = False
        result_columns['is_block'] = False
    
    # POST flag
    if 'hitPost' in df.columns:
        result_columns['is_post'] = (df['hitPost'] == 1).astype(bool)
    else:
        result_columns['is_post'] = False
    
    # Add zone classifications if coordinates are available
    if 'xCord' in df.columns and 'yCord' in df.columns:
        result_columns['is_slot_shot'] = (
            (abs(df['xCord']) < 30) & 
            (df['yCord'] > 0) & 
            (df['yCord'] < 30)
        ).astype(bool)
        
        result_columns['is_point_shot'] = (df['yCord'] > 45).astype(bool)
        
        result_columns['is_home_plate'] = (
            (abs(df['xCord']) < 40) & 
            (df['yCord'] > -10) & 
            (df['yCord'] < 40)
        ).astype(bool)
        
        # Derive is_perimeter_shot from the others
        result_columns['is_perimeter_shot'] = ~(
            result_columns['is_slot_shot'] | result_columns['is_point_shot']
        )
    else:
        # Default values if coordinates not available
        result_columns['is_slot_shot'] = False
        result_columns['is_point_shot'] = False
        result_columns['is_perimeter_shot'] = False
        result_columns['is_home_plate'] = False
    
    # Ensure we have shotID for integrity checks
    if 'shotID' not in df.columns and 'id' in df.columns:
        result_columns['shotID'] = df['id'].copy()
    elif 'shotID' not in df.columns:
        result_columns['shotID'] = np.arange(len(df))
    
    # Create a new DataFrame with all columns
    return pd.DataFrame(result_columns)


def validate_shot_data_integrity(shot_data: pd.DataFrame) -> bool:
    """
    Verify that team-SOG + MISS + BLOCK + POST = total attempts after every game.
    
    Args:
        shot_data: DataFrame with shot data
        
    Returns:
        True if integrity check passes, False otherwise
    """
    # Make a copy to avoid modifying the original
    df = shot_data.copy()
    
    # Check for required ID columns
    id_cols = []
    if 'game_id' in df.columns:
        id_cols.append('game_id')
    else:
        print("Warning: Missing 'game_id' column")
        return False
    
    if 'teamCode' in df.columns:
        id_cols.append('teamCode')
    elif 'teamId' in df.columns:
        id_cols.append('teamId')
    else:
        print("Warning: Missing team identifier column")
        return False
    
    # Check required shot type columns and create them if possible
    required_shot_cols = ['is_sog', 'is_miss', 'is_block', 'is_post']
    missing_shot_cols = [col for col in required_shot_cols if col not in df.columns]
    
    if missing_shot_cols:
        print(f"Warning: Missing shot type columns: {missing_shot_cols}")
        
        # Try to create missing columns if we have the base data
        if 'event' in df.columns:
            if 'is_sog' not in df.columns:
                if 'shotWasOnGoal' in df.columns:
                    df['is_sog'] = ((df['event'] == 'SHOT') & (df['shotWasOnGoal'] == 1)) | (df['event'] == 'GOAL')
                else:
                    df['is_sog'] = (df['event'] == 'SHOT') | (df['event'] == 'GOAL')
            
            if 'is_miss' not in df.columns:
                if 'shotWasOnGoal' in df.columns:
                    df['is_miss'] = (df['event'] == 'MISS') | ((df['event'] == 'SHOT') & (df['shotWasOnGoal'] == 0))
                else:
                    df['is_miss'] = (df['event'] == 'MISS')
            
            if 'is_block' not in df.columns:
                df['is_block'] = (df['event'] == 'BLOCK')
        
        if 'is_post' not in df.columns:
            if 'hitPost' in df.columns:
                df['is_post'] = df['hitPost'] == 1
            else:
                df['is_post'] = False
        # ------------------------------------------------------------------
    #  NEW: playoff flag (needed by downstream code)
    # ------------------------------------------------------------------
    if 'is_playoff' not in df.columns:
        # MoneyPuck has a boolean/int column called 'playoffGame'
        # (1 for playoff games, 0 for regular‑season).  Fallback to 0.
        df['is_playoff'] = df.get('playoffGame', 0).fillna(0).astype(bool)
        
    # Check if required columns now exist
    missing_shot_cols = [col for col in required_shot_cols if col not in df.columns]
    if missing_shot_cols:
        print(f"Error: Unable to create required columns: {missing_shot_cols}")
        return False
    
    # Check for count column
    count_col = None
    if 'shotID' in df.columns:
        count_col = 'shotID'
    elif 'id' in df.columns:
        count_col = 'id'
    
    if count_col is None:
        print("Creating synthetic shot ID column for counting")
        df['shotID'] = range(len(df))
        count_col = 'shotID'
    
    # Perform integrity check
    try:
        # Count SOG + MISS + BLOCK + POST
        agg_dict = {col: 'sum' for col in required_shot_cols if col in df.columns}
        
        # Add count of total rows
        game_totals = df.groupby(id_cols).agg(agg_dict)
        game_totals['total_attempts'] = df.groupby(id_cols)[count_col].count()
        
        # Sum up the shot type columns
        available_cols = [col for col in required_shot_cols if col in game_totals.columns]
        if not available_cols:
            print("Error: No shot type columns available")
            return False
        
        game_totals['calculated_attempts'] = game_totals[available_cols].sum(axis=1)
        
        # Check for mismatches
        mismatches = game_totals[game_totals['total_attempts'] != game_totals['calculated_attempts']]
        
        if not mismatches.empty:
            print(f"Integrity check failed for {len(mismatches)} games")
            print(mismatches.head())
            return False
        
        print("Integrity check passed for all games")
        return True
    
    except Exception as e:
        print(f"Error during integrity check: {e}")
        return False


def get_nhl_averages(shot_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate NHL average tempo and shots against per game.
    
    Args:
        shot_data: DataFrame with shot data
        
    Returns:
        Tuple of (avg_tempo, avg_sa_per_game)
    """
    try:
        # Process shots
        df = process_shots(shot_data)
        
        # Calculate team shots per game
        team_shots = df.groupby(["teamCode", "game_id"]).size().reset_index(name="attempts")
        avg_shots_per_game = team_shots["attempts"].mean()
        
        # Calculate tempo (total events per game)
        game_events = df.groupby("game_id").size().reset_index(name="events")
        avg_tempo = game_events["events"].mean() / 2  # Divide by 2 for per-team
        
        return avg_tempo, avg_shots_per_game
    except Exception as e:
        print(f"Error calculating league averages: {e}")
        # Return default values
        return 60.0, 30.0


def build_player_matchup_history(
    shots_df: pd.DataFrame,
    player_col: str = "shooterPlayerId",
    team_col: str = "teamCode",
    game_col: str = "game_id",
    sog_col: str = "is_sog",
    min_games: int = 3  # Minimum games for reliable h2h stats
) -> pd.DataFrame:
    """
    Build head-to-head player vs opponent team history.
    
    Args:
        shots_df: DataFrame with shot data
        player_col: Column with player IDs
        team_col: Column with team codes
        game_col: Column with game IDs
        sog_col: Column with SOG indicator
        min_games: Minimum games for reliable h2h stats
        
    Returns:
        DataFrame with head-to-head statistics
    """
    if player_col not in shots_df.columns or team_col not in shots_df.columns:
        print(f"Warning: Required columns missing for h2h history: {player_col}, {team_col}")
        return pd.DataFrame()  # Return empty DataFrame
        
    try:
        # Create a copy of the dataframe to avoid modifying the original
        shots = shots_df.copy()
        
        # Ensure player_id column is string type
        shots[player_col] = shots[player_col].astype(str)
        
        # Get opponent for each game
        game_teams = shots.groupby(game_col)[team_col].unique()
        
        # Expand to get all team combinations per game
        game_opponents = {}
        for game_id, teams in game_teams.items():
            if len(teams) == 2:  # Ensure there are exactly 2 teams
                game_opponents[game_id] = {
                    teams[0]: teams[1],
                    teams[1]: teams[0]
                }
        
        # Add opponent column to shots_df
        opponent_list = []
        for _, row in shots.iterrows():
            game_id = row[game_col]
            team = row[team_col]
            opponent = game_opponents.get(game_id, {}).get(team, None)
            opponent_list.append(opponent)
        
        shots['opponent'] = opponent_list
        
        # Filter out rows with missing opponents
        shots = shots.dropna(subset=['opponent'])
        
        if shots.empty:
            print("Warning: No valid opponent data available for h2h history")
            return pd.DataFrame()
        
        # Group by player and opponent to get head-to-head stats
        h2h_stats = (
            shots
            .groupby([player_col, 'opponent'])
            .agg(
                h2h_games=pd.NamedAgg(column=game_col, aggfunc='nunique'),
                h2h_sog_total=pd.NamedAgg(column=sog_col, aggfunc='sum'),
                h2h_shots_total=pd.NamedAgg(column=player_col, aggfunc='size')
            )
            .reset_index()
        )
        
        # Calculate per-game averages
        h2h_stats['h2h_mu'] = h2h_stats['h2h_sog_total'] / h2h_stats['h2h_games']
        h2h_stats['h2h_frequency'] = h2h_stats['h2h_shots_total'] / h2h_stats['h2h_games']
        
        # Filter by minimum games for reliability
        return h2h_stats[h2h_stats['h2h_games'] >= min_games]
    except Exception as e:
        print(f"Error building head-to-head history: {e}")
        return pd.DataFrame()


def calculate_confidence_intervals(
    mu: float,
    sigma: float,
    line: float,
    confidence_levels: List[float] = [0.80, 0.90, 0.95]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate confidence intervals for SOG predictions.
    
    Args:
        mu: Mean SOG projection
        sigma: Standard deviation
        line: SOG line
        confidence_levels: List of confidence levels (0-1)
        
    Returns:
        Dictionary with confidence intervals
    """
    from scipy import stats
    
    results = {
        'probability': stats.norm.sf(line + 0.5, loc=mu, scale=sigma),
        'intervals': {}
    }
    
    for level in confidence_levels:
        # Calculate confidence interval for mean
        z_value = stats.norm.ppf((1 + level) / 2)
        lower = mu - z_value * sigma
        upper = mu + z_value * sigma
        
        # Calculate equivalent interval for probability
        prob_lower = stats.norm.sf(line + 0.5, loc=lower, scale=sigma)
        prob_upper = stats.norm.sf(line + 0.5, loc=upper, scale=sigma)
        
        # Ensure correct order (probability decreases as mean increases)
        prob_lower, prob_upper = max(prob_lower, prob_upper), min(prob_lower, prob_upper)
        
        results['intervals'][level] = {
            'mean_lower': max(0, lower),  # Can't have negative SOG
            'mean_upper': upper,
            'prob_lower': prob_lower,
            'prob_upper': prob_upper
        }
    
    return results
