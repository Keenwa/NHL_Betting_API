import os
import sys
import pandas as pd
import numpy as np

# Make sure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def process_shots_moneypuck(df):
    """Process MoneyPuck shot data format"""
    df = df.copy()
    
    # Create SOG flag (shotWasOnGoal is already in MoneyPuck data)
    df["is_sog"] = ((df["event"] == "SHOT") & (df["shotWasOnGoal"] == 1)) | (df["event"] == "GOAL")
    
    # Create MISS flag - exclude BLOCK and POST
    df["is_miss"] = (df["event"] == "MISS") | ((df["event"] == "SHOT") & (df["shotWasOnGoal"] == 0))
    
    # Create BLOCK flag
    df["is_block"] = (df["event"] == "BLOCK")
    
    # Create POST flag if hitPost exists
    if "hitPost" in df.columns:
        df["is_post"] = df["hitPost"] == 1
    else:
        df["is_post"] = False
    
    # Ensure required columns
    for col in ["shooterPlayerId", "game_id", "teamId"]:
        if col not in df.columns:
            print(f"Warning: Expected column '{col}' not found in shot data")
    
    # Rename teamId to teamCode if necessary
    if "teamId" in df.columns and "teamCode" not in df.columns:
        df["teamCode"] = df["teamId"]
    
    return df

def validate_shot_data_integrity_moneypuck(shot_data):
    """Validate shot data integrity using MoneyPuck format"""
    # Process shots
    df = process_shots_moneypuck(shot_data)
    
    # Group by game and team
    id_cols = ["game_id", "teamCode"] if "teamCode" in df.columns else ["game_id", "teamId"]
    
    # Count SOG + MISS + BLOCK + POST
    game_totals = df.groupby(id_cols).agg({
        "is_sog": "sum",     # SOG
        "is_miss": "sum",    # MISS
        "is_block": "sum",   # BLOCK
        "is_post": "sum"     # POST
    })
    
    # Calculate total attempts
    game_totals["total_attempts"] = df.groupby(id_cols).size()
    game_totals["calculated_attempts"] = (
        game_totals["is_sog"] +
        game_totals["is_miss"] +
        game_totals["is_block"] +
        game_totals["is_post"]
    )
    
    # Check for mismatches
    mismatches = game_totals[game_totals["total_attempts"] != game_totals["calculated_attempts"]]
    
    if not mismatches.empty:
        print(f"Integrity check failed for {len(mismatches)} games")
        print(mismatches.head())
        return False
    
    return True

def compute_weighted_mu_sigma_moneypuck(
    shots_df,
    player_col="shooterPlayerId",
    game_col="game_id",
    season_col="season",
    playoff_col="isPlayoffGame",
    sog_col="is_sog",
    team_col="teamCode",
):
    """Compute weighted mu/sigma using MoneyPuck format"""
    # Ensure playoff column exists
    if playoff_col not in shots_df.columns:
        # If isPlayoffGame doesn't exist, create it based on season format
        if season_col in shots_df.columns:
            playoff_indicator = shots_df[season_col].astype(str).str.contains("P")
            shots_df[playoff_col] = playoff_indicator
        else:
            # Default to regular season
            shots_df[playoff_col] = False
    
    # Calculate SOG per game for each player
    game_sog = (
        shots_df.groupby([player_col, team_col, game_col])[sog_col]
        .sum()
        .reset_index(name="sog_count")
    )
    
    # Add playoff indicator
    if playoff_col in shots_df.columns:
        playoff_info = (
            shots_df.groupby(game_col)[playoff_col]
            .first()
            .reset_index()
        )
        game_sog = game_sog.merge(playoff_info, on=game_col, how="left")
    
    # Sort games by recency (we'll assume game_id increases with time)
    game_sog = game_sog.sort_values([player_col, game_col], ascending=[True, False])
    
    # Prepare results
    results = []
    
    # Process each player
    for player, player_data in game_sog.groupby(player_col):
        # Get team (use most recent)
        team = player_data[team_col].iloc[0]
        
        # Get player's playoff and regular season games
        if playoff_col in player_data.columns:
            player_playoff = player_data[player_data[playoff_col] == True]
            player_rs = player_data[player_data[playoff_col] == False]
        else:
            # If no playoff indicator, assume all regular season
            player_playoff = pd.DataFrame()
            player_rs = player_data
        
        # Calculate weighted stats
        weights = []
        sog_values = []
        
        # Last playoff game: 50%
        if not player_playoff.empty:
            last_playoff_sog = player_playoff.iloc[0]["sog_count"]
            weights.append(0.5)
            sog_values.append(last_playoff_sog)
            
            # G-2 playoff game: 25%
            if len(player_playoff) > 1:
                g2_playoff_sog = player_playoff.iloc[1]["sog_count"]
                weights.append(0.25)
                sog_values.append(g2_playoff_sog)
                
            # Older playoff games: 15%
            if len(player_playoff) > 2:
                older_playoff_sog = player_playoff.iloc[2:]["sog_count"].mean()
                weights.append(0.15)
                sog_values.append(older_playoff_sog)
        
        # Regular season: 10%
        if not player_rs.empty:
            rs_sog = player_rs["sog_count"].mean()
            weights.append(0.10)
            sog_values.append(rs_sog)
            
        # If no data, skip
        if not weights:
            continue
            
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Calculate weighted mean
        weighted_mean = np.sum(np.array(weights) * np.array(sog_values))
        
        # Calculate weighted variance and std
        if len(sog_values) > 1:
            weighted_var = np.sum(weights * (np.array(sog_values) - weighted_mean) ** 2)
            weighted_std = np.sqrt(weighted_var)
        else:
            # Default std if only one game
            weighted_std = 0.5
            
        results.append({
            player_col: player,
            team_col: team,
            "mu": weighted_mean,
            "sigma": weighted_std
        })
    
    return pd.DataFrame(results)

def test_shot_features_moneypuck():
    """Test shot features pipeline with MoneyPuck data format"""
    print("=== TESTING SHOT FEATURES PIPELINE WITH MONEYPUCK FORMAT ===\n")
    
    # 1) Create sample shot data (MoneyPuck format)
    shots_data = {
        "game_id": [1001, 1001, 1001, 1002, 1002, 1003, 1003, 1003],
        "season": ["20232024", "20232024", "20232024", "20232024P", "20232024P", "20232024P", "20232024P", "20232024P"],
        "isPlayoffGame": [False, False, False, True, True, True, True, True],
        "shooterPlayerId": ["8470600", "8470600", "8471675", "8470600", "8471675", "8470600", "8471675", "8478402"],
        "teamId": ["NYR", "NYR", "WSH", "NYR", "WSH", "NYR", "WSH", "EDM"],
        "event": ["SHOT", "SHOT", "GOAL", "MISS", "SHOT", "SHOT", "GOAL", "SHOT"],
        "shotWasOnGoal": [1, 1, 1, 0, 1, 1, 1, 1],
        "goal": [0, 0, 1, 0, 0, 0, 1, 0],
        "hitPost": [0, 0, 0, 0, 0, 0, 0, 0],
        "dateTime": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-05-01", "2024-05-01", "2024-05-10", "2024-05-10", "2024-05-10"]
    }
    
    shots_df = pd.DataFrame(shots_data)
    
    # 2) Process shots data to add required flags
    processed_shots = process_shots_moneypuck(shots_df)
    print("Processed shots sample:")
    print(processed_shots.head(3))
    print()
    
    # 3) Validate data integrity
    valid = validate_shot_data_integrity_moneypuck(processed_shots)
    print(f"Data integrity check: {'PASSED' if valid else 'FAILED'}")
    print()
    
    # 4) Compute weighted mu/sigma
    mu_sigma = compute_weighted_mu_sigma_moneypuck(
        processed_shots,
        player_col="shooterPlayerId",
        game_col="game_id",
        season_col="season",
        playoff_col="isPlayoffGame",
        sog_col="is_sog",
        team_col="teamCode"
    )
    print("Weighted µ/σ sample:")
    print(mu_sigma)
    print()
    
    # 5) Apply simple opponent context (for testing)
    opponent_data = pd.DataFrame([
        {"teamCode": "BOS", "tempo": 63.0, "sa_per_game": 28.5, "block_rate": 0.32},
        {"teamCode": "FLA", "tempo": 66.0, "sa_per_game": 31.5, "block_rate": 0.20},
        {"teamCode": "TBL", "tempo": 61.2, "sa_per_game": 30.0, "block_rate": 0.25}
    ])
    
    # Testing cross join manually
    results = []
    for _, player_row in mu_sigma.iterrows():
        for _, opp_row in opponent_data.iterrows():
            # Pace multiplier
            mu_opponent = player_row["mu"] * (opp_row["tempo"] / 60.0)  # Assume 60.0 is NHL avg
            
            results.append({
                "shooterPlayerId": player_row["shooterPlayerId"],
                "teamCode": player_row["teamCode"],
                "opponent": opp_row["teamCode"],
                "mu": player_row["mu"],
                "sigma": player_row["sigma"],
                "mu_opponent": mu_opponent,
                "sigma_opponent": player_row["sigma"]
            })
    
    opponent_adjusted = pd.DataFrame(results)
    print("After simple opponent context adjustments:")
    print(opponent_adjusted.head())
    print()
    
    # 6) Calculate SOG line probabilities
    from scipy import stats
    
    for _, row in opponent_adjusted.iterrows():
        player_id = row["shooterPlayerId"]
        mu = row["mu_opponent"]
        sigma = row["sigma_opponent"]
        
        print(f"Player {player_id} vs {row['opponent']} SOG Probabilities:")
        for line in [0.5, 1.5, 2.5, 3.5]:
            prob = stats.norm.sf(line + 0.5, loc=mu, scale=sigma)
            print(f"  P(SOG > {line}): {prob:.3f}")
        break  # Just show one example
    
    return mu_sigma, opponent_adjusted

if __name__ == "__main__":
    mu_sigma, opponent_adjusted = test_shot_features_moneypuck()