import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations

# Make sure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def project_sog_probs(mu_sigma_df, lines_df):
    """Project SOG probabilities using Normal distribution"""
    from scipy import stats
    
    # Ensure we have player IDs in both dataframes
    if "shooterPlayerId" not in mu_sigma_df.columns:
        raise ValueError("mu_sigma_df must contain shooterPlayerId column")
    
    if "shooterPlayerId" not in lines_df.columns:
        if "player_id" in lines_df.columns:
            lines_df = lines_df.rename(columns={"player_id": "shooterPlayerId"})
        else:
            raise ValueError("lines_df must contain shooterPlayerId or player_id column")
    
    # Ensure we have sog_line in lines_df
    if "sog_line" not in lines_df.columns:
        if "line" in lines_df.columns:
            lines_df = lines_df.rename(columns={"line": "sog_line"})
        else:
            raise ValueError("lines_df must contain sog_line or line column")
    
    # Merge datasets
    merged = lines_df.merge(mu_sigma_df, on="shooterPlayerId", how="left")
    if merged.empty:
        print("Warning: No matches found between lines and player statistics")
        return pd.DataFrame()
    
    # Calculate probabilities
    merged["p_over"] = merged.apply(
        lambda row: stats.norm.sf(row["sog_line"] + 0.5, loc=row["mu"], scale=row["sigma"]),
        axis=1
    )
    
    return merged

def compute_edge(proj_df, odds_df):
    """Compute edge between model probability and implied odds"""
    # Ensure we have player IDs in both dataframes
    if "shooterPlayerId" not in proj_df.columns:
        raise ValueError("proj_df must contain shooterPlayerId column")
    
    if "shooterPlayerId" not in odds_df.columns:
        if "player_id" in odds_df.columns:
            odds_df = odds_df.rename(columns={"player_id": "shooterPlayerId"})
        else:
            raise ValueError("odds_df must contain shooterPlayerId or player_id column")
    
    # Ensure we have american_odds in odds_df
    if "american_odds" not in odds_df.columns:
        raise ValueError("odds_df must contain american_odds column")
    
    # Merge datasets
    merged = proj_df.merge(odds_df, on="shooterPlayerId", how="left")
    if merged.empty:
        print("Warning: No matches found between projections and odds")
        return pd.DataFrame()
    
    # Convert American odds to implied probability
    def american_to_implied(american_odds):
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    merged["implied_probability"] = merged["american_odds"].apply(american_to_implied)
    
    # Calculate edge
    merged["edge"] = merged["p_over"] - merged["implied_probability"]
    
    return merged

def build_tickets(edges_df, min_edge=0.05, max_legs=3):
    """Build tickets (parlays) from edges"""
    # Filter by minimum edge
    filtered = edges_df[edges_df["edge"] >= min_edge].copy()
    if filtered.empty:
        print("Warning: No edges above threshold")
        return pd.DataFrame()
    
    # Ensure we have game_id
    if "game_id" not in filtered.columns:
        if "game" in filtered.columns:
            filtered["game_id"] = filtered["game"]
        else:
            print("Warning: No game_id column found, creating dummy game IDs")
            filtered["game_id"] = [f"GAME-{i}" for i in range(len(filtered))]
    
    # Convert American odds to decimal
    filtered["decimal_odds"] = np.where(
        filtered["american_odds"] > 0,
        filtered["american_odds"] / 100 + 1,
        100 / abs(filtered["american_odds"]) + 1
    )
    
    tickets = []
    
    # Single legs
    for _, leg in filtered.iterrows():
        tickets.append({
            "legs": 1,
            "players": [leg["shooterPlayerId"]],
            "games": [leg["game_id"]],
            "combined_odds": leg["decimal_odds"],
            "combined_p": leg["p_over"],
            "EV": leg["decimal_odds"] * leg["p_over"] - 1,
        })
    
    # Multi-leg combos (2 to max_legs)
    for r in range(2, max_legs + 1):
        for combo_idx in combinations(filtered.index, r):
            legs = filtered.loc[list(combo_idx)]
            
            # One leg per game
            if legs["game_id"].nunique() < len(legs):
                continue
            
            dec_odds = legs["decimal_odds"].prod()
            p_win = legs["p_over"].prod()
            ev = dec_odds * p_win - 1
            
            if ev <= 0:
                continue
            
            tickets.append({
                "legs": r,
                "players": legs["shooterPlayerId"].tolist(),
                "games": legs["game_id"].tolist(),
                "combined_odds": dec_odds,
                "combined_p": p_win,
                "EV": ev,
            })
    
    return pd.DataFrame(tickets).sort_values("EV", ascending=False)

def test_full_pipeline():
    """Test the full projection pipeline"""
    print("=== TESTING FULL PROJECTION PIPELINE ===\n")
    
    # 1) Create sample player stats
    mu_sigma = pd.DataFrame([
        {"shooterPlayerId": "8470600", "teamCode": "NYR", "mu": 2.1, "sigma": 0.8},  # Kreider
        {"shooterPlayerId": "8471675", "teamCode": "WSH", "mu": 3.4, "sigma": 1.1},  # Ovechkin
        {"shooterPlayerId": "8478402", "teamCode": "EDM", "mu": 2.8, "sigma": 0.9},  # McDavid
    ])
    
    # 2) Create sample betting lines
    lines = pd.DataFrame([
        {"shooterPlayerId": "8470600", "sog_line": 1.5, "game_id": "NYR-BOS"},
        {"shooterPlayerId": "8471675", "sog_line": 2.5, "game_id": "WSH-TBL"},
        {"shooterPlayerId": "8478402", "sog_line": 2.5, "game_id": "EDM-CGY"},
    ])
    
    # 3) Create sample odds
    odds = pd.DataFrame([
        {"shooterPlayerId": "8470600", "american_odds": 110, "game_id": "NYR-BOS"},
        {"shooterPlayerId": "8471675", "american_odds": -120, "game_id": "WSH-TBL"},
        {"shooterPlayerId": "8478402", "american_odds": -110, "game_id": "EDM-CGY"},
    ])
    
    # 4) Project probabilities
    print("Step 1: Projecting probabilities")
    proj = project_sog_probs(mu_sigma, lines)
    print(proj)
    print()
    
    # 5) Compute edge
    print("Step 2: Computing edge")
    edged = compute_edge(proj, odds)
    print(edged)
    print()
    
    # 6) Build tickets
    print("Step 3: Building tickets")
    tickets = build_tickets(edged)
    print(tickets)
    print()
    
    return proj, edged, tickets

def test_with_sample_data():
    """Test with more realistic sample data"""
    print("=== TESTING WITH REALISTIC SAMPLE DATA ===\n")
    
    # 1) Create more comprehensive player stats
    mu_sigma = pd.DataFrame([
        {"shooterPlayerId": "8470600", "teamCode": "NYR", "mu": 2.1, "sigma": 0.8},  # Kreider
        {"shooterPlayerId": "8471675", "teamCode": "WSH", "mu": 3.4, "sigma": 1.1},  # Ovechkin
        {"shooterPlayerId": "8478402", "teamCode": "EDM", "mu": 2.8, "sigma": 0.9},  # McDavid
        {"shooterPlayerId": "8471214", "teamCode": "COL", "mu": 3.1, "sigma": 0.9},  # MacKinnon
        {"shooterPlayerId": "8480012", "teamCode": "VGK", "mu": 2.4, "sigma": 0.7},  # Tuch
        {"shooterPlayerId": "8476453", "teamCode": "TOR", "mu": 2.7, "sigma": 0.8},  # Matthews
    ])
    
    # 2) Create betting lines for multiple games
    lines = pd.DataFrame([
        {"shooterPlayerId": "8470600", "sog_line": 1.5, "game_id": "NYR-BOS", "team": "NYR", "opponent": "BOS"},
        {"shooterPlayerId": "8471675", "sog_line": 2.5, "game_id": "WSH-TBL", "team": "WSH", "opponent": "TBL"},
        {"shooterPlayerId": "8478402", "sog_line": 2.5, "game_id": "EDM-CGY", "team": "EDM", "opponent": "CGY"},
        {"shooterPlayerId": "8471214", "sog_line": 3.5, "game_id": "COL-MIN", "team": "COL", "opponent": "MIN"},
        {"shooterPlayerId": "8480012", "sog_line": 2.5, "game_id": "VGK-SEA", "team": "VGK", "opponent": "SEA"},
        {"shooterPlayerId": "8476453", "sog_line": 2.5, "game_id": "TOR-MTL", "team": "TOR", "opponent": "MTL"},
    ])
    
    # 3) Create odds with various American odds values
    odds = pd.DataFrame([
        {"shooterPlayerId": "8470600", "american_odds": 110, "game_id": "NYR-BOS"},
        {"shooterPlayerId": "8471675", "american_odds": -120, "game_id": "WSH-TBL"},
        {"shooterPlayerId": "8478402", "american_odds": -110, "game_id": "EDM-CGY"},
        {"shooterPlayerId": "8471214", "american_odds": -130, "game_id": "COL-MIN"},
        {"shooterPlayerId": "8480012", "american_odds": 100, "game_id": "VGK-SEA"},
        {"shooterPlayerId": "8476453", "american_odds": -115, "game_id": "TOR-MTL"},
    ])
    
    # 4) Project probabilities
    print("Step 1: Projecting probabilities")
    proj = project_sog_probs(mu_sigma, lines)
    print(proj)
    print()
    
    # 5) Compute edge
    print("Step 2: Computing edge")
    edged = compute_edge(proj, odds)
    print(edged)
    print()
    
    # 6) Build tickets
    print("Step 3: Building tickets")
    tickets = build_tickets(edged)
    print("Top 5 tickets by EV:")
    print(tickets.head(5))
    print()
    
    # 7) Analyze ticket distribution
    ticket_analysis = tickets.groupby("legs").agg(
        count=("EV", "size"),
        avg_ev=("EV", "mean"),
        max_ev=("EV", "max"),
    ).reset_index()
    
    print("Ticket analysis by number of legs:")
    print(ticket_analysis)
    print()
    
    return proj, edged, tickets

if __name__ == "__main__":
    # Test the basic pipeline
    proj, edged, tickets = test_full_pipeline()
    
    # Test with more realistic sample data
    sample_proj, sample_edged, sample_tickets = test_with_sample_data()