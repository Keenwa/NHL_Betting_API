import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import os
import glob

from shot_features import (
    load_shot_data, 
    process_shots,
    compute_dynamic_weighted_mu_sigma,
    apply_opponent_context_with_h2h,
    build_player_matchup_history,
    apply_advanced_game_state_adjustments,
    apply_player_specific_quirks,
    calculate_final_projection_with_confidence,
    get_nhl_averages,
    validate_shot_data_integrity
)

from game_state_model import SituationalModel

class EnhancedSOGModel:
    """
    Enhanced Shots on Goal (SOG) prediction model with improved features
    and comprehensive validation.
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the enhanced SOG model.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir
        
        # Data containers
        self.shot_data = None
        self.processed_shots = None
        self.player_stats = None
        self.opponent_stats = None
        self.h2h_stats = None
        self.league_avg_tempo = None
        self.league_avg_sa_per_game = None
        
        # Situational model
        self.situational_model = None
        self.situational_model_trained = False
        
        # Load and process data
        self.load_data()
    
    def load_data(self):
        """Load and process all required data."""
        try:
            print("Loading shot data...")
            self.shot_data = load_shot_data(self.data_dir)
            self.processed_shots = process_shots(self.shot_data)
            
            print("Validating data integrity...")
            valid = validate_shot_data_integrity(self.processed_shots)
            if not valid:
                print("WARNING: Shot data integrity check failed! Continuing with caution...")
            
            print("Calculating league averages...")
            try:
                self.league_avg_tempo, self.league_avg_sa_per_game = get_nhl_averages(self.processed_shots)
                print(f"League averages - Tempo: {self.league_avg_tempo:.1f}, SA/G: {self.league_avg_sa_per_game:.1f}")
            except Exception as e:
                print(f"Error calculating league averages: {e}")
                self.league_avg_tempo, self.league_avg_sa_per_game = 60.0, 30.0
                print(f"Using default values - Tempo: {self.league_avg_tempo:.1f}, SA/G: {self.league_avg_sa_per_game:.1f}")
            
            print("Computing player statistics...")
            try:
                # Use the fixed compute_dynamic_weighted_mu_sigma function to handle missing is_playoff column
                self.player_stats = compute_dynamic_weighted_mu_sigma(self.processed_shots)
                print(f"Computed statistics for {len(self.player_stats)} players")
            except Exception as e:
                print(f"Error computing player statistics: {e}")
                # Create minimal player stats
                self.player_stats = pd.DataFrame([
                    {"shooterPlayerId": "8471675", "teamCode": "WSH", "mu": 3.4, "sigma": 1.1},  # Ovechkin
                    {"shooterPlayerId": "8478402", "teamCode": "EDM", "mu": 2.8, "sigma": 0.9},  # McDavid
                    {"shooterPlayerId": "8471214", "teamCode": "COL", "mu": 3.1, "sigma": 0.9},  # MacKinnon
                ])
                print(f"Using fallback statistics for {len(self.player_stats)} players")
            
            print("Building head-to-head matchup history...")
            try:
                # Use the fixed build_player_matchup_history function to handle data type inconsistencies
                self.h2h_stats = build_player_matchup_history(self.processed_shots)
                print(f"Built matchup history with {len(self.h2h_stats)} player-team pairs")
            except Exception as e:
                print(f"Error building head-to-head history: {e}")
                self.h2h_stats = pd.DataFrame()
                print("Using empty head-to-head history")
            
            print("Loading opponent team statistics...")
            try:
                self._load_opponent_stats()
                print(f"Loaded statistics for {len(self.opponent_stats)} teams")
            except Exception as e:
                print(f"Error loading opponent stats: {e}")
                # Create minimal opponent stats
                self.opponent_stats = pd.DataFrame([
                    {"teamCode": "BOS", "tempo": 62.5, "sa_per_game": 29.5, "block_rate": 0.28},
                    {"teamCode": "TBL", "tempo": 65.1, "sa_per_game": 32.1, "block_rate": 0.24},
                    {"teamCode": "WSH", "tempo": 63.8, "sa_per_game": 31.2, "block_rate": 0.26},
                    {"teamCode": "FLA", "tempo": 67.2, "sa_per_game": 30.8, "block_rate": 0.22},
                ])
                print(f"Using fallback statistics for {len(self.opponent_stats)} teams")
            
            print("Training situational model...")
            try:
                self._train_situational_model()
                status = "trained" if self.situational_model_trained else "not trained"
                print(f"Situational model {status}")
            except Exception as e:
                print(f"Error training situational model: {e}")
                self.situational_model_trained = False
                print("Situational model not available")
            
            print("Model initialization complete!")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            print("Falling back to default values...")
            
            # Set default values for essential components
            self.league_avg_tempo, self.league_avg_sa_per_game = 60.0, 30.0
            self.situational_model_trained = False
            
            # Create minimal player stats
            self.player_stats = pd.DataFrame([
                {"shooterPlayerId": "8471675", "teamCode": "WSH", "mu": 3.4, "sigma": 1.1},  # Ovechkin
                {"shooterPlayerId": "8478402", "teamCode": "EDM", "mu": 2.8, "sigma": 0.9},  # McDavid
                {"shooterPlayerId": "8471214", "teamCode": "COL", "mu": 3.1, "sigma": 0.9},  # MacKinnon
            ])
            
            # Create minimal opponent stats
            self.opponent_stats = pd.DataFrame([
                {"teamCode": "BOS", "tempo": 62.5, "sa_per_game": 29.5, "block_rate": 0.28},
                {"teamCode": "TBL", "tempo": 65.1, "sa_per_game": 32.1, "block_rate": 0.24},
                {"teamCode": "WSH", "tempo": 63.8, "sa_per_game": 31.2, "block_rate": 0.26},
                {"teamCode": "FLA", "tempo": 67.2, "sa_per_game": 30.8, "block_rate": 0.22},
            ])
            
            # Empty h2h stats
            self.h2h_stats = pd.DataFrame()
    
    def _load_opponent_stats(self):
        """Load opponent team statistics from file or calculate from data."""
        # Check if opponent stats file exists
        opponent_file = os.path.join(self.data_dir, "opponent_stats.csv")
        
        if os.path.exists(opponent_file):
            # Load from file
            self.opponent_stats = pd.read_csv(opponent_file)
        else:
            # Calculate from shot data
            print("Calculating opponent statistics from shot data...")
            self.opponent_stats = self._calculate_opponent_stats()
            
            # Save for future use
            self.opponent_stats.to_csv(opponent_file, index=False)
    
    def _calculate_opponent_stats(self):
        """Calculate opponent statistics from shot data."""
    # First, calculate shots per game for each team
    team_shots = self.processed_shots.groupby(['teamCode', 'game_id']).size().reset_index(name='shots')
    shots_per_game = team_shots.groupby('teamCode')['shots'].mean().reset_index(name='shots_per_game')
    
    # Calculate average tempo by team (events per game)
    games_per_team = team_shots.groupby('teamCode')['game_id'].nunique().reset_index(name='games')
    
    # Group by team to calculate stats
    team_stats = pd.DataFrame()
    
    # For each team, calculate:
    for team in self.processed_shots['teamCode'].unique():
        # Get team shots
        team_sog = self.processed_shots[self.processed_shots['teamCode'] == team]['is_sog'].sum()
        team_blocks = self.processed_shots[self.processed_shots['teamCode'] == team]['is_block'].sum()
        
        # Safely calculate block rate
        block_rate = 0.0
        if team_sog + team_blocks > 0:
            block_rate = team_blocks / (team_sog + team_blocks)
        
        # Get shots per game
        if team in shots_per_game['teamCode'].values:
            spg = shots_per_game.loc[shots_per_game['teamCode'] == team, 'shots_per_game'].iloc[0]
        else:
            spg = 0.0
        
        # Get number of games
        if team in games_per_team['teamCode'].values:
            games = games_per_team.loc[games_per_team['teamCode'] == team, 'games'].iloc[0]
        else:
            games = 1
        
        # Calculate tempo (approximate)
        tempo = 60.0  # Default value
        if games > 0:
            tempo = len(self.processed_shots[self.processed_shots['teamCode'] == team]) / games
        
        # Add to team stats
        team_stats = pd.concat([
            team_stats,
            pd.DataFrame({
                'teamCode': [team],
                'tempo': [tempo],
                'sa_per_game': [spg],
                'block_rate': [block_rate],
                'pp_time_share': [0.2],  # Default value
                'pk_weakness': [0.8]     # Default value
            })
        ], ignore_index=True)
    
    return team_stats
    
    def _train_situational_model(self):
        """Train the situational model on historical data."""
        # Check if we have necessary game state data
        if 'score_diff' in self.processed_shots.columns and 'time_remaining' in self.processed_shots.columns:
            # Train model
            self.situational_model.train(self.processed_shots)
            self.situational_model_trained = True
        else:
            print("Warning: Insufficient game state data to train situational model")
            self.situational_model_trained = False
    
    def project_player(
        self,
        player_id: str,
        opponent_id: str,
        period: int = 3,
        score_diff: int = 0,
        is_overtime: bool = False,
        time_remaining: int = 600,  # Default to half period
        expected_pp: float = 2.0,
        actual_pp: float = 2.0,
        last_game_ev_toi: Optional[float] = None,
        lines: List[float] = [0.5, 1.5, 2.5, 3.5, 4.5]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive SOG projection for a player.
        
        Args:
            player_id: Player ID
            opponent_id: Opponent team ID
            period: Current period
            score_diff: Score difference (player team - opponent)
            is_overtime: Whether game is in overtime
            time_remaining: Time remaining in seconds
            expected_pp: Expected power play opportunities
            actual_pp: Actual power play opportunities
            last_game_ev_toi: Even strength TOI in last game
            lines: SOG lines to calculate probabilities for
            
        Returns:
            Dictionary with comprehensive projection details
        """
        # Check if player exists
        player_data = self.player_stats[self.player_stats['shooterPlayerId'] == player_id]
        if player_data.empty:
            raise ValueError(f"Player {player_id} not found in data")
        
        # Check if opponent exists
        opponent_data = self.opponent_stats[self.opponent_stats['teamCode'] == opponent_id]
        if opponent_data.empty:
            raise ValueError(f"Opponent {opponent_id} not found in data")
        
        # 1. Apply opponent context with head-to-head history
        with_opponent = apply_opponent_context_with_h2h(
            player_data,
            opponent_data,
            self.h2h_stats,
            self.league_avg_tempo,
            self.league_avg_sa_per_game,
            h2h_weight=0.3
        )
        
        # 2. Create game state data
        game_state_data = pd.DataFrame([{
            'shooterPlayerId': player_id,
            'teamCode': player_data['teamCode'].iloc[0],
            'period': period,
            'score_diff': score_diff,
            'is_overtime': is_overtime,
            'time_remaining': time_remaining,
            'trail_probability': self._calculate_trail_probability(score_diff, period, time_remaining),
            # Add player attributes from data or use defaults
            'is_top_shooter': True,  # Should come from player data
            'is_star': True,         # Should come from player data
            'is_top_six': True       # Should come from player data
        }])
        
        # 3. Apply advanced game state adjustments
        with_game_state = apply_advanced_game_state_adjustments(
            with_opponent,
            game_state_data,
            self.situational_model if self.situational_model_trained else None
        )
        
        # 4. Create player quirks data
        player_quirks_data = pd.DataFrame([{
            'shooterPlayerId': player_id,
            'is_high_post_shooter': player_id in ["8478483", "8478402"],  # Example IDs
            'expected_pp': expected_pp,
            'actual_pp': actual_pp,
            'last_game_ev_toi': last_game_ev_toi,
            'avg_ev_toi': 15.0  # Default value, should come from player data
        }])
        
        # 5. Apply player-specific quirks
        player_df = with_game_state.merge(player_quirks_data, on='shooterPlayerId', how='left')
        with_quirks = apply_player_specific_quirks(player_df)
        
        # 6. Calculate final projection with confidence intervals
        final_projection = calculate_final_projection_with_confidence(
            with_quirks,
            lines=lines
        )
        
        # 7. Return the first row as a dictionary
        result = final_projection.iloc[0].to_dict()
        
        return result
    
    def _calculate_trail_probability(
        self,
        score_diff: int,
        period: int,
        time_remaining: int
    ) -> float:
        """
        Calculate probability of trailing at end of game.
        
        Args:
            score_diff: Current score difference
            period: Current period
            time_remaining: Time remaining in seconds
            
        Returns:
            Probability of trailing (0-1)
        """
        # Already trailing
        if score_diff < 0:
            return 1.0
        
        # Simple model based on period, score diff, and time remaining
        base_prob = 0.5  # Default 50/50 if tied in 1st period
        
        # Adjust for score
        score_factor = 0.15 * score_diff
        
        # Adjust for period
        if period == 1:
            period_factor = 0
        elif period == 2:
            period_factor = -0.1
        else:
            period_factor = -0.2
        
        # Adjust for time remaining (as percentage of period)
        time_pct = time_remaining / 1200.0  # 1200 seconds in a period
        time_factor = -0.1 * (1 - time_pct)  # More impact as time decreases
        
        # Calculate final probability
        trail_prob = base_prob + period_factor + time_factor - score_factor
        
        # Ensure probability is between 0 and 1
        return max(0.0, min(1.0, trail_prob))
    
    def calculate_edge(
        self,
        player_id: str,
        opponent_id: str,
        line: float,
        american_odds: int,
        period: int = 3,
        score_diff: int = 0,
        is_overtime: bool = False,
        time_remaining: int = 600,
        expected_pp: float = 2.0,
        actual_pp: float = 2.0,
        last_game_ev_toi: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate edge between model probability and implied odds.
        
        Args:
            player_id: Player ID
            opponent_id: Opponent team ID
            line: SOG line
            american_odds: American odds format
            period: Current period
            score_diff: Score difference
            is_overtime: Whether game is in overtime
            time_remaining: Time remaining in seconds
            expected_pp: Expected power play opportunities
            actual_pp: Actual power play opportunities
            last_game_ev_toi: Even strength TOI in last game
            
        Returns:
            Dictionary with edge calculation details
        """
        # Generate projection
        projection = self.project_player(
            player_id=player_id,
            opponent_id=opponent_id,
            period=period,
            score_diff=score_diff,
            is_overtime=is_overtime,
            time_remaining=time_remaining,
            expected_pp=expected_pp,
            actual_pp=actual_pp,
            last_game_ev_toi=last_game_ev_toi
        )
        
        # Check if probability for this line exists in projection
        line_str = str(line).replace('.', '_')
        key = f'p_over_{line_str}'
        
        if key in projection:
            model_prob = projection[key]
        else:
            # Calculate probability
            model_prob = stats.norm.sf(
                line + 0.5, 
                loc=projection['mu_final'], 
                scale=projection['sigma_final']
            )
        
        # Calculate implied probability from odds
        implied_prob = self._american_to_implied(american_odds)
        
        # Calculate edge
        edge = model_prob - implied_prob
        
        return {
            'player_id': player_id,
            'opponent_id': opponent_id,
            'line': line,
            'american_odds': american_odds,
            'mu': projection['mu_final'],
            'sigma': projection['sigma_final'],
            'model_probability': model_prob,
            'implied_probability': implied_prob,
            'edge': edge
        }
    
    def _american_to_implied(self, american_odds: int) -> float:
        """
        Convert American odds to implied probability.
        
        Args:
            american_odds: American odds format
            
        Returns:
            Implied probability (0-1)
        """
        if american_odds > 0:
            # Positive odds (e.g., +110)
            return 100 / (american_odds + 100)
        else:
            # Negative odds (e.g., -110)
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def find_edges(
        self,
        odds_df: pd.DataFrame,
        min_edge: float = 0.05,
        game_state: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Find edges in a dataframe of odds.
        
        Args:
            odds_df: DataFrame with odds data
            min_edge: Minimum edge threshold
            game_state: Optional game state data to use for all projections
            
        Returns:
            DataFrame with edges above threshold
        """
        results = []
        
        for _, row in odds_df.iterrows():
            try:
                # Extract values from row with defaults
                player_id = row['player_id']
                opponent_id = row.get('opponent_id')
                line = row['line']
                american_odds = row['american_odds']
                
                # Use provided game state or defaults
                if game_state is not None:
                    period = game_state.get('period', 3)
                    score_diff = game_state.get('score_diff', 0)
                    is_overtime = game_state.get('is_overtime', False)
                    time_remaining = game_state.get('time_remaining', 600)
                    expected_pp = game_state.get('expected_pp', 2.0)
                    actual_pp = game_state.get('actual_pp', 2.0)
                    last_game_ev_toi = game_state.get('last_game_ev_toi')
                else:
                    # Use row values if available, otherwise defaults
                    period = row.get('period', 3)
                    score_diff = row.get('score_diff', 0)
                    is_overtime = row.get('is_overtime', False)
                    time_remaining = row.get('time_remaining', 600)
                    expected_pp = row.get('expected_pp', 2.0)
                    actual_pp = row.get('actual_pp', 2.0)
                    last_game_ev_toi = row.get('last_game_ev_toi')
                
                # Calculate edge
                edge = self.calculate_edge(
                    player_id=player_id,
                    opponent_id=opponent_id,
                    line=line,
                    american_odds=american_odds,
                    period=period,
                    score_diff=score_diff,
                    is_overtime=is_overtime,
                    time_remaining=time_remaining,
                    expected_pp=expected_pp,
                    actual_pp=actual_pp,
                    last_game_ev_toi=last_game_ev_toi
                )
                
                # Add if above threshold
                if edge['edge'] >= min_edge:
                    results.append(edge)
                
            except Exception as e:
                print(f"Error processing {row['player_id']}: {e}")
        
        # Convert to DataFrame and sort by edge
        if results:
            return pd.DataFrame(results).sort_values('edge', ascending=False)
        else:
            return pd.DataFrame()
    
    def build_tickets(
        self,
        edges_df: pd.DataFrame,
        max_legs: int = 3
    ) -> pd.DataFrame:
        """
        Build ticket (parlay) recommendations from edges.
        
        Args:
            edges_df: DataFrame with edges above threshold
            max_legs: Maximum number of legs in a parlay
            
        Returns:
            DataFrame with ticket recommendations
        """
        from itertools import combinations
        
        if edges_df.empty:
            return pd.DataFrame()
        
        # Convert American odds to decimal
        df = edges_df.copy()
        df["decimal_odds"] = np.where(
            df["american_odds"] > 0,
            df["american_odds"] / 100 + 1,
            100 / abs(df["american_odds"]) + 1
        )
        
        tickets = []
        
        # Single legs
        for _, leg in df.iterrows():
            tickets.append({
                "legs": 1,
                "players": [leg["player_id"]],
                "opponents": [leg["opponent_id"]],
                "lines": [leg["line"]],
                "american_odds": [leg["american_odds"]],
                "combined_odds": leg["decimal_odds"],
                "combined_p": leg["model_probability"],
                "EV": leg["decimal_odds"] * leg["model_probability"] - 1,
            })
        
        # Multi-leg combos (2 to max_legs)
        for r in range(2, max_legs + 1):
            for combo_idx in combinations(df.index, r):
                legs = df.loc[list(combo_idx)]
                
                # One leg per game (assuming unique player/opponent combinations)
                unique_pairs = set(zip(legs["player_id"], legs["opponent_id"]))
                if len(unique_pairs) < len(legs):
                    continue
                
                dec_odds = legs["decimal_odds"].prod()
                p_win = legs["model_probability"].prod()
                ev = dec_odds * p_win - 1
                
                if ev <= 0:
                    continue
                
                tickets.append({
                    "legs": r,
                    "players": legs["player_id"].tolist(),
                    "opponents": legs["opponent_id"].tolist(),
                    "lines": legs["line"].tolist(),
                    "american_odds": legs["american_odds"].tolist(),
                    "combined_odds": dec_odds,
                    "combined_p": p_win,
                    "EV": ev,
                })
        
        # Convert to DataFrame and sort by EV
        return pd.DataFrame(tickets).sort_values("EV", ascending=False)
    
    def get_player_name(self, player_id: str) -> str:
        """
        Get player name from ID using skaters.csv if available.
        
        Args:
            player_id: Player ID
            
        Returns:
            Player name or ID if not found
        """
        # Check if skaters file exists
        skaters_file = os.path.join(self.data_dir, "skaters.csv")
        
        if os.path.exists(skaters_file):
            # Load player data
            skaters = pd.read_csv(skaters_file)
            
            # Find player
            player = skaters[skaters['player_id'] == player_id]
            
            if not player.empty and 'name' in player.columns:
                return player['name'].iloc[0]
        
        return player_id  # Return ID if name not found
    
    def get_team_name(self, team_code: str) -> str:
        """
        Get team name from code using teams.csv if available.
        
        Args:
            team_code: Team code
            
        Returns:
            Team name or code if not found
        """
        # Check if teams file exists
        teams_file = os.path.join(self.data_dir, "teams.csv")
        
        if os.path.exists(teams_file):
            # Load team data
            teams = pd.read_csv(teams_file)
            
            # Find team
            team = teams[teams['code'] == team_code]
            
            if not team.empty and 'name' in team.columns:
                return team['name'].iloc[0]
        
        return team_code  # Return code if name not found
    
    def generate_projection_report(
        self,
        player_id: str,
        opponent_id: str,
        line: float,
        american_odds: int,
        **kwargs
    ) -> str:
        """
        Generate a detailed projection report as text.
        
        Args:
            player_id: Player ID
            opponent_id: Opponent team ID
            line: SOG line
            american_odds: American odds
            **kwargs: Additional game state parameters
            
        Returns:
            Detailed projection report as text
        """
        # Get projection
        projection = self.project_player(
            player_id=player_id,
            opponent_id=opponent_id,
            **kwargs
        )
        
        # Get names
        player_name = self.get_player_name(player_id)
        team_name = self.get_team_name(projection.get('teamCode', ''))
        opponent_name = self.get_team_name(opponent_id)
        
        # Calculate edge
        edge_calc = self.calculate_edge(
            player_id=player_id,
            opponent_id=opponent_id,
            line=line,
            american_odds=american_odds,
            **kwargs
        )
        
        # Format line
        line_str = str(line).replace('.', '_')
        
        # Build report
        report = []
        report.append(f"SOG PROJECTION REPORT: {player_name} ({team_name}) vs {opponent_name}")
        report.append("-" * 80)
        
        # Game state
        period = kwargs.get('period', 3)
        score_diff = kwargs.get('score_diff', 0)
        is_overtime = kwargs.get('is_overtime', False)
        
        game_state = f"Period: {period}"
        if is_overtime:
            game_state += " (OT)"
        
        if score_diff > 0:
            game_state += f", Leading by {score_diff}"
        elif score_diff < 0:
            game_state += f", Trailing by {abs(score_diff)}"
        else:
            game_state += ", Tied"
        
        report.append(f"Game State: {game_state}")
        report.append("")
        
        # Projection details
        report.append("PROJECTION DETAILS:")
        report.append(f"Base Mean: {projection.get('mu', 0):.2f} SOG")
        report.append(f"Opponent Adjusted: {projection.get('mu_opponent', 0):.2f} SOG")
        report.append(f"Game State Adjusted: {projection.get('mu_game_state', 0):.2f} SOG")
        report.append(f"Final Projection: {projection.get('mu_final', 0):.2f} SOG (σ = {projection.get('sigma_final', 0):.2f})")
        report.append("")
        
        # Confidence intervals
        report.append("CONFIDENCE INTERVALS:")
        for level in [80, 90, 95]:
            report.append(f"{level}% CI: {projection.get(f'mu_lower_{level}', 0):.2f} - {projection.get(f'mu_upper_{level}', 0):.2f} SOG")
        report.append("")
        
        # Line probabilities
        report.append("LINE PROBABILITIES:")
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
            l_str = str(l).replace('.', '_')
            key = f'p_over_{l_str}'
            if key in projection:
                report.append(f"Over {l} SOG: {projection[key]:.1%}")
        report.append("")
        
        # Betting analysis
        report.append("BETTING ANALYSIS:")
        report.append(f"Line: Over {line} SOG")
        report.append(f"Odds: {american_odds} ({edge_calc['implied_probability']:.1%} implied)")
        report.append(f"Model Probability: {edge_calc['model_probability']:.1%}")
        report.append(f"Edge: {edge_calc['edge']:.1%}")
        
        # Add value rating
        if edge_calc['edge'] >= 0.10:
            rating = "STRONG VALUE"
        elif edge_calc['edge'] >= 0.05:
            rating = "GOOD VALUE"
        elif edge_calc['edge'] >= 0.02:
            rating = "SLIGHT VALUE"
        elif edge_calc['edge'] <= -0.10:
            rating = "STRONG FADE"
        elif edge_calc['edge'] <= -0.05:
            rating = "FADE"
        elif edge_calc['edge'] <= -0.02:
            rating = "SLIGHT FADE"
        else:
            rating = "FAIR PRICE"
        
        report.append(f"Rating: {rating}")
        
        return "\n".join(report)