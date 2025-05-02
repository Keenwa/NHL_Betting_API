import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from itertools import combinations


class SOGProjection:
    """
    Handles SOG (Shots on Goal) projections, edge calculations, and ticket building
    for NHL betting.
    """
    
    def __init__(self, player_stats: Dict = None, odds_file: str = 'data/lines.csv'):
        """
        Initialize the SOG Projection class.
        
        Args:
            player_stats: Dictionary of player statistics
            odds_file: Path to the odds CSV file
        """
        self.player_stats = player_stats or {}
        self.odds_file = odds_file
        self.odds_data = self.load_odds()
        self.edge_threshold = 0.05  # Edge threshold as decimal (5 percentage points)
        
    def load_odds(self) -> pd.DataFrame:
        """
        Load betting odds from the odds file.
        
        Returns:
            DataFrame containing betting odds
        """
        try:
            return pd.read_csv(self.odds_file)
        except Exception as e:
            print(f"Error loading odds data: {e}")
            return pd.DataFrame()
    
    def calculate_probability(self, mean: float, std: float, line: float) -> float:
        """
        Calculate the probability of a player exceeding a SOG line.
        
        Args:
            mean: Mean SOG projection
            std: Standard deviation
            line: SOG line to exceed
            
        Returns:
            Probability of exceeding the line
        """
        # Using survival function (1 - CDF) of normal distribution
        # Since we want P(SOG > line), we need to use line + 0.5 for the continuous approximation
        # of the discrete SOG distribution
        return stats.norm.sf(line + 0.5, loc=mean, scale=std)
    
    def american_to_implied_probability(self, american_odds: int) -> float:
        """
        Convert American odds to implied probability.
        
        Args:
            american_odds: American odds format
            
        Returns:
            Implied probability (0-1)
        """
        if american_odds > 0:
            # Positive odds (e.g., +120)
            return 100 / (american_odds + 100)
        else:
            # Negative odds (e.g., -150)
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def calculate_edge(self, model_prob: float, american_odds: int) -> float:
        """
        Calculate the edge between model probability and implied odds.
        
        Args:
            model_prob: Model probability
            american_odds: American odds format
            
        Returns:
            Edge as a decimal (model_prob - implied_prob)
        """
        implied_prob = self.american_to_implied_probability(american_odds)
        return model_prob - implied_prob
    
    def project_player(self, player_id: str, line: float, opponent_id: str, 
                      period: int = 3, score_diff: int = 0, is_overtime: bool = False,
                      expected_pp: float = 2.0, actual_pp: float = 2.0,
                      last_game_ev_toi: Optional[float] = None) -> Dict:
        """
        Project a player's SOG and calculate probability of exceeding the line.
        
        Args:
            player_id: ID of the player
            line: SOG line to exceed
            opponent_id: ID of the opposing team
            period: Current period
            score_diff: Score difference (player team - opponent team)
            is_overtime: Whether the game is in overtime
            expected_pp: Expected power play opportunities
            actual_pp: Actual power play opportunities
            last_game_ev_toi: Even strength TOI in last game
            
        Returns:
            Dictionary with projection details
        """
        if player_id not in self.player_stats:
            raise KeyError(f"Player {player_id} not found in player stats")
            
        # Get player team
        team_id = self.player_stats[player_id]['team_id']
        
        # Get player mean and std from historical weighting
        base_mean = self.player_stats[player_id].get('base_mean', 0)
        base_std = self.player_stats[player_id].get('base_std', 0)
        
        # Apply opponent context adjustments
        opponent_adj_mean = base_mean
        opponent_adj_std = base_std
        
        # Apply pace multiplier
        if 'tempo' in self.player_stats[player_id]:
            opponent_tempo = self.player_stats[player_id]['tempo']
            nhl_avg_tempo = self.player_stats.get('nhl_avg_tempo', 1.0)
            opponent_adj_mean *= (opponent_tempo / nhl_avg_tempo)
        
        # Apply team shots against per game adjustment
        if 'sa_per_game' in self.player_stats[player_id]:
            opponent_sa_per_game = self.player_stats[player_id]['sa_per_game']
            nhl_avg_sa_per_game = self.player_stats.get('nhl_avg_sa_per_game', 30.0)
            sa_diff = opponent_sa_per_game - nhl_avg_sa_per_game
            if abs(sa_diff) >= 2:
                opponent_adj_mean += 0.15 * (sa_diff / 2)
        
        # Apply series block rate adjustment
        if 'block_rate' in self.player_stats[player_id]:
            block_rate = self.player_stats[player_id]['block_rate']
            if block_rate > 0.30:
                opponent_adj_mean -= 0.25
                opponent_adj_std *= 1.10
            elif block_rate < 0.22:
                opponent_adj_mean += 0.15
        
        # Apply shutdown matchup flag
        if self.player_stats[player_id].get('is_shadowed', False):
            opponent_adj_mean -= 0.25
        
        # Apply game state adjustments
        game_state_adj_mean = opponent_adj_mean
        game_state_adj_std = opponent_adj_std
        
        # Lead-protect taper
        if period > 2 and score_diff >= 2:
            if self.player_stats[player_id].get('is_top_shooter', False):
                game_state_adj_mean -= 0.3
        
        # Chase-mode bonus
        trail_probability = 0.0
        if score_diff < 0:
            trail_probability = 1.0
        elif period == 1:
            trail_probability = 0.3 if score_diff == 0 else 0.2
        elif period == 2:
            trail_probability = 0.4 if score_diff == 0 else 0.3
        else:
            trail_probability = 0.5 if score_diff == 0 else 0.4
            
        if trail_probability >= 0.6 or score_diff < 0:
            if self.player_stats[player_id].get('is_star', False):
                game_state_adj_mean += 0.4
        
        # Overtime/comeback for specific teams
        if team_id == 'CAR' and (is_overtime or score_diff <= -2):
            if self.player_stats[player_id].get('is_top_six', False):
                game_state_adj_mean += 0.3
        
        # Increase variance in overtime
        if is_overtime:
            game_state_adj_std *= 1.15
        
        # Apply player-specific quirks
        final_mean = game_state_adj_mean
        final_std = game_state_adj_std
        
        # High-post shooter adjustment
        if self.player_stats[player_id].get('is_high_post_shooter', False):
            final_mean -= 0.05
            final_std *= 1.1
        
        # Power-play share adjustment
        pp_diff = actual_pp - expected_pp
        final_mean += 0.15 * (pp_diff / 2)
        
        # TOI adjustment based on last game
        if last_game_ev_toi:
            avg_toi = self.player_stats[player_id].get('avg_ev_toi', 0)
            if avg_toi > 0:
                toi_factor = last_game_ev_toi / avg_toi
                final_mean *= toi_factor
        
        # Calculate probability of exceeding the line
        p_over = self.calculate_probability(final_mean, final_std, line)
        
        return {
            'player_id': player_id,
            'team_id': team_id,
            'opponent_id': opponent_id,
            'line': line,
            'base_mean': base_mean,
            'base_std': base_std,
            'opponent_adj_mean': opponent_adj_mean,
            'opponent_adj_std': opponent_adj_std,
            'game_state_adj_mean': game_state_adj_mean,
            'game_state_adj_std': game_state_adj_std,
            'final_mean': final_mean,
            'final_std': final_std,
            'p_over': p_over
        }
    
    def process_player_odds(self, player_id: str, game_id: str, line: float, 
                           american_odds: int, opponent_id: str, period: int = 3, 
                           score_diff: int = 0, is_overtime: bool = False,
                           expected_pp: float = 2.0, actual_pp: float = 2.0,
                           last_game_ev_toi: Optional[float] = None) -> Dict:
        """
        Process a player's odds, calculating probability and edge.
        
        Args:
            player_id: ID of the player
            game_id: ID of the game
            line: SOG line to exceed
            american_odds: American odds for the bet
            opponent_id: ID of the opposing team
            period: Current period
            score_diff: Score difference (player team - opponent team)
            is_overtime: Whether the game is in overtime
            expected_pp: Expected power play opportunities
            actual_pp: Actual power play opportunities
            last_game_ev_toi: Even strength TOI in last game
            
        Returns:
            Dictionary with odds processing details
        """
        # Get projection
        projection = self.project_player(
            player_id, line, opponent_id, period, score_diff, 
            is_overtime, expected_pp, actual_pp, last_game_ev_toi
        )
        
        # Calculate implied probability and edge
        implied_prob = self.american_to_implied_probability(american_odds)
        edge = projection['p_over'] - implied_prob
        
        return {
            'player_id': player_id,
            'game_id': game_id,
            'line': line,
            'american_odds': american_odds,
            'model_probability': projection['p_over'],
            'implied_probability': implied_prob,
            'edge': edge,
            'final_mean': projection['final_mean'],
            'final_std': projection['final_std']
        }
    
    def process_all_lines(self, lines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all lines in the odds file.
        
        Args:
            lines_df: DataFrame containing betting lines
            
        Returns:
            DataFrame with processed odds
        """
        results = []
        
        for _, row in lines_df.iterrows():
            try:
                result = self.process_player_odds(
                    player_id=row['player_id'],
                    game_id=row['game_id'],
                    line=row['line'],
                    american_odds=row['american_odds'],
                    opponent_id=row['opponent_id'],
                    period=row.get('period', 3),
                    score_diff=row.get('score_diff', 0),
                    is_overtime=row.get('is_overtime', False),
                    expected_pp=row.get('expected_pp', 2.0),
                    actual_pp=row.get('actual_pp', 2.0),
                    last_game_ev_toi=row.get('last_game_ev_toi')
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {row['player_id']}: {e}")
        
        return pd.DataFrame(results)
    
    def find_edges(self) -> pd.DataFrame:
        """
        Find edges in the current odds data.
        
        Returns:
            DataFrame with edges
        """
        processed = self.process_all_lines(self.odds_data)
        
        # Filter by edge threshold
        edges = processed[processed['edge'] >= self.edge_threshold]
        
        # Sort by edge
        edges = edges.sort_values('edge', ascending=False)
        
        return edges
    
    def build_tickets(self, edges: pd.DataFrame, max_legs: int = 3) -> pd.DataFrame:
        """
        Build tickets (parlays) from edges.
        
        Args:
            edges: DataFrame with edges
            max_legs: Maximum number of legs in a parlay
            
        Returns:
            DataFrame with tickets
        """
        df = edges.copy()
        
        # Convert American odds to decimal
        df["decimal_odds"] = np.where(
            df["american_odds"] > 0,
            df["american_odds"] / 100 + 1,
            100 / abs(df["american_odds"]) + 1
        )
        
        tickets = []
        
        # Single legs (already filtered by edge threshold)
        for _, leg in df.iterrows():
            tickets.append({
                "legs": 1,
                "players": [leg["player_id"]],
                "games": [leg["game_id"]],
                "combined_odds": leg["decimal_odds"],
                "combined_p": leg["model_probability"],
                "EV": leg["decimal_odds"] * leg["model_probability"] - 1,
            })
        
        # Multi-leg combos
        for r in range(2, max_legs + 1):
            for combo_idx in combinations(df.index, r):
                legs = df.loc[list(combo_idx)]
                
                # One leg per game
                if legs["game_id"].nunique() < len(legs):
                    continue
                
                dec_odds = legs["decimal_odds"].prod()
                p_win = legs["model_probability"].prod()
                ev = dec_odds * p_win - 1
                
                if ev <= 0:
                    continue
                
                tickets.append({
                    "legs": r,
                    "players": legs["player_id"].tolist(),
                    "games": legs["game_id"].tolist(),
                    "combined_odds": dec_odds,
                    "combined_p": p_win,
                    "EV": ev,
                })
        
        return pd.DataFrame(tickets).sort_values("EV", ascending=False)
    
    def get_best_tickets(self, max_legs: int = 3) -> pd.DataFrame:
        """
        Get the best tickets based on EV.
        
        Args:
            max_legs: Maximum number of legs in a parlay
            
        Returns:
            DataFrame with best tickets
        """
        edges = self.find_edges()
        if edges.empty:
            return pd.DataFrame()
        
        tickets = self.build_tickets(edges, max_legs)
        return tickets

# Function from existing code
def build_tickets(
    edges: pd.DataFrame,
    odds_col: str = "american_odds",
    player_col: str = "shooterPlayerId",
    game_col: str = "game_id",
    pcol: str = "p_over",
    min_edge: float = 0.05,
    max_legs: int = 3,
) -> pd.DataFrame:
    """
    Enumerate all 1- to max_legs-leg parlays that keep one leg per game
    and have positive expected value (EV).

    • edges must already contain columns:  edge, american_odds, p_over, game_id
    • EV = (product of decimal odds) × (product of model probabilities) − 1
    • Keeps only combos whose EV > 0

    Returns a DataFrame with one row per valid ticket.
    """
    df = edges.copy()

    # 1) decimal odds
    df["decimal_odds"] = np.where(
        df[odds_col] > 0,
        df[odds_col] / 100 + 1,
        100 / (-df[odds_col]) + 1
    )

    # 2) keep only single legs with sufficient model edge
    df = df[df["edge"] >= min_edge]

    tickets = []
    for r in range(1, max_legs + 1):
        for combo_idx in combinations(df.index, r):
            legs = df.loc[list(combo_idx)]

            # one leg per game
            if legs[game_col].nunique() < len(legs):
                continue

            dec_odds = legs["decimal_odds"].prod()
            p_win    = legs[pcol].prod()
            ev       = dec_odds * p_win - 1

            if ev <= 0:
                continue

            tickets.append(
                {
                    "legs":          r,
                    "players":       legs[player_col].tolist(),
                    "games":         legs[game_col].tolist(),
                    "combined_odds": dec_odds,
                    "combined_p":    p_win,
                    "EV":            ev,
                }
            )

    return pd.DataFrame(tickets)