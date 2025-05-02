import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, brier_score_loss

class SOGModelValidator:
    """
    Validation framework for SOG prediction models.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.results = None
    
    def validate(
        self, 
        predictions: pd.DataFrame, 
        actuals: pd.DataFrame,
        player_col: str = "shooterPlayerId",
        game_col: str = "game_id",
        line_col: str = "line",
        prob_col: str = "p_over",
        actual_col: str = "actual_sog"
    ) -> Dict[str, float]:
        """
        Validate model predictions against actual results.
        
        Args:
            predictions: DataFrame with model predictions
            actuals: DataFrame with actual SOG results
            player_col: Column with player IDs
            game_col: Column with game IDs
            line_col: Column with SOG lines
            prob_col: Column with model probabilities
            actual_col: Column with actual SOG results
            
        Returns:
            Dictionary with validation metrics
        """
        # Merge predictions with actuals
        merged = predictions.merge(
            actuals,
            on=[player_col, game_col],
            how='inner'
        )
        
        if merged.empty:
            raise ValueError("No matching predictions and actuals found")
        
        # Store results for further analysis
        self.results = merged.copy()
        
        # Calculate binary outcome (over/under)
        self.results['actual_over'] = self.results[actual_col] > self.results[line_col]
        
        # Calculate metrics
        metrics = {}
        
        # Mean absolute error
        metrics['mae'] = mean_absolute_error(
            self.results[actual_col], 
            self.results['mu_final'] if 'mu_final' in self.results.columns else 0
        )
        
        # Root mean squared error
        metrics['rmse'] = np.sqrt(mean_squared_error(
            self.results[actual_col], 
            self.results['mu_final'] if 'mu_final' in self.results.columns else 0
        ))
        
        # Brier score (for probabilistic accuracy)
        metrics['brier_score'] = brier_score_loss(
            self.results['actual_over'],
            self.results[prob_col]
        )
        
        # Calibration metrics
        metrics.update(self._calculate_calibration_metrics())
        
        # Profitability metrics if odds are available
        if 'american_odds' in self.results.columns:
            metrics.update(self._calculate_profitability_metrics())
        
        return metrics
    
    def _calculate_calibration_metrics(self) -> Dict[str, float]:
        """
        Calculate calibration metrics for the model.
        
        Returns:
            Dictionary with calibration metrics
        """
        if self.results is None:
            return {}
        
        metrics = {}
        
        # Create probability buckets
        self.results['prob_bucket'] = pd.cut(
            self.results['p_over'], 
            bins=np.arange(0, 1.1, 0.1),
            labels=np.arange(0.05, 1, 0.1)
        )
        
        # Calculate observed frequencies in each bucket
        calibration = (
            self.results
            .groupby('prob_bucket')
            .agg(
                observed_freq=pd.NamedAgg(column='actual_over', aggfunc='mean'),
                count=pd.NamedAgg(column='actual_over', aggfunc='size')
            )
            .reset_index()
        )
        
        # Calculate calibration error
        if not calibration.empty:
            calibration['error'] = calibration['prob_bucket'].astype(float) - calibration['observed_freq']
            metrics['mean_calibration_error'] = np.sqrt(np.mean(calibration['error'] ** 2))
            
            # Store for plotting
            self.calibration_data = calibration
        
        return metrics
    
    def _calculate_profitability_metrics(self) -> Dict[str, float]:
        """
        Calculate profitability metrics for the model.
        
        Returns:
            Dictionary with profitability metrics
        """
        if self.results is None or 'american_odds' not in self.results.columns:
            return {}
        
        metrics = {}
        
        # Convert American odds to decimal
        self.results['decimal_odds'] = np.where(
            self.results['american_odds'] > 0,
            self.results['american_odds'] / 100 + 1,
            100 / abs(self.results['american_odds']) + 1
        )
        
        # Calculate returns
        self.results['bet_placed'] = self.results['p_over'] > self.results['implied_probability']
        self.results['bet_won'] = self.results['bet_placed'] & self.results['actual_over']
        self.results['profit'] = np.where(
            self.results['bet_placed'],
            np.where(
                self.results['bet_won'],
                self.results['decimal_odds'] - 1,
                -1
            ),
            0
        )
        
        # Calculate metrics
        bets_placed = self.results['bet_placed'].sum()
        if bets_placed > 0:
            metrics['bets_placed'] = bets_placed
            metrics['win_rate'] = self.results['bet_won'].sum() / bets_placed
            metrics['profit_per_bet'] = self.results['profit'].sum() / bets_placed
            metrics['total_profit'] = self.results['profit'].sum()
            metrics['roi'] = self.results['profit'].sum() / bets_placed
        
        # Edge analysis
        for edge in [0.01, 0.03, 0.05, 0.07, 0.10]:
            edge_bets = self.results[self.results['edge'] >= edge]
            if len(edge_bets) > 0:
                metrics[f'edge_{int(edge*100)}_win_rate'] = edge_bets['actual_over'].mean()
                metrics[f'edge_{int(edge*100)}_profit'] = edge_bets['profit'].sum()
                metrics[f'edge_{int(edge*100)}_roi'] = edge_bets['profit'].sum() / len(edge_bets)
        
        return metrics
    
    def plot_calibration(self, save_path: Optional[str] = None):
        """
        Plot calibration curve.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not hasattr(self, 'calibration_data') or self.calibration_data.empty:
            print("No calibration data available")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Plot actual calibration
        plt.scatter(
            self.calibration_data['prob_bucket'], 
            self.calibration_data['observed_freq'],
            s=self.calibration_data['count'] * 5,  # Size based on number of samples
            alpha=0.7,
            label='Observed frequencies'
        )
        
        # Add size legend
        sizes = [50, 100, 200]
        for size in sizes:
            plt.scatter([], [], s=size * 5, alpha=0.7, color='blue', label=f'{size} samples')
        
        plt.title('Calibration Plot')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Observed Frequency')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add perfect calibration text
        mce = self.calibration_data['error'].abs().mean()
        plt.text(0.05, 0.9, f'Mean Calibration Error: {mce:.3f}', transform=plt.gca().transAxes)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_edge_analysis(self, save_path: Optional[str] = None):
        """
        Plot edge analysis showing ROI by edge threshold.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.results is None or 'edge' not in self.results.columns:
            print("No edge data available")
            return
        
        # Create edge buckets
        self.results['edge_bucket'] = pd.cut(
            self.results['edge'], 
            bins=np.arange(-0.2, 0.21, 0.02),
            labels=np.arange(-0.19, 0.2, 0.02)
        )
        
        # Calculate ROI by edge bucket
        edge_analysis = (
            self.results
            .groupby('edge_bucket')
            .agg(
                roi=pd.NamedAgg(column='profit', aggfunc=lambda x: x.sum() / len(x)),
                count=pd.NamedAgg(column='profit', aggfunc='size')
            )
            .reset_index()
        )
        
        # Filter buckets with too few samples
        edge_analysis = edge_analysis[edge_analysis['count'] >= 5]
        
        plt.figure(figsize=(12, 6))
        
        # Plot ROI by edge
        plt.bar(
            edge_analysis['edge_bucket'].astype(float),
            edge_analysis['roi'],
            width=0.015,
            alpha=0.7
        )
        
        # Add count as text
        for _, row in edge_analysis.iterrows():
            plt.text(
                row['edge_bucket'].astype(float),
                row['roi'] + (0.02 if row['roi'] >= 0 else -0.05),
                str(row['count']),
                ha='center',
                va='center',
                fontsize=8
            )
        
        # Add zero line
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add trend line
        if len(edge_analysis) > 1:
            x = edge_analysis['edge_bucket'].astype(float)
            y = edge_analysis['roi']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8)
        
        plt.title('ROI by Edge')
        plt.xlabel('Edge (Predicted - Implied Probability)')
        plt.ylabel('Return on Investment (ROI)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def run_backtest(
        self, 
        historical_odds: pd.DataFrame, 
        historical_games: pd.DataFrame,
        model,
        min_edge: float = 0.05,
        stake: float = 1.0
    ) -> pd.DataFrame:
        """
        Run a backtest of the model on historical odds data.
        
        Args:
            historical_odds: DataFrame with historical odds
            historical_games: DataFrame with historical game outcomes
            model: Model object with a project_player method
            min_edge: Minimum edge to place a bet
            stake: Stake size for each bet
            
        Returns:
            DataFrame with backtest results
        """
        results = []
        
        for _, odds_row in historical_odds.iterrows():
            player_id = odds_row['player_id']
            game_id = odds_row['game_id']
            line = odds_row['line']
            american_odds = odds_row['american_odds']
            
            # Get game information
            game_info = historical_games[historical_games['game_id'] == game_id]
            if game_info.empty:
                continue
            
            game_row = game_info.iloc[0]
            
            # Get model projection
            try:
                projection = model.project_player(
                    player_id=player_id,
                    opponent_id=game_row['opponent_id'],
                    period=game_row.get('period', 3),
                    score_diff=game_row.get('score_diff', 0),
                    is_overtime=game_row.get('is_overtime', False)
                )
                
                # Calculate probability
                p_over = stats.norm.sf(
                    line + 0.5, 
                    loc=projection['mu_final'], 
                    scale=projection['sigma_final']
                )
                
                # Calculate implied probability
                implied_prob = 0.0
                if american_odds > 0:
                    implied_prob = 100 / (american_odds + 100)
                else:
                    implied_prob = abs(american_odds) / (abs(american_odds) + 100)
                
                # Calculate edge
                edge = p_over - implied_prob
                
                # Get actual result
                actual_sog = historical_games.loc[
                    (historical_games['game_id'] == game_id) &
                    (historical_games['player_id'] == player_id),
                    'actual_sog'
                ].values[0]
                
                # Determine if bet would be placed
                bet_placed = edge >= min_edge
                
                # Determine outcome
                actual_over = actual_sog > line
                bet_won = bet_placed and actual_over
                
                # Calculate profit
                profit = 0.0
                if bet_placed:
                    decimal_odds = 0.0
                    if american_odds > 0:
                        decimal_odds = american_odds / 100 + 1
                    else:
                        decimal_odds = 100 / abs(american_odds) + 1
                    
                    profit = (decimal_odds - 1) * stake if bet_won else -stake
                
                # Store result
                results.append({
                    'player_id': player_id,
                    'game_id': game_id,
                    'line': line,
                    'american_odds': american_odds,
                    'mu_final': projection['mu_final'],
                    'sigma_final': projection['sigma_final'],
                    'p_over': p_over,
                    'implied_probability': implied_prob,
                    'edge': edge,
                    'bet_placed': bet_placed,
                    'actual_sog': actual_sog,
                    'actual_over': actual_over,
                    'bet_won': bet_won,
                    'profit': profit,
                    'date': game_row.get('date')
                })
                
            except Exception as e:
                print(f"Error processing {player_id} in game {game_id}: {e}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add cumulative profit
        if not results_df.empty and 'profit' in results_df.columns:
            # Sort by date if available
            if 'date' in results_df.columns:
                results_df = results_df.sort_values('date')
            
            results_df['cumulative_profit'] = results_df['profit'].cumsum()
        
        return results_df