import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from model_validation import SOGModelValidator

def run_model_evaluation(
    model,
    historical_odds: pd.DataFrame,
    historical_results: pd.DataFrame,
    validation_periods: List[Tuple[str, str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run a comprehensive model evaluation across multiple time periods.
    
    Args:
        model: Model object with projection methods
        historical_odds: DataFrame with historical odds
        historical_results: DataFrame with actual results
        validation_periods: List of (start_date, end_date) tuples
        
    Returns:
        Dictionary with metrics for each validation period
    """
    validator = SOGModelValidator()
    
    if validation_periods is None:
        # Default to full dataset
        return {
            'full_dataset': _evaluate_period(
                model, validator, historical_odds, historical_results
            )
        }
    
    results = {}
    
    for period_name, (start_date, end_date) in validation_periods.items():
        # Filter data for period
        period_odds = historical_odds[
            (historical_odds['date'] >= start_date) &
            (historical_odds['date'] <= end_date)
        ]
        
        period_results = historical_results[
            (historical_results['date'] >= start_date) &
            (historical_results['date'] <= end_date)
        ]
        
        if period_odds.empty or period_results.empty:
            print(f"Insufficient data for period {period_name}")
            continue
        
        # Run evaluation
        results[period_name] = _evaluate_period(
            model, validator, period_odds, period_results
        )
    
    return results

def _evaluate_period(
    model,
    validator: SOGModelValidator,
    odds_df: pd.DataFrame,
    results_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate model for a specific time period.
    
    Args:
        model: Model object with projection methods
        validator: SOGModelValidator instance
        odds_df: DataFrame with odds for the period
        results_df: DataFrame with results for the period
        
    Returns:
        Dictionary with metrics
    """
    # Generate predictions
    predictions = []
    
    for _, row in odds_df.iterrows():
        try:
            # Get model projection
            projection = model.project_player(
                player_id=row['player_id'],
                opponent_id=row['opponent_id'],
                period=row.get('period', 3),
                score_diff=row.get('score_diff', 0),
                is_overtime=row.get('is_overtime', False)
            )
            
            # Add to predictions
            predictions.append({
                'player_id': row['player_id'],
                'game_id': row['game_id'],
                'line': row['line'],
                'american_odds': row['american_odds'],
                'mu_final': projection['mu_final'],
                'sigma_final': projection['sigma_final'],
                'p_over': projection.get(f'p_over_{row["line"]}', 
                                        stats.norm.sf(row['line'] + 0.5, 
                                                     projection['mu_final'],
                                                     projection['sigma_final'])),
                'implied_probability': _american_to_implied(row['american_odds']),
                'edge': projection.get(f'p_over_{row["line"]}', 
                                      stats.norm.sf(row['line'] + 0.5, 
                                                   projection['mu_final'],
                                                   projection['sigma_final'])) - 
                        _american_to_implied(row['american_odds'])
            })
        except Exception as e:
            print(f"Error processing {row['player_id']} in game {row['game_id']}: {e}")
    
    predictions_df = pd.DataFrame(predictions)
    
    # Run validation
    metrics = validator.validate(
        predictions=predictions_df,
        actuals=results_df,
        player_col='player_id',
        game_col='game_id',
        line_col='line',
        prob_col='p_over',
        actual_col='actual_sog'
    )
    
    # Generate plots
    validator.plot_calibration()
    validator.plot_edge_analysis()
    
    return metrics

def _american_to_implied(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def plot_backtest_results(backtest_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot backtest results.
    
    Args:
        backtest_df: DataFrame with backtest results
        save_path: Optional path to save the plot
    """
    if backtest_df.empty or 'cumulative_profit' not in backtest_df.columns:
        print("No valid backtest data available")
        return
    
    # Setup figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot 1: Cumulative profit
    ax1.plot(backtest_df.index, backtest_df['cumulative_profit'], 'b-')
    ax1.set_title('Cumulative Profit')
    ax1.set_ylabel('Profit')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Edge distribution
    edge_bins = np.arange(-0.2, 0.21, 0.02)
    ax2.hist(backtest_df['edge'], bins=edge_bins, alpha=0.7)
    ax2.set_title('Edge Distribution')
    ax2.set_xlabel('Edge')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Win rate by edge
    edge_groups = backtest_df.groupby(pd.cut(backtest_df['edge'], bins=edge_bins))
    win_rates = edge_groups['bet_won'].mean()
    counts = edge_groups.size()
    
    # Filter groups with too few samples
    min_samples = 5
    win_rates = win_rates[counts >= min_samples]
    counts = counts[counts >= min_samples]
    
    ax3.bar(
        [x.mid for x in win_rates.index], 
        win_rates.values,
        width=0.015,
        alpha=0.7
    )
    
    # Add count labels
    for i, (idx, count) in enumerate(counts.items()):
        ax3.text(
            idx.mid,
            win_rates.iloc[i] + 0.02,
            str(count),
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    # Add expected win rate line based on edge
    x_vals = np.array([x.mid for x in win_rates.index])
    expected_win_rate = x_vals + 0.5  # Edge + fair probability (0.5)
    ax3.plot(x_vals, expected_win_rate, 'r--', alpha=0.8, label='Expected Win Rate')
    
    ax3.set_title('Win Rate by Edge')
    ax3.set_xlabel('Edge')
    ax3.set_ylabel('Win Rate')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()