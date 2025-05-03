import pandas as pd
import numpy as np
from scipy import stats
from models_core.enhanced_model import EnhancedSOGModel
from model_validation import SOGModelValidator
import models_core.backtest_utils as backtest_utils

def main():
    # Initialize model
    print("Initializing enhanced SOG model...")
    model = EnhancedSOGModel(data_dir="data")
    
    # Example: Project a player against an opponent
    player_id = "8471675"  # Example: Alex Ovechkin
    opponent_id = "BOS"    # Example: Boston Bruins
    
    print(f"\nGenerating projection for player {player_id} vs {opponent_id}...")
    projection = model.project_player(
        player_id=player_id,
        opponent_id=opponent_id,
        period=3,
        score_diff=-1,  # Trailing by 1
        is_overtime=False,
        lines=[0.5, 1.5, 2.5, 3.5, 4.5]
    )
    
    # Print key projection details
    print("\nKey projection details:")
    print(f"Mean SOG: {projection.get('mu_final', 0):.2f}")
    print(f"Standard deviation: {projection.get('sigma_final', 0):.2f}")
    
    # Print probabilities for common lines
    print("\nProbabilities for common lines:")
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        line_str = str(line).replace('.', '_')
        key = f'p_over_{line_str}'
        if key in projection:
            print(f"P(SOG > {line}): {projection[key]:.1%}")
        else:
            # Calculate probability if not in projection
            from scipy import stats
            prob = stats.norm.sf(line + 0.5, loc=projection.get('mu_final', 0), scale=projection.get('sigma_final', 1))
            print(f"P(SOG > {line}): {prob:.1%}")
    
    # Print confidence intervals if available
    print("\n90% confidence interval:")
    if 'mu_lower_90' in projection and 'mu_upper_90' in projection:
        print(f"Mean: {projection['mu_lower_90']:.2f} - {projection['mu_upper_90']:.2f}")
    else:
        print("Confidence intervals not available")
    
    if 'p_over_2_5_lower_90' in projection and 'p_over_2_5_upper_90' in projection:
        print(f"P(SOG > 2.5): {projection['p_over_2_5_lower_90']:.1%} - {projection['p_over_2_5_upper_90']:.1%}")
    
    # Example: Calculate edge for a betting line
    line = 2.5
    american_odds = -110
    
    print(f"\nAnalyzing edge for line: over {line} SOG @ {american_odds}")
    try:
        edge = model.calculate_edge(
            player_id=player_id,
            opponent_id=opponent_id,
            line=line,
            american_odds=american_odds,
            period=3,
            score_diff=-1
        )
        
        print(f"Model probability: {edge.get('model_probability', 0):.1%}")
        print(f"Implied probability: {edge.get('implied_probability', 0):.1%}")
        print(f"Edge: {edge.get('edge', 0):.1%}")
    except Exception as e:
        print(f"Error calculating edge: {e}")
        
        # Fallback: Calculate edge manually
        from scipy import stats
        mu = projection.get('mu_final', 0)
        sigma = projection.get('sigma_final', 1)
        model_prob = stats.norm.sf(line + 0.5, loc=mu, scale=sigma)
        
        # Calculate implied probability
        implied_prob = 0.0
        if american_odds > 0:
            implied_prob = 100 / (american_odds + 100)
        else:
            implied_prob = abs(american_odds) / (abs(american_odds) + 100)
        
        edge_value = model_prob - implied_prob
        
        print(f"Model probability (calculated): {model_prob:.1%}")
        print(f"Implied probability (calculated): {implied_prob:.1%}")
        print(f"Edge (calculated): {edge_value:.1%}")
    
    # If you have historical odds and results, run backtest
    print("\nBacktest functionality is currently disabled.")
    # Uncomment the following code once you have historical data
    """
    if False:  # Set to True if you have historical data
        print("\nRunning backtest...")
        historical_odds = pd.read_csv("data/historical_odds.csv")
        historical_results = pd.read_csv("data/historical_results.csv")
        
        backtest_results = backtest_utils.run_model_evaluation(
            model, 
            historical_odds,
            historical_results
        )
        
        print("Backtest metrics:")
        for metric, value in backtest_results['full_dataset'].items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    """

if __name__ == "__main__":
    main()
