import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

# Add the project root to Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models_core.enhanced_model import EnhancedSOGModel
from model_validation import SOGModelValidator
import models_core.backtest_utils

def main():
    # Initialize model
    print("Initializing enhanced SOG model...")
    model = EnhancedSOGModel(data_dir="data")
    
    # Example: Project a player against an opponent
    player_id = "8471675"  # Example: Alex Ovechkin
    opponent_id = "BOS"    # Example: Boston Bruins
    
    print(f"\nGenerating projection for player {player_id} vs {opponent_id}...")
    try:
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
    
    except Exception as e:
        print(f"Error generating projection: {e}")
        print("This might be due to missing player or opponent data.")
    
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
        
        # Fallback: Calculate edge manually if projection exists
        if 'projection' in locals():
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
    
    # Example: Generate a detailed report
    try:
        print("\nGenerating detailed projection report...")
        report = model.generate_projection_report(
            player_id=player_id,
            opponent_id=opponent_id,
            line=line,
            american_odds=american_odds,
            period=3,
            score_diff=-1,
            is_overtime=False
        )
        print(report)
    except Exception as e:
        print(f"Error generating report: {e}")
    
    # Example: Find edges in sample odds data
    print("\n" + "="*50)
    print("EXAMPLE: Finding Edges in Sample Odds")
    print("="*50)
    
    # Create sample odds data
    sample_odds = pd.DataFrame([
        {'player_id': '8471675', 'opponent_id': 'BOS', 'line': 2.5, 'american_odds': -120},  # Ovechkin
        {'player_id': '8478402', 'opponent_id': 'CGY', 'line': 2.5, 'american_odds': -110},  # McDavid
        {'player_id': '8471214', 'opponent_id': 'MIN', 'line': 3.5, 'american_odds': +150},  # MacKinnon
        {'player_id': '8475311', 'opponent_id': 'NYR', 'line': 1.5, 'american_odds': -130},  # Gaudreau
        {'player_id': '8474141', 'opponent_id': 'TOR', 'line': 2.5, 'american_odds': +110},  # Barzal
    ])
    
    try:
        edges_df = model.find_edges(sample_odds, min_edge=0.02)
        if not edges_df.empty:
            print(f"Found {len(edges_df)} edges:")
            for _, edge in edges_df.iterrows():
                print(f"Player {edge['player_id']} vs {edge['opponent_id']}: "
                      f"Line {edge['line']} at {edge['american_odds']}, "
                      f"Edge: {edge['edge']:.1%}")
        else:
            print("No edges found")
    except Exception as e:
        print(f"Error finding edges: {e}")
    
    # If you have historical odds and results, run backtest
    print("\nBacktest functionality is currently disabled.")
    print("To enable backtest, provide historical_odds.csv and historical_results.csv files.")
    
    # Uncomment the following code once you have historical data
    """
    if os.path.exists("data/historical_odds.csv") and os.path.exists("data/historical_results.csv"):
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
