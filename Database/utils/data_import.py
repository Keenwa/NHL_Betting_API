import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from database.repositories import RepositoryFactory

class DataImporter:
    """Utility for importing data into the database"""
    
    def __init__(self, db: Session, data_dir: str = "data"):
        self.db = db
        self.repo_factory = RepositoryFactory(db)
        self.data_dir = data_dir
    
    def import_teams(self, file_path: Optional[str] = None) -> int:
        """
        Import teams from CSV
        
        Args:
            file_path: Path to CSV file (default: data/teams.csv)
            
        Returns:
            Number of teams imported
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "teams.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Team data file not found: {file_path}")
        
        teams_df = pd.read_csv(file_path)
        
        # Map DataFrame columns to database model
        team_repo = self.repo_factory.get_team_repository()
        teams_data = []
        
        for _, row in teams_df.iterrows():
            team_data = {
                "team_id": str(row.get("team_id", "")),
                "code": row.get("code", ""),
                "name": row.get("name", ""),
                "conference": row.get("conference", ""),
                "division": row.get("division", ""),
                "tempo": row.get("tempo", None),
                "sa_per_game": row.get("sa_per_game", None),
                "block_rate": row.get("block_rate", None)
            }
            teams_data.append(team_data)
        
        # Import into database
        teams = team_repo.bulk_create_or_update(teams_data)
        return len(teams)
    
    def import_players(self, file_path: Optional[str] = None) -> int:
        """
        Import players from CSV
        
        Args:
            file_path: Path to CSV file (default: data/skaters.csv)
            
        Returns:
            Number of players imported
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "skaters.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Player data file not found: {file_path}")
        
        players_df = pd.read_csv(file_path)
        
        # Map DataFrame columns to database model
        player_repo = self.repo_factory.get_player_repository()
        players_data = []
        
        for _, row in players_df.iterrows():
            player_data = {
                "player_id": str(row.get("player_id", "")),
                "name": row.get("name", ""),
                "position": row.get("position", ""),
                "team_id": str(row.get("team_id", "")),
                "is_top_shooter": bool(row.get("is_top_shooter", False)),
                "is_star": bool(row.get("is_star", False)),
                "is_top_six": bool(row.get("is_top_six", False)),
                "is_high_post_shooter": bool(row.get("is_high_post_shooter", False)),
                "avg_ev_toi": row.get("avg_ev_toi", None)
            }
            players_data.append(player_data)
        
        # Import into database
        players = player_repo.bulk_create_or_update(players_data)
        return len(players)
    
    def import_shot_data(self, shot_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Import shot data from CSV files
        
        Args:
            shot_dir: Directory containing shot data CSVs (default: data/shot_data)
            
        Returns:
            Dictionary with import statistics
        """
        if shot_dir is None:
            shot_dir = os.path.join(self.data_dir, "shot_data")
        
        if not os.path.exists(shot_dir):
            raise FileNotFoundError(f"Shot data directory not found: {shot_dir}")
        
        # Get CSV files
        csv_files = glob.glob(os.path.join(shot_dir, "**/*.csv"), recursive=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {shot_dir}")
        
        # Repositories
        shot_repo = self.repo_factory.get_shot_repository()
        game_repo = self.repo_factory.get_game_repository()
        
        # Import statistics
        stats = {
            "files_processed": 0,
            "shots_imported": 0,
            "games_imported": 0
        }
        
        # Track games for bulk import
        games_data = {}
        
        for file_path in csv_files:
            try:
                shots_df = pd.read_csv(file_path)
                
                # Process shots in batches
                batch_size = 1000
                total_rows = len(shots_df)
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_df = shots_df.iloc[start_idx:end_idx]
                    
                    shots_data = []
                    for _, row in batch_df.iterrows():
                        # Extract game data for later import
                        game_id = str(row.get("game_id", ""))
                        if game_id and game_id not in games_data:
                            # Detect if playoff game based on season format (e.g., "20232024P")
                            season = str(row.get("season", ""))
                            is_playoff = season.endswith("P") if season else False
                            
                            games_data[game_id] = {
                                "game_id": game_id,
                                "season": season,
                                "date": row.get("date", None),
                                "is_playoff": is_playoff,
                                "home_team_id": str(row.get("home_team_id", "")),
                                "away_team_id": str(row.get("away_team_id", "")),
                                "home_score": row.get("home_score", 0),
                                "away_score": row.get("away_score", 0),
                                "status": row.get("status", "final"),
                                "period": row.get("period", 3)
                            }
                        
                        # Create shot data
                        shot_data = {
                            "shot_id": str(row.get("shotID", "")),
                            "game_id": game_id,
                            "shooter_id": str(row.get("shooterPlayerId", "")),
                            "team_id": str(row.get("teamId", "")),
                            "event": row.get("event", ""),
                            "period": row.get("period", 1),
                            "time": row.get("time", ""),
                            "shot_type": row.get("shotType", ""),
                            "shot_was_on_goal": bool(row.get("shotWasOnGoal", False)),
                            "is_goal": bool(row.get("goal", False)),
                            "hit_post": bool(row.get("hitPost", False)),
                            "is_scoring_chance": bool(row.get("scoringChance", False))
                        }
                        
                        # Derive flags
                        event = shot_data["event"]
                        shot_was_on_goal = shot_data["shot_was_on_goal"]
                        is_goal = shot_data["is_goal"]
                        hit_post = shot_data["hit_post"]
                        
                        shot_data["is_sog"] = (event == "SHOT" and shot_was_on_goal) or (event == "GOAL")
                        shot_data["is_miss"] = event == "MISS" or (event == "SHOT" and not shot_was_on_goal)
                        shot_data["is_block"] = event == "BLOCK"
                        shot_data["is_post"] = hit_post
                        
                        shots_data.append(shot_data)
                    
                    # Import batch of shots
                    imported_shots = shot_repo.bulk_create(shots_data)
                    stats["shots_imported"] += len(imported_shots)
                
                stats["files_processed"] += 1
                
            except Exception as e:
                print(f"Error importing file {file_path}: {e}")
        
        # Import games
        if games_data:
            games_list = list(games_data.values())
            imported_games = game_repo.bulk_create_or_update(games_list)
            stats["games_imported"] = len(imported_games)
        
        # Verify data integrity
        integrity_check, integrity_details = shot_repo.verify_data_integrity()
        stats["integrity_check_passed"] = integrity_check
        stats["integrity_details"] = integrity_details
        
        return stats
    
    def import_betting_lines(self, file_path: Optional[str] = None) -> int:
        """
        Import betting lines from CSV
        
        Args:
            file_path: Path to CSV file (default: data/lines.csv)
            
        Returns:
            Number of lines imported
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "lines.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Betting lines file not found: {file_path}")
        
        lines_df = pd.read_csv(file_path)
        
        # Map DataFrame columns to database model
        line_repo = self.repo_factory.get_betting_line_repository()
        lines_data = []
        
        for _, row in lines_df.iterrows():
            # Convert American odds to implied probability
            american_odds = row.get("american_odds", 0)
            implied_probability = line_repo.american_to_implied(american_odds)
            
            line_data = {
                "player_id": str(row.get("player_id", "")),
                "game_id": str(row.get("game_id", "")),
                "line": float(row.get("line", 0)),
                "american_odds": int(american_odds),
                "implied_probability": implied_probability,
                "bookmaker": row.get("bookmaker", "")
            }
            lines_data.append(line_data)
        
        # Import into database
        lines = line_repo.bulk_create(lines_data)
        return len(lines)
    
    def update_player_statistics(self) -> Dict[str, int]:
        """
        Update player statistics based on shot data
        
        Returns:
            Dictionary with update statistics
        """
        player_repo = self.repo_factory.get_player_repository()
        shot_repo = self.repo_factory.get_shot_repository()
        
        # Get all players
        players = player_repo.get_all()
        
        stats = {
            "players_processed": 0,
            "statistics_updated": 0
        }
        
        for player in players:
            try:
                # Get player's SOG statistics
                player_stats = shot_repo.get_player_sog_stats(player.player_id)
                
                # Update player statistics
                player_repo.update_player_statistics(
                    player_id=player.player_id,
                    season="current",  # Use current as default season
                    stats_data={
                        "games_played": player_stats["games_played"],
                        "total_shots": player_stats["total_shots"],
                        "total_sog": player_stats["total_sog"],
                        "sog_per_game": player_stats["sog_per_game"],
                        "mu": player_stats["weighted_mean"],
                        "sigma": player_stats["weighted_std"]
                    }
                )
                
                # Update player attributes
                shot_repo.update_player_attributes(
                    player_id=player.player_id,
                    attributes={
                        "is_top_shooter": player_stats["sog_per_game"] >= 2.5,
                        "is_star": player_stats["sog_per_game"] >= 3.0,
                        # Other attributes could be updated here
                    }
                )
                
                stats["statistics_updated"] += 1
            except Exception as e:
                print(f"Error updating statistics for player {player.player_id}: {e}")
            
            stats["players_processed"] += 1
        
        return stats