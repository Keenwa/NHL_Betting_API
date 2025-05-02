#!/usr/bin/env python3
"""
Initialize the database and import initial data.

Usage:
    python init_db.py
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from database.database import init_db, SessionLocal
from database.utils.data_import import DataImporter

def parse_args():
    parser = argparse.ArgumentParser(description="Initialize database and import data")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--teams", action="store_true", help="Import teams")
    parser.add_argument("--players", action="store_true", help="Import players")
    parser.add_argument("--shots", action="store_true", help="Import shot data")
    parser.add_argument("--lines", action="store_true", help="Import betting lines")
    parser.add_argument("--all", action="store_true", help="Import all data")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Initializing database...")
    init_db()
    print("Database schema created successfully.")
    
    db = SessionLocal()
    try:
        importer = DataImporter(db, data_dir=args.data_dir)
        
        if args.all or args.teams:
            print("Importing teams...")
            num_teams = importer.import_teams()
            print(f"Imported {num_teams} teams.")
        
        if args.all or args.players:
            print("Importing players...")
            num_players = importer.import_players()
            print(f"Imported {num_players} players.")
        
        if args.all or args.shots:
            print("Importing shot data...")
            shot_stats = importer.import_shot_data()
            print(f"Processed {shot_stats['files_processed']} files")
            print(f"Imported {shot_stats['shots_imported']} shots")
            print(f"Imported {shot_stats['games_imported']} games")
            print(f"Integrity check: {'PASSED' if shot_stats['integrity_check_passed'] else 'FAILED'}")
            
            # Update player statistics
            print("Updating player statistics...")
            stats_stats = importer.update_player_statistics()
            print(f"Updated statistics for {stats_stats['statistics_updated']} players")
        
        if args.all or args.lines:
            print("Importing betting lines...")
            num_lines = importer.import_betting_lines()
            print(f"Imported {num_lines} betting lines.")
        
        print("Database initialization complete.")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        db.close()

if __name__ == "__main__":
    main()