#!/usr/bin/env python3
"""
Scheduled data update script for NHL Betting API.

This script:
1. Downloads fresh data from MoneyPuck
2. Updates the database
3. Refreshes projections
4. Logs the update

Usage:
    python scripts/update_data.py [--full] [--no-shots] [--no-projections]

Options:
    --full          Perform a full data refresh (ignore metadata.json)
    --no-shots      Skip shot data download (faster)
    --no-projections Skip projection refresh
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/data_updates.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("data_update")

def parse_args():
    parser = argparse.ArgumentParser(description="Update NHL data")
    parser.add_argument("--full", action="store_true", help="Perform full refresh")
    parser.add_argument("--no-shots", action="store_true", help="Skip shot data download")
    parser.add_argument("--no-projections", action="store_true", help="Skip projection refresh")
    return parser.parse_args()

def main():
    start_time = time.time()
    logger.info("Starting data update process")
    
    args = parse_args()
    
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # 1. Update data
        if args.full:
            logger.info("Performing full data refresh")
            from data_ingest import run_nightly
            run_nightly()
        else:
            logger.info("Performing incremental data update")
            from data_updater import download_data
            download_data()
        
        # 2. Import data into database
        logger.info("Importing data into database")
        from database.database import init_db, SessionLocal
        from database.utils.data_import import DataImporter
        
        # Initialize database if needed
        init_db()
        
        db = SessionLocal()
        importer = DataImporter(db)
        
        try:
            # Import teams and players
            logger.info("Importing teams and players")
            importer.import_teams()
            importer.import_players()
            
            # Import shot data if not skipped
            if not args.no_shots:
                logger.info("Importing shot data")
                shot_stats = importer.import_shot_data()
                logger.info(f"Processed {shot_stats['files_processed']} files")
                logger.info(f"Imported {shot_stats['shots_imported']} shots")
                logger.info(f"Imported {shot_stats['games_imported']} games")
                
                # Update player statistics based on new shot data
                logger.info("Updating player statistics")
                stats_stats = importer.update_player_statistics()
                logger.info(f"Updated statistics for {stats_stats['statistics_updated']} players")
            
            # Import betting lines
            logger.info("Importing betting lines")
            importer.import_betting_lines()
            
        finally:
            db.close()
        
        # 3. Refresh projections if not skipped
        if not args.no_projections:
            logger.info("Refreshing projections")
            import requests
            
            try:
                response = requests.post("http://localhost:8000/projections/refresh")
                if response.status_code == 200:
                    logger.info("Projections refreshed successfully")
                else:
                    logger.error(f"Failed to refresh projections: {response.text}")
            except requests.RequestException as e:
                logger.error(f"Error connecting to API: {e}")
                logger.info("API may not be running, skipping projection refresh")
        
        # Log completion
        duration = time.time() - start_time
        logger.info(f"Data update completed in {duration:.2f} seconds")
        
        # Write to update log
        with open("data/last_update.txt", "w") as f:
            f.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
    
    except Exception as e:
        logger.error(f"Error during data update: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())