# NHL Betting API

A comprehensive API for NHL statistics, player and team data, SOG (Shots on Goal) projections, and betting edge calculations.

## Features

- **Database-backed architecture** with SQLAlchemy ORM
- **Team and player data** from MoneyPuck
- **Shot data processing** with data integrity validation
- **SOG projections** based on historical weighting, opponent context, and game state
- **Betting edge calculations** and parlay ticket recommendations
- **RESTful API** built with FastAPI

## Installation

### Prerequisites

- Python 3.8+
- pip
- virtualenv (optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nhl-betting-api.git
cd nhl-betting-api
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on the template:
```bash
cp .env.template .env
```

5. Edit the `.env` file to configure your database connection and other settings.

### Database Initialization

Initialize the database and import data:

```bash
python scripts/init_db.py --all
```

This script will:
- Create all database tables
- Import teams, players, and shot data
- Import betting lines
- Update player statistics

## Running the API

Start the FastAPI server:

```bash
python main.py
```

The API will be available at http://localhost:8000

API documentation is available at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## API Endpoints

### Teams
- `GET /teams` - Get all teams
- `GET /teams/{team_id}` - Get a team by ID or code
- `POST /teams` - Create a new team
- `PUT /teams/{team_id}` - Update a team

### Players
- `GET /players` - Get all players
- `GET /players/{player_id}` - Get a player by ID
- `POST /players` - Create a new player
- `PUT /players/{player_id}` - Update a player
- `GET /players/{player_id}/statistics` - Get player statistics

### Games
- `GET /games` - Get games
- `GET /games/{game_id}` - Get a game by ID

### Projections
- `GET /projections/sog/{player_id}` - Get SOG projection for a player
- `GET /projections/edges` - Get SOG edges
- `GET /projections/tickets` - Get ticket recommendations
- `POST /projections/refresh` - Refresh projections

### Betting Lines
- `GET /betting-lines` - Get betting lines
- `POST /betting-lines` - Create a new betting line
- `POST /betting-lines/bulk` - Create multiple betting lines

### Tickets
- `GET /tickets` - Get all tickets
- `POST /tickets` - Create a new ticket
- `GET /tickets/build` - Build ticket recommendations
- `POST /tickets/save` - Build and save ticket recommendations

## Data Model

The API uses the following data model:

- **Teams**: NHL teams with stats like tempo, shots against per game, and block rate
- **Players**: NHL players with attributes relevant to SOG modeling
- **Games**: NHL games with score, status, and period information
- **Shots**: Individual shot events from games
- **Player Statistics**: Aggregated statistics for each player
- **Projections**: SOG projections for players in specific games
- **Betting Lines**: Lines from bookmakers with associated odds
- **Tickets**: Recommended parlays with expected value calculations

## Project Structure

```
nhl_betting_api/
├── app/                      # API application
│   ├── __init__.py
│   ├── routes.py             # API endpoints
│   ├── schemas.py            # Pydantic models for API
│   └── features.py           # Feature endpoints
│
├── database/                 # Database layer
│   ├── __init__.py
│   ├── database.py           # Database connection
│   ├── models.py             # SQLAlchemy ORM models
│   ├── repositories/         # Data access repositories
│   │   ├── __init__.py
│   │   ├── team_repository.py
│   │   ├── player_repository.py
│   │   ├── game_repository.py
│   │   ├── shot_repository.py
│   │   └── projection_repository.py
│   └── utils/                # Database utilities
│       └── data_import.py    # Data import utilities
│
├── scripts/                  # Utility scripts
│   └── init_db.py            # Database initialization
│
├── data/                     # Data directory
│   ├── metadata.json
│   ├── shot_data/
│   │   ├── shots_2007-2023.zip
│   │   ├── shots_2024.zip
│   │   └── … extracted *.csv
│   ├── teams.csv
│   ├── skaters.csv
│   └── lines.csv
│
├── main.py                   # Application entry point
├── requirements.txt          # Dependencies
├── .env.template             # Environment variables template
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
