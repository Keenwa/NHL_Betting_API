from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment variable, or use a default SQLite URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nhl_betting.db")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    # SQLite requires special connect_args
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    # PostgreSQL, MySQL, etc.
    engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import Base from models
from .models import Base

def get_db():
    """
    Dependency function to get a database session.
    Use this in FastAPI endpoint dependencies.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize the database by creating all tables.
    Call this function when starting the application.
    """
    Base.metadata.create_all(bind=engine)