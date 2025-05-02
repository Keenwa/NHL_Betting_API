from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.routes import router as nhl_router
from app.features import router as features_router
from database.database import init_db, get_db

# Initialize the database on startup
init_db()

app = FastAPI(
    title="NHL Betting API",
    description="Stats, features and shot data for NHL players with betting projections.",
    version="0.2.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main API router
app.include_router(nhl_router)

# Include feature endpoints under /features
app.include_router(features_router)

# Health check endpoint
@app.get("/health", tags=["health"])
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "version": "0.2.0"
    }

# Root endpoint
@app.get("/", tags=["root"])
def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to the NHL Betting API. See /docs for available endpoints.",
        "documentation": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)