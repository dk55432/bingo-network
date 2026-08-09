"""
Database setup: SQLite via SQLModel.

pip install sqlmodel

SQLite is fine to start — it's a single file (bingo.db), no server to run.
If you outgrow it (many concurrent writers, need to scale across machines),
switch DATABASE_URL to a Postgres connection string; SQLModel/SQLAlchemy
code below doesn't otherwise change.
"""

from sqlmodel import SQLModel, Session, create_engine

DATABASE_URL = "sqlite:///./bingo.db"

# check_same_thread=False is needed because SQLite by default only allows
# the thread that created a connection to use it, but FastAPI can handle
# a request on a different thread than it was created on.
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def create_db_and_tables():
    """Call once at startup to create tables that don't exist yet."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI dependency that yields a database session per request."""
    with Session(engine) as session:
        yield session
