import sqlite3
from datetime import datetime, timedelta
from contextlib import contextmanager
import os

# Register datetime adapters and converters
def adapt_datetime(dt):
    return dt.isoformat()

def convert_datetime(value):
    return datetime.fromisoformat(value.decode())

sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("datetime", convert_datetime)

class MarketDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_database()  # Initialize the database schema
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def _initialize_database(self):
        """Initialize the database schema if it doesn't exist."""
        with self.connection() as conn:
            print("Database connection successful!")
            # Create tables if they don't exist
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_state (
                    timestamp TEXT PRIMARY KEY,
                    price REAL,
                    key_levels TEXT,
                    market_trend TEXT,
                    volatility_index REAL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    market_state TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    feedback_score REAL,
                    commentary TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS commentary_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    commentary TEXT NOT NULL,
                    market_state TEXT NOT NULL,
                    state_duration REAL,
                    event_type TEXT,
                    template_id INTEGER
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS special_events (
                    event_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    impact TEXT NOT NULL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pattern_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    details_json TEXT NOT NULL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS key_level_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level_name TEXT NOT NULL,
                    level_value REAL NOT NULL,
                    interaction_type TEXT NOT NULL,
                    price REAL NOT NULL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS learning_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    template_id INTEGER NOT NULL,
                    event_type TEXT,
                    market_conditions_json TEXT NOT NULL,
                    effectiveness_score REAL,
                    adaptation_json TEXT
                )
            ''')
            print("Schema executed successfully!")

    @contextmanager
    def connection(self):
        """Create and return a database connection."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        try:
            yield conn
            conn.commit()  # Commit changes
        finally:
            conn.close()

    def log_commentary(self, timestamp, commentary, market_state, state_duration):
        """Log commentary to the commentary_history table."""
        # Convert state_duration to a float value in seconds
        if isinstance(state_duration, int):
            state_duration_seconds = float(state_duration)  # Convert to float
        elif isinstance(state_duration, timedelta):
            state_duration_seconds = state_duration.total_seconds()
        elif isinstance(state_duration, float):
            state_duration_seconds = state_duration  # Already a float
        else:
            raise ValueError(f"state_duration must be an integer, float, or timedelta object, got {type(state_duration)}")

        with self.connection() as conn:
            conn.execute('''
                INSERT INTO commentary_history (timestamp, commentary, market_state, state_duration)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, commentary, market_state, state_duration_seconds))
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

