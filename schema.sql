CREATE TABLE IF NOT EXISTS market_state (
    timestamp DATETIME PRIMARY KEY,
    price REAL,
    key_levels JSON,
    market_trend TEXT,
    volatility_index REAL
);

CREATE TABLE IF NOT EXISTS commentary_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME,
    template_id TEXT,
    commentary_text TEXT,
    accuracy_score REAL,
    engagement_score REAL
);

CREATE TABLE IF NOT EXISTS special_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT CHECK(event_type IN ('news', 'session', 'earnings')),
    start_time DATETIME,
    end_time DATETIME,
    impact_level INTEGER
);

CREATE TABLE IF NOT EXISTS user_data (
    user_id TEXT PRIMARY KEY,
    last_interaction DATETIME,
    preferred_name TEXT,
    risk_profile TEXT
);