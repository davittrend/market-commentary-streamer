from datetime import datetime
from database import MarketDatabase

# Initialize database
db = MarketDatabase()

# Define events
events = [
    ('FOMC_MEETING', 'news', datetime.now().replace(hour=21, minute=55), datetime.now().replace(hour=16, minute=0), 5),
    ('NY_OPEN', 'session', datetime.now().replace(hour=9, minute=30), datetime.now().replace(hour=9, minute=35), 3)
]

# Print events for debugging
print("Events to insert:")
for event in events:
    print(event)

# Insert events
with db.connection() as conn:
    conn.executemany('''
        INSERT INTO special_events 
        VALUES (?, ?, ?, ?, ?)
    ''', events)
    print("Events inserted successfully!")