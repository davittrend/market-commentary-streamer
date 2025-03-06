import json
import os
from datetime import datetime
from database import MarketDatabase

class StreamAgent:
    def __init__(self, db_path):
        # Initialize MarketDatabase with the provided db_path
        self.db = MarketDatabase(db_path)
        self.current_state = self._load_last_state()
        self.scheduled_events = self._load_scheduled_events()
        self.output_file = os.path.join("..", "data", "stream_output.json")  # Path to output file

    def _load_last_state(self):
        """Load the last market state from the database."""
        try:
            with self.db.connection() as conn:
                row = conn.execute('''
                    SELECT * FROM market_state 
                    ORDER BY timestamp DESC LIMIT 1
                ''').fetchone()
                return row or {'price': 0, 'key_levels': {}, 'market_trend': 'neutral'}
        except Exception as e:
            print(f"Error loading last state: {e}")
            return {'price': 0, 'key_levels': {}, 'market_trend': 'neutral'}

    def _load_scheduled_events(self):
        """Load scheduled events from the database."""
        try:
            with self.db.connection() as conn:
                return conn.execute('''
                    SELECT * FROM special_events 
                    WHERE start_time <= ? AND end_time >= ?
                ''', (datetime.now().isoformat(), datetime.now().isoformat())).fetchall()
        except Exception as e:
            print(f"Error loading scheduled events: {e}")
            return []

    def generate_commentary(self, market_data):
        """Generate commentary based on the current market state and scheduled events."""
        # Check scheduled events
        current_events = self._check_current_events()
        
        # Generate base commentary
        base_commentary = f"Market is currently {market_data['market_trend']}. "
        
        # Add event context
        if current_events:
            base_commentary += f"Note: {current_events[0]['event_id']} is ongoing. "
        
        # Add key levels
        base_commentary += f"Support: {market_data['key_levels']['s1']}, Resistance: {market_data['key_levels']['r1']}."
        
        # Store state
        self._store_current_state(market_data)
        
        # Save commentary to file
        self.save_commentary_to_file(base_commentary)
        
        return base_commentary

    def _check_current_events(self):
        """Check for currently active scheduled events."""
        return [
            {
                'event_id': event[0],
                'start_time': event[2],
                'end_time': event[3]
            }
            for event in self.scheduled_events
            if datetime.fromisoformat(event[2]) <= datetime.now() <= datetime.fromisoformat(event[3])
        ]

    def _store_current_state(self, market_data):
        """Store the current market state in the database."""
        try:
            with self.db.connection() as conn:
                conn.execute('''
                    INSERT INTO market_state 
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(), 
                    market_data['price']['current'], 
                    json.dumps(market_data['key_levels']), 
                    market_data['market_trend'], 
                    market_data.get('volatility', 1.0)
                ))
        except Exception as e:
            print(f"Error storing current state: {e}")

    def save_commentary_to_file(self, commentary):
        """Save the generated commentary to stream_output.json."""
        try:
            # Create the output dictionary
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "commentary": commentary
            }

            # Write to the JSON file
            with open(self.output_file, "w") as f:
                json.dump(output_data, f, indent=4)
            print(f"Commentary saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving commentary to file: {e}")

