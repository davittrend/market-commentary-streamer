import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from database import MarketDatabase  # Assumes a custom database class

class PredictiveAnalysis:
    """
    A module for generating predictive market commentary based on historical patterns.
    This uses cautious language and focuses on probabilities rather than certainties.
    """

    def __init__(self, db_path=None):
        """Initialize the predictive analysis module."""
        # Determine the database path.  Use a default path if none is provided.
        self.db_path = db_path or os.path.join("..", "data", "market_stream.db")
        # Create a MarketDatabase object if a path is provided.
        self.db = MarketDatabase(self.db_path) if db_path else None
        # Initialize an empty list to store historical pattern data.
        self.pattern_history = []
        # Initialize an empty dictionary to store key level interaction data.
        self.level_reactions = {}
        # Initialize a dictionary to store success rates of different interaction types.
        self.success_rates = {
            "breakout_up": 0.0,
            "breakout_down": 0.0,
            "rejection": 0.0,
            "bounce": 0.0
        }

        # Load historical data from the database (if available).
        self._load_historical_data()

    def _load_historical_data(self):
        """Load historical pattern data from database."""
        # If no database object is available, return.
        if not self.db:
            return

        try:
            # Establish a connection to the database.  This uses a context manager (`with ... as conn:`)
            # which automatically handles closing the connection even if errors occur.
            with self.db.connection() as conn:
                # Load pattern detections from the database.
                # Execute a SQL query to select all columns from the 'pattern_detections' table,
                # ordered by timestamp in descending order (most recent first), limited to 100 records.
                patterns = conn.execute('''
                    SELECT * FROM pattern_detections
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''').fetchall()  # Fetch all results of the query.

                # If pattern data was retrieved:
                if patterns:
                    # Convert the raw database results into a list of dictionaries,
                    # making the data easier to work with.
                    self.pattern_history = [
                        {
                            "timestamp": p[1],  # The second column (index 1) is assumed to be the timestamp.
                            "pattern_type": p[2], # Third column is the pattern type.
                            "confidence": p[3],   # Fourth column is the confidence.
                            "price": p[4],       # Fifth column is the price.
                            "details": json.loads(p[5]) if p[5] else {}  # Sixth column: details, parse as JSON if it exists.
                        }
                        for p in patterns  # Iterate through each row (p) returned by the query.
                    ]

                # Load key level interactions from the database.
                # Similar to the pattern loading, but from the 'key_level_interactions' table,
                # limited to 200 records.
                interactions = conn.execute('''
                    SELECT * FROM key_level_interactions
                    ORDER BY timestamp DESC
                    LIMIT 200
                ''').fetchall()

                # If interaction data was retrieved:
                if interactions:
                    # Group the interactions by the 'level_name'.
                    for i in interactions:
                        level_name = i[2]  # The third column (index 2) is assumed to be the level name.
                        # If this level name hasn't been seen before, initialize an empty list for it.
                        if level_name not in self.level_reactions:
                            self.level_reactions[level_name] = []

                        # Append the interaction data (as a dictionary) to the list for this level name.
                        self.level_reactions[level_name].append({
                            "timestamp": i[1],          # Second column: timestamp
                            "level_value": i[3],      # Fourth column: level value
                            "interaction_type": i[4], # Fifth column: interaction type
                            "price": i[5]             # Sixth column: price
                        })

                    # Calculate success rates based on the loaded interaction data.
                    self._calculate_success_rates()

        except Exception as e:
            # If any error occurs during database operations, print an error message.
            print(f"Error loading historical data: {e}")

    def _calculate_success_rates(self):
        """Calculate success rates for different interaction types."""
        # Initialize counters for each interaction type.
        breakout_up_success = 0
        breakout_up_total = 0
        breakout_down_success = 0
        breakout_down_total = 0
        rejection_success = 0
        rejection_total = 0
        bounce_success = 0
        bounce_total = 0

        # Analyze each level's interaction history.
        for level_name, interactions in self.level_reactions.items():
            # Sort the interactions by timestamp to ensure chronological order.
            sorted_interactions = sorted(interactions, key=lambda x: x["timestamp"])

            # Iterate through the sorted interactions, comparing each interaction with the *next* one.
            for i in range(len(sorted_interactions) - 1):
                current = sorted_interactions[i]  # The current interaction.
                next_interaction = sorted_interactions[i + 1]  # The next interaction.

                # Convert timestamps to datetime objects for time difference calculations.
                # Handles both string and datetime objects
                current_time = datetime.fromisoformat(current["timestamp"]) if isinstance(current["timestamp"], str) else current["timestamp"]
                next_time = datetime.fromisoformat(next_interaction["timestamp"]) if isinstance(next_interaction["timestamp"], str) else next_interaction["timestamp"]

                # Only consider interactions within a 1-hour timeframe.  This is a hardcoded limit.
                if (next_time - current_time) > timedelta(hours=1):
                    continue  # Skip to the next interaction if the time difference is too large.

                # Analyze breakout success (upwards).
                if current["interaction_type"] == "breakout_up":
                    breakout_up_total += 1  # Increment the total count for this interaction type.
                    # If the price in the *next* interaction is higher than the current price, it's a success.
                    if next_interaction["price"] > current["price"]:
                        breakout_up_success += 1

                # Analyze breakout success (downwards).
                elif current["interaction_type"] == "breakout_down":
                    breakout_down_total += 1
                    # If the price in the *next* interaction is lower, it's a success.
                    if next_interaction["price"] < current["price"]:
                        breakout_down_success += 1

                # Analyze rejection success.
                elif current["interaction_type"] == "rejection":
                    rejection_total += 1
                    level_value = current["level_value"]  # Get the value of the level being rejected.
                    # Determine success based on whether the price moved *away* from the level after the rejection.
                    if (current["price"] > level_value and next_interaction["price"] < current["price"]) or \
                       (current["price"] < level_value and next_interaction["price"] > current["price"]):
                        rejection_success += 1

                # Analyze bounce success.
                elif "bounce" in current["interaction_type"]:
                    bounce_total += 1
                    # Determine success based on whether the price continued in the expected direction after the bounce.
                    if "up" in current["interaction_type"] and next_interaction["price"] > current["price"]:
                        bounce_success += 1
                    elif "down" in current["interaction_type"] and next_interaction["price"] < current["price"]:
                        bounce_success += 1

        # Calculate the success rates for each interaction type.  If no interactions of a type exist,
        # the success rate defaults to 0.5 (essentially a coin flip).
        self.success_rates["breakout_up"] = breakout_up_success / breakout_up_total if breakout_up_total > 0 else 0.5
        self.success_rates["breakout_down"] = breakout_down_success / breakout_down_total if breakout_down_total > 0 else 0.5
        self.success_rates["rejection"] = rejection_success / rejection_total if rejection_total > 0 else 0.5
        self.success_rates["bounce"] = bounce_success / bounce_total if bounce_total > 0 else 0.5

    def generate_prediction(self, market_data: Dict[str, Any], state_tracker: Any) -> str:
        """Generate a predictive commentary based on current market conditions and historical patterns."""
        current_price = market_data["price"]["current"]  # Get the current price from the market data.

        # Check for recent key level interactions (last 3 interactions).  Relies on a 'state_tracker' object.
        recent_interactions = state_tracker.key_level_interactions[-3:] if hasattr(state_tracker, "key_level_interactions") else []

        # If there are recent interactions:
        if recent_interactions:
            latest_interaction = recent_interactions[-1]  # Get the most recent interaction.
            interaction_type = latest_interaction.get("type", "")  # Get the type (e.g., "breakout_up").
            level_name = latest_interaction.get("level_name", "")  # Get the level name (e.g., "R1").
            level_value = latest_interaction.get("level_value", 0.0) # Get the level value.

            # Generate prediction text based on the interaction type and historical success rates.
            if "breakout_up" in interaction_type:
                success_rate = self.success_rates["breakout_up"] * 100
                return (f"Based on historical patterns, breakouts above {level_name} have continued higher "
                        f"{success_rate:.0f}% of the time. Next potential resistance could be at "
                        f"{self._find_next_resistance(market_data, current_price)}.")

            elif "breakout_down" in interaction_type:
                success_rate = self.success_rates["breakout_down"] * 100
                return (f"Based on historical patterns, breakouts below {level_name} have continued lower "
                        f"{success_rate:.0f}% of the time. Next potential support could be at "
                        f"{self._find_next_support(market_data, current_price)}.")

            elif "rejection" in interaction_type:
                success_rate = self.success_rates["rejection"] * 100
                return (f"Historically, rejections at {level_name} have led to reversals {success_rate:.0f}% "
                        f"of the time. Price may move back toward the opposite side of the range.")

            elif "bounce" in interaction_type:
                success_rate = self.success_rates["bounce"] * 100
                direction = "higher" if "up" in interaction_type else "lower"  # Determine direction of bounce.
                return (f"Bounces from {level_name} have historically continued {direction} {success_rate:.0f}% "
                        f"of the time. This level may continue to act as "
                        f"{level_value < current_price and 'support' or 'resistance'}.")

        # If there are no recent interactions, base the prediction on the market phase (from 'state_tracker').
        market_phase = state_tracker.get_market_phase() if hasattr(state_tracker, "get_market_phase") else "unknown"

        if market_phase == "accumulation":
            return ("Price appears to be in an accumulation phase. Historically, this could lead to an upward "
                    "move if buying pressure increases.")

        elif market_phase == "distribution":
            return ("Price appears to be in a distribution phase. Historically, this could lead to a downward "
                    "move if selling pressure increases.")

        elif market_phase == "markup":
            return (f"Current markup phase suggests potential continuation higher. Next key level to watch may be "
                    f"{self._find_next_resistance(market_data, current_price)}.")

        elif market_phase == "markdown":
            return (f"Current markdown phase suggests potential continuation lower. Next key level to watch may be "
                    f"{self._find_next_support(market_data, current_price)}.")


        # If no recent interactions and no market phase, base prediction on overall trend (from 'state_tracker').
        overall_trend = state_tracker.get_overall_trend() if hasattr(state_tracker, "get_overall_trend") else "neutral"

        if overall_trend == "bullish":
            return "The overall trend remains bullish. Price could potentially continue higher if this trend persists."

        elif overall_trend == "bearish":
            return "The overall trend remains bearish. Price could potentially continue lower if this trend persists."

        # If nothing else applies, provide a very generic fallback prediction.
        return ("Market conditions suggest a period of uncertainty may continue. Watch for a break of nearby "
                "support or resistance for potential direction.")

    def _find_next_support(self, market_data: Dict[str, Any], current_price: float) -> str:
        """Find the next support level below current price."""
        closest_support = None  # Initialize variable to store the closest support level.
        closest_distance = float('inf')  # Initialize with positive infinity to find the minimum distance.

        # Iterate through the key levels in the provided market data.
        for k, v in market_data["key_levels"].items():
            # Skip non-numeric values and specific keys.
            if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
                continue

            # If the level is below the current price (i.e., it's a potential support):
            if v < current_price:
                distance = current_price - v  # Calculate the distance between the current price and the level.
                # If this distance is smaller than the current closest distance:
                if distance < closest_distance:
                    closest_distance = distance  # Update the closest distance.
                    closest_support = f"{k} at {v:.2f}"  # Update the closest support level string.

        # Return the closest support level (or a default message if none was found).
        return closest_support or "the next support level"

    def _find_next_resistance(self, market_data: Dict[str, Any], current_price: float) -> str:
        """Find the next resistance level above current price."""
        closest_resistance = None  # Initialize variable to store the closest resistance level.
        closest_distance = float('inf')  # Initialize with positive infinity.

        # Iterate through the key levels in the provided market data.
        for k, v in market_data["key_levels"].items():
            # Skip non-numeric values and specific keys.
            if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
                continue

            # If the level is above the current price (i.e., it's a potential resistance):
            if v > current_price:
                distance = v - current_price  # Calculate the distance.
                # If this distance is smaller than the current closest distance:
                if distance < closest_distance:
                    closest_distance = distance  # Update closest distance.
                    closest_resistance = f"{k} at {v:.2f}"  # Update closest resistance string.

        # Return closest resistance level (or a default message if none was found).
        return closest_resistance or "the next resistance level"
    def log_prediction_outcome(self, prediction: str, outcome: bool):
        """Log the outcome of a prediction to improve future predictions."""

        if not self.db:
            return

        try:
            with self.db.connection() as conn:
                conn.execute('''
                    INSERT INTO learning_data (timestamp, template_id, event_type, market_conditions_json, effectiveness_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),  # Current timestamp
                    0,  # No template ID is used in this case, so set to 0
                    "prediction",  # Event type is "prediction"
                    json.dumps({"prediction": prediction}),  # Store the prediction text as JSON
                    1.0 if outcome else 0.0  # Store 1.0 for success, 0.0 for failure
                ))
        except Exception as e:
            print(f"Error logging prediction outcome: {e}")

