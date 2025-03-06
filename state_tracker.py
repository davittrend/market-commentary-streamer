from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

class StateTracker:
    def __init__(self, history_length: int = 50):
        self.current_state = None
        self.previous_state = None
        self.state_start_time = None
        self.state_duration = None
        self.price_history = []
        self.state_history = []
        self.bar_size_history = []  # Added to store bar sizes
        self.volume_history = []    # Added to store volume data
        self.history_length = history_length  # Store the history length
        self.momentum_data = {
            "direction": None,
            "strength": 0,
            "duration": 0,
            "exhaustion": False
        }
        self.key_level_interactions = []
        self.patterns = []
        self.last_update_time = datetime.now()
        
    def update(self, current_state: str, market_data: Dict[str, Any]) -> bool:
        """
        Update the market state based on the provided data.
        
        Args:
            current_state: The current market state
            market_data: A dictionary containing market data.
            
        Returns:
            bool: True if the state has changed, False otherwise.
        """
        # Store current price in history
        self.price_history.append(market_data["price"]["current"])
        if len(self.price_history) > self.history_length:  # Use the configured history length
            self.price_history.pop(0)
        
        # Store bar size if available
        if "price" in market_data and "bar_size" in market_data["price"]:
            self.bar_size_history.append(market_data["price"]["bar_size"])
            if len(self.bar_size_history) > self.history_length:
                self.bar_size_history.pop(0)
        
        # Store volume if available
        if "volume" in market_data:
            volume = market_data["volume"].get("buy_volume", 0) + market_data["volume"].get("sell_volume", 0)
            self.volume_history.append(volume)
            if len(self.volume_history) > self.history_length:
                self.volume_history.pop(0)
        
        # Detect candlestick patterns
        candlestick_pattern = self._detect_candlestick_patterns(market_data)
        if candlestick_pattern:
            if not hasattr(self, "candlestick_patterns"):
                self.candlestick_patterns = []
            
            self.candlestick_patterns.append({
                "timestamp": datetime.now(),
                "pattern": candlestick_pattern["pattern"],
                "strength": candlestick_pattern["strength"],
                "price": market_data["price"]["current"]
            })
            
            # Limit history size
            if len(self.candlestick_patterns) > self.history_length:
                self.candlestick_patterns.pop(0)
        
        # Check if the state has changed
        state_changed = current_state != self.current_state
        
        if state_changed:
            # Update state history
            if self.current_state is not None:
                self.state_history.append({
                    "state": self.current_state,
                    "start_time": self.state_start_time,
                    "end_time": datetime.now(),
                    "duration": self.state_duration
                })
                
                # Limit history size
                if len(self.state_history) > self.history_length:
                    self.state_history.pop(0)
            
            # Update current state
            self.previous_state = self.current_state
            self.current_state = current_state
            self.state_start_time = datetime.now()
            self.state_duration = timedelta(seconds=0)
        else:
            # Update state duration
            self.state_duration = datetime.now() - self.state_start_time
        
        # Update momentum analysis
        self._update_momentum(market_data)
        
        # Update key level interactions
        self._update_key_level_interactions(market_data)
        
        # Update pattern detection
        self._update_patterns(market_data)
        
        self.last_update_time = datetime.now()
        
        return state_changed
    
    def _determine_market_state(self, market_data: Dict[str, Any]) -> str:
        """
        Determine the current market state based on the provided data.
        
        Args:
            market_data: A dictionary containing market data.
        
        Returns:
            str: The current market state.
        """
        # Get price data
        current_price = market_data['price']['current']
        price_change = market_data['price']['change']
        percent_change = market_data['price']['percent_change']
        is_large_move = market_data['price'].get('is_large_move', False)
        
        # Get trend data
        higher_highs = market_data['trend']['higher_highs']
        lower_lows = market_data['trend']['lower_lows']
        adx = market_data['trend']['adx']
        
        # Get key levels
        daily_open = market_data['key_levels'].get('daily_open', None)
        
        # Check for big moves
        if is_large_move or abs(percent_change) > 0.3:
            if price_change > 0:
                return "big_move_up"
            else:
                return "big_move_down"
        
        # Check for directional moves
        if higher_highs and adx > 15:
            if daily_open is not None and current_price > daily_open:
                return "moving_up_above_open"
            else:
                return "moving_up"
        
        if lower_lows and adx > 15:
            if daily_open is not None and current_price < daily_open:
                return "moving_down_below_open"
            else:
                return "moving_down"
        
        # Check for consolidation
        if adx < 15 and abs(percent_change) < 0.1:
            if daily_open is not None:
                if abs(current_price - daily_open) < 10:
                    return "consolidating_at_open"
                elif current_price > daily_open:
                    return "consolidating_above_open"
                else:
                    return "consolidating_below_open"
            else:
                return "consolidating"
        
        # Default state
        return "mixed"
    
    def _update_momentum(self, market_data: Dict[str, Any]) -> None:
        """
        Update momentum analysis based on the provided data.
        
        Args:
            market_data: A dictionary containing market data.
        """
        # Determine momentum direction
        if market_data['price']['change'] > 0:
            direction = "up"
        elif market_data['price']['change'] < 0:
            direction = "down"
        else:
            direction = "flat"
        
        # Calculate momentum strength (0-100)
        strength = min(100, abs(market_data['price']['percent_change']) * 10)
        
        # Check for momentum exhaustion
        exhaustion = False
        if direction == self.momentum_data["direction"]:
            # Same direction, update duration
            self.momentum_data["duration"] += 1
            
            # Check for exhaustion
            if self.momentum_data["duration"] > 5 and strength < self.momentum_data["strength"] * 0.8:
                exhaustion = True
        else:
            # Direction changed, reset duration
            self.momentum_data["duration"] = 1
        
        # Update momentum data
        self.momentum_data["direction"] = direction
        self.momentum_data["strength"] = strength
        self.momentum_data["exhaustion"] = exhaustion
    
    def _update_key_level_interactions(self, market_data: Dict[str, Any]) -> None:
        """
        Update key level interactions based on the provided data.
        
        Args:
            market_data: A dictionary containing market data.
        """
        # Check for key level interactions
        if market_data['key_levels']['interaction'] != "none":
            interaction_type = market_data['key_levels']['interaction']
            level_name = None
            level_value = None
            
            if market_data['key_levels']['breakout_level']:
                level_value = float(market_data['key_levels']['breakout_level'])
                
                # Find the level name
                for k, v in market_data['key_levels'].items():
                    if isinstance(v, (int, float)) and abs(v - level_value) < 0.01:
                        level_name = k
                        break
            
            if level_name and level_value:
                # Add to key level interactions
                self.key_level_interactions.append({
                    "timestamp": datetime.now(),
                    "type": interaction_type,
                    "level_name": level_name,
                    "level_value": level_value,
                    "price": market_data['price']['current']
                })
                
                # Limit history size
                if len(self.key_level_interactions) > self.history_length:
                    self.key_level_interactions.pop(0)
    
    def _update_patterns(self, market_data: Dict[str, Any]) -> None:
        """
        Update pattern detection based on the provided data.
        
        Args:
            market_data: A dictionary containing market data.
        """
        # Simple pattern detection
        if len(self.price_history) < 5:
            return
        
        # Check for double top
        if self._detect_double_top():
            self.patterns.append({
                "timestamp": datetime.now(),
                "type": "double_top",
                "price": market_data['price']['current']
            })
        
        # Check for double bottom
        if self._detect_double_bottom():
            self.patterns.append({
                "timestamp": datetime.now(),
                "type": "double_bottom",
                "price": market_data['price']['current']
            })
        
        # Limit history size
        if len(self.patterns) > self.history_length:
            self.patterns.pop(0)
    
    def _detect_double_top(self) -> bool:
        """
        Detect a double top pattern.
        
        Returns:
            bool: True if a double top is detected, False otherwise.
        """
        # Simple double top detection
        if len(self.price_history) < 10:
            return False
        
        # Get recent prices
        recent_prices = self.price_history[-10:]
        
        # Find local maxima
        maxima = []
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                maxima.append((i, recent_prices[i]))
        
        # Check for two similar maxima
        if len(maxima) >= 2:
            # Check the last two maxima
            last_max = maxima[-1]
            prev_max = maxima[-2]
            
            # Check if they are similar in price
            price_diff = abs(last_max[1] - prev_max[1]) / prev_max[1]
            if price_diff < 0.01:  # Within 1%
                return True
        
        return False
    
    def _detect_double_bottom(self) -> bool:
        """
        Detect a double bottom pattern.
        
        Returns:
            bool: True if a double bottom is detected, False otherwise.
        """
        # Simple double bottom detection
        if len(self.price_history) < 10:
            return False
        
        # Get recent prices
        recent_prices = self.price_history[-10:]
        
        # Find local minima
        minima = []
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                minima.append((i, recent_prices[i]))
        
        # Check for two similar minima
        if len(minima) >= 2:
            # Check the last two minima
            last_min = minima[-1]
            prev_min = minima[-2]
            
            # Check if they are similar in price
            price_diff = abs(last_min[1] - prev_min[1]) / prev_min[1]
            if price_diff < 0.01:  # Within 1%
                return True
        
        return False
    
    def get_momentum_analysis(self) -> Dict[str, Any]:
        """
        Get the current momentum analysis.
        
        Returns:
            Dict[str, Any]: The momentum analysis.
        """
        return self.momentum_data
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """
        Get the most recent pattern detection.
        
        Returns:
            Dict[str, Any]: The pattern analysis, or None if no pattern is detected.
        """
        if not self.patterns:
            return {
                "pattern_detected": False
            }
        
        # Get the most recent pattern
        pattern = self.patterns[-1]
        
        # Check if the pattern is recent (within the last 5 minutes)
        if (datetime.now() - pattern["timestamp"]).total_seconds() > 300:
            return {
                "pattern_detected": False
            }
        
        return {
            "pattern_detected": True,
            "pattern_type": pattern["type"],
            "timestamp": pattern["timestamp"],
            "price": pattern["price"]
        }
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current market state.
        
        Returns:
            Dict[str, Any]: A summary of the current market state.
        """
        return {
            "current_state": self.current_state,
            "previous_state": self.previous_state,
            "state_duration": self.state_duration.total_seconds() if self.state_duration else 0,
            "momentum": self.momentum_data,
            "recent_key_level_interactions": self.key_level_interactions[-3:] if self.key_level_interactions else [],
            "recent_patterns": self.patterns[-3:] if self.patterns else []
        }
    
    def to_json(self) -> str:
        """
        Convert the state tracker to a JSON string.
        
        Returns:
            str: A JSON string representation of the state tracker.
        """
        state_data = {
            "current_state": self.current_state,
            "previous_state": self.previous_state,
            "state_start_time": self.state_start_time.isoformat() if self.state_start_time else None,
            "state_duration": self.state_duration.total_seconds() if self.state_duration else 0,
            "momentum": self.momentum_data,
            "recent_key_level_interactions": self.key_level_interactions[-3:] if self.key_level_interactions else [],
            "recent_patterns": self.patterns[-3:] if self.patterns else []
        }
        
        return json.dumps(state_data, indent=2)

    def _detect_candlestick_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect common candlestick patterns"""
        if len(self.price_history) < 3:
            return None
        
        # Get recent price data
        current_open = market_data.get("price", {}).get("open", self.price_history[-1])
        current_close = market_data.get("price", {}).get("current", self.price_history[-1])
        current_high = market_data.get("price", {}).get("high", current_close)
        current_low = market_data.get("price", {}).get("low", current_close)
        
        prev_open = market_data.get("previous_bar", {}).get("open", self.price_history[-2])
        prev_close = market_data.get("previous_bar", {}).get("close", self.price_history[-2])
        prev_high = market_data.get("previous_bar", {}).get("high", prev_close)
        prev_low = market_data.get("previous_bar", {}).get("low", prev_close)
        
        # Detect bullish engulfing
        if prev_close < prev_open and current_close > current_open and \
           current_close > prev_open and current_open < prev_close:
            return {
                "pattern": "bullish_engulfing",
                "strength": "strong" if current_close > prev_high else "moderate"
            }
        
        # Detect bearish engulfing
        if prev_close > prev_open and current_close < current_open and \
           current_close < prev_open and current_open > prev_close:
            return {
                "pattern": "bearish_engulfing",
                "strength": "strong" if current_close < prev_low else "moderate"
            }
        
        # Detect hammer (bullish)
        if current_close > current_open and \
           (current_high - current_close) < 0.2 * (current_close - current_low) and \
           (current_close - current_open) < 0.3 * (current_close - current_low):
            return {
                "pattern": "hammer",
                "strength": "strong" if prev_close < prev_open else "moderate"
            }
        
        # Detect shooting star (bearish)
        if current_close < current_open and \
           (current_close - current_low) < 0.2 * (current_high - current_close) and \
           (current_open - current_close) < 0.3 * (current_high - current_close):
            return {
                "pattern": "shooting_star",
                "strength": "strong" if prev_close > prev_open else "moderate"
            }
        
        # Detect doji
        if abs(current_close - current_open) < 0.1 * (current_high - current_low):
            return {
                "pattern": "doji",
                "strength": "moderate",
                "context": "bullish" if self.price_history[-2] < self.price_history[-1] else "bearish"
            }
        
        return None

    def get_candlestick_pattern(self) -> Dict[str, Any]:
        """Get the most recent candlestick pattern"""
        if not hasattr(self, "candlestick_patterns") or not self.candlestick_patterns:
            return None
        
        return self.candlestick_patterns[-1]

    def get_overall_trend(self) -> str:
        """Determine the overall trend based on price history"""
        if len(self.price_history) < 10:
            return "neutral"
        
        # Simple moving average calculation
        short_ma = sum(self.price_history[-5:]) / 5
        long_ma = sum(self.price_history[-10:]) / 10
        
        # Determine trend based on MA relationship
        if short_ma > long_ma * 1.005:  # 0.5% buffer
            return "bullish"
        elif short_ma < long_ma * 0.995:  # 0.5% buffer
            return "bearish"
        else:
            return "neutral"

    def get_market_phase(self) -> str:
        """Determine the current market phase (accumulation, distribution, markup, markdown)"""
        if len(self.price_history) < 20:
            return "unknown"
        
        # Calculate some metrics
        current_price = self.price_history[-1]
        recent_high = max(self.price_history[-20:])
        recent_low = min(self.price_history[-20:])
        range_size = recent_high - recent_low
        
        # Simple volume trend if available
        volume_trend = "neutral"
        if hasattr(self, "volume_history") and len(self.volume_history) >= 10:
            recent_volume = sum(self.volume_history[-5:]) / 5
            older_volume = sum(self.volume_history[-10:-5]) / 5
            if recent_volume > older_volume * 1.1:
                volume_trend = "increasing"
            elif recent_volume < older_volume * 0.9:
                volume_trend = "decreasing"
        
        # Determine phase
        if current_price < recent_low + range_size * 0.3:
            # Near bottom of range
            if volume_trend == "increasing":
                return "accumulation"
            else:
                return "markdown"
        elif current_price > recent_high - range_size * 0.3:
            # Near top of range
            if volume_trend == "increasing":
                return "markup"
            else:
                return "distribution"
        else:
            # Middle of range
            return "neutral"

