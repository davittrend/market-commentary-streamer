import json
import time
import os
import sys
import numpy as np
from datetime import datetime, timedelta

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "config"))
from configmanager import ConfigManager

from state_tracker import StateTracker
from commentary_generator import CommentaryGenerator
from database import MarketDatabase

class PriceActionMonitor:
  def __init__(self, config_path=os.path.join("..", "config", "config.json")):
      self.config = ConfigManager(config_path)
      self.db_path = self.config.get("database_path", os.path.join("..", "data", "market_stream.db"))
      self.db = MarketDatabase(self.db_path)
      self.state_tracker = StateTracker(history_length=self.config.get("history_length", 50))
      self.commentary_gen = CommentaryGenerator()
      
      self.last_commentary_time = datetime.min
      self.last_significant_event_time = datetime.min
      self.significant_move_threshold = self.config.get("significant_move_atr_multiple", 1.5)
      self.regular_update_interval = timedelta(minutes=self.config.get("regular_update_interval_minutes", 5))
      
      # Initialize tracking variables
      self.last_price = None
      self.last_modified_time = 0
      self.last_quick_update_time = 0
      self.last_quick_data = None
      self.last_full_market_data = None
  
  def process_market_data(self, market_data, is_quick_update=False):
    """Process new market data and generate commentary if needed"""
    # Store full market data for future reference
    if not is_quick_update:
        self.last_full_market_data = market_data.copy()
    
    # Validate and correct key levels
    market_data = self._validate_key_levels(market_data)
    
    # Update state tracker
    current_state = self.determine_market_state(market_data)
    state_changed = self.state_tracker.update(current_state, market_data)
    
    # Store market data in database (only for full updates)
    if not is_quick_update:
        try:
            with self.db.connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO market_state (timestamp, price, key_levels, market_trend, volatility_index)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    market_data['timestamp'],
                    market_data['price']['current'],
                    json.dumps(market_data['key_levels']),
                    market_data.get('market_state', current_state),
                    market_data.get('volatility', 1.0)
                ))
        except Exception as e:
            print(f"Error storing market data: {e}")
    
    # Detect significant events
    events = self._detect_events(market_data)
    
    # For quick updates, use different criteria to determine if commentary is needed
    if is_quick_update:
        should_comment, event_type = self._should_generate_quick_commentary(events)
    else:
        should_comment, event_type = self._should_generate_commentary(events, state_changed)
    
    if should_comment:
        # Generate commentary
        commentary, template_id = self.commentary_gen.generate_new_state_commentary(market_data, self.state_tracker)
        
        # Store in database (only for full updates)
        if not is_quick_update:
            try:
                state_duration_seconds = self.state_tracker.state_duration.total_seconds()
                
                self.db.log_commentary(
                    timestamp=market_data["timestamp"],
                    commentary=commentary,
                    market_state=current_state,
                    state_duration=state_duration_seconds
                )
            except Exception as e:
                print(f"Error logging commentary: {e}")
        
        # Update last commentary time
        self.last_commentary_time = datetime.now()
        
        # If this was a significant event, update that time too
        if event_type == "significant":
            self.last_significant_event_time = datetime.now()
        
        return commentary
    
    return None
  
  def _validate_key_levels(self, market_data):
      """Validate and correct key levels if necessary"""
      if "key_levels" not in market_data:
          return market_data
      
      key_levels = market_data["key_levels"]
      
      # Log suspicious values for debugging
      if "daily_high" in key_levels and "daily_low" in key_levels:
          if key_levels["daily_high"] < key_levels["daily_low"]:
              print(f"Warning: daily_high ({key_levels['daily_high']}) is less than daily_low ({key_levels['daily_low']})")
      
      if "prior_day_high" in key_levels and "prior_day_low" in key_levels:
          if key_levels["prior_day_high"] < key_levels["prior_day_low"]:
              print(f"Warning: prior_day_high ({key_levels['prior_day_high']}) is less than prior_day_low ({key_levels['prior_day_low']})")
      
      # Check if we need to swap prior_day values (if they're incorrectly labeled)
      if "prior_day_high" in key_levels and "prior_day_low" in key_levels:
          if key_levels["prior_day_high"] < key_levels["prior_day_low"]:
              # They appear to be swapped, so swap them back
              temp = key_levels["prior_day_high"]
              key_levels["prior_day_high"] = key_levels["prior_day_low"]
              key_levels["prior_day_low"] = temp
              print(f"Corrected: Swapped prior_day_high and prior_day_low values")
      
      # If daily_high is missing or suspicious, try to derive it from other data
      if "daily_high" not in key_levels or key_levels["daily_high"] < key_levels.get("daily_low", 0):
          # Try to use the current price's high if available
          if "price" in market_data and "high" in market_data["price"]:
              key_levels["daily_high"] = market_data["price"]["high"]
              print(f"Corrected: Set daily_high to current price high: {key_levels['daily_high']}")
      
      # If daily_low is missing or suspicious, try to derive it
      if "daily_low" not in key_levels or key_levels["daily_low"] > key_levels.get("daily_high", float('inf')):
          # Try to use the current price's low if available
          if "price" in market_data and "low" in market_data["price"]:
              key_levels["daily_low"] = market_data["price"]["low"]
              print(f"Corrected: Set daily_low to current price low: {key_levels['daily_low']}")
      
      # Validate swing_high and swing_low
      if "swing_high" in key_levels and "swing_low" in key_levels:
          if key_levels["swing_high"] < key_levels["swing_low"]:
              print(f"Warning: swing_high ({key_levels['swing_high']}) is less than swing_low ({key_levels['swing_low']})")
              # Don't auto-correct these as they might be legitimate based on recent price action
      
      # Update the market_data with corrected key_levels
      market_data["key_levels"] = key_levels
      return market_data
  
  def determine_market_state(self, market_data):
      """Determine the current market state with enhanced detection and multi-timeframe awareness"""
      if 'trend' not in market_data:
          return 'unknown'
      
      trend = market_data['trend']
      price = market_data['price']
      
      # Extract key metrics
      adx = trend.get('adx', 0)
      atr = trend.get('atr', 0)
      higher_highs = trend.get('higher_highs', False)
      lower_lows = trend.get('lower_lows', False)
      distance_from_ema = trend.get('distance_from_ema', 0)
      distance_from_sma = trend.get('distance_from_sma', 0)
      volume_oscillator = market_data.get('volume', {}).get('volume_oscillator', 0)
      
      # Get overall trend if available (looking back 50-100 bars)
      overall_trend = "neutral"
      if hasattr(self.state_tracker, "get_overall_trend"):
          overall_trend = self.state_tracker.get_overall_trend()
      
      # Check for daily open relationship (added as per discussion)
      daily_open = market_data["key_levels"].get("daily_open", None)
      above_daily_open = daily_open is not None and price['current'] > daily_open
      
      # Store previous market state for persistence logic
      if not hasattr(self, 'previous_market_state'):
          self.previous_market_state = None
      
      # Determine the new market state
      new_market_state = None
      
      # Check for breakouts first (highest priority)
      if market_data["key_levels"]["interaction"] != "none":
          if market_data["key_levels"]["interaction"] == "breakout_down":
              new_market_state = 'breakout_down'
          elif market_data["key_levels"]["interaction"] == "breakout_up":
              new_market_state = 'breakout_up'
      
      # Check for big moves (second priority)
      if not new_market_state and ('bar_size' in price and price.get('is_large_move', False) or abs(price['change']) > 40):
          price_change = price['change']
          if price_change >= 40:  # 40+ points up
              new_market_state = 'big_move_up'
          elif price_change <= -40:  # 40+ points down
              new_market_state = 'big_move_down'
      
      # Check for rally/selloff (sustained large moves)
      if not new_market_state and abs(distance_from_ema) > 150:  # Far from EMA
          if distance_from_ema > 150 and higher_highs:
              new_market_state = 'rally'
          elif distance_from_ema < -150 and lower_lows:
              new_market_state = 'selloff'
      
      # Check for trend conditions
      if not new_market_state and adx > 28:  # Strong trend
          if distance_from_sma > 0 and distance_from_ema > 0:
              # Add context from overall trend
              if overall_trend == "bullish":
                  new_market_state = 'strong_uptrend'
              else:
                  new_market_state = 'uptrend'
          elif distance_from_sma < 0 and distance_from_ema < 0:
              # Add context from overall trend
              if overall_trend == "bearish":
                  new_market_state = 'strong_downtrend'
              else:
                  new_market_state = 'downtrend'
      
      # Check for moving conditions (weaker trend)
      if not new_market_state and price['change'] > 0 and not (adx > 28 and distance_from_sma > 0 and distance_from_ema > 0):
          # Add daily open context
          if above_daily_open:
              new_market_state = 'moving_up_above_open'
          else:
              new_market_state = 'moving_up'
      elif not new_market_state and price['change'] < 0 and not (adx > 28 and distance_from_sma < 0 and distance_from_ema < 0):
          # Add daily open context
          if not above_daily_open:
              new_market_state = 'moving_down_below_open'
          else:
              new_market_state = 'moving_down'
      
      # Default to sideways/consolidation with more context
      if not new_market_state:
          if adx < 25 and abs(volume_oscillator) < 150:
              # Add context about position relative to daily open
              if above_daily_open:
                  new_market_state = 'sideways_above_open'
              elif daily_open is not None:
                  new_market_state = 'sideways_below_open'
              else:
                  new_market_state = 'sideways'
          else:
              new_market_state = 'sideways'  # Fallback
      
      # Apply state persistence logic to prevent rapid oscillations
      if self.previous_market_state is not None and self.previous_market_state != new_market_state:
          # For big moves, require stronger evidence
          if new_market_state in ['big_move_up', 'big_move_down'] and self.previous_market_state not in ['big_move_up', 'big_move_down']:
              # Check if this is sustained over multiple bars
              if not self._is_sustained_move(market_data, direction='up' if new_market_state == 'big_move_up' else 'down'):
                  # Use a transitional state instead
                  new_market_state = 'moving_up' if new_market_state == 'big_move_up' else 'moving_down'
        
          # For trend changes, require confirmation
          if (self.previous_market_state.startswith('uptrend') and new_market_state.startswith('downtrend')) or \
             (self.previous_market_state.startswith('downtrend') and new_market_state.startswith('uptrend')):
              # Check if the trend change is confirmed
              if not self._is_trend_change_confirmed(market_data):
                  # Keep previous state until confirmed
                  new_market_state = self.previous_market_state
      
      # Update previous state for next call
      self.previous_market_state = new_market_state
    
      return new_market_state

  def _is_sustained_move(self, market_data, direction, min_bars=2):
      """Check if a price move has been sustained over multiple bars"""
      if not hasattr(self.state_tracker, "price_history") or len(self.state_tracker.price_history) < min_bars + 1:
          return False
          
      recent_changes = []
      for i in range(1, min_bars + 1):
          if i < len(self.state_tracker.price_history):
              change = self.state_tracker.price_history[-i] - self.state_tracker.price_history[-i-1]
              recent_changes.append(change)
      
      # Check if all recent changes are in the same direction
      if direction == 'up':
          return all(change > 0 for change in recent_changes)
      else:
          return all(change < 0 for change in recent_changes)

  def _is_trend_change_confirmed(self, market_data):
      """Check if a trend change is confirmed by multiple indicators"""
      # Need at least 3 confirming factors for a trend change
      confirmation_count = 0
      
      # Check price relative to moving averages
      if market_data['trend'].get('distance_from_ema', 0) > 0 and market_data['trend'].get('distance_from_sma', 0) > 0:
          confirmation_count += 1  # Bullish
      elif market_data['trend'].get('distance_from_ema', 0) < 0 and market_data['trend'].get('distance_from_sma', 0) < 0:
          confirmation_count += 1  # Bearish
      
      # Check momentum
      if market_data.get('momentum', {}).get('direction') == 'up':
          confirmation_count += 1  # Bullish
      elif market_data.get('momentum', {}).get('direction') == 'down':
          confirmation_count += 1  # Bearish
      
      # Check volume
      if market_data.get('volume', {}).get('buy_volume', 0) > market_data.get('volume', {}).get('sell_volume', 0):
          confirmation_count += 1  # Bullish
      elif market_data.get('volume', {}).get('buy_volume', 0) < market_data.get('volume', {}).get('sell_volume', 0):
          confirmation_count += 1  # Bearish
      
      # Check if we have enough confirmation
      return confirmation_count >= 2
    
  def _detect_events(self, market_data):
      """Detect significant market events from the data with enhanced detection"""
      events = []
      
      # Check for key level interactions
      if market_data["key_levels"]["interaction"] != "none":
          interaction_type = market_data["key_levels"]["interaction"]
          level_name = market_data["key_levels"]["breakout_level"] or market_data["key_levels"]["rejection_level"]
          
          events.append({
              "type": "key_level",
              "importance": "high",
              "details": {
                  "interaction_type": interaction_type,
                  "level": level_name,
                  "level_value": self._get_level_value(market_data, level_name)
              }
          })
          
      # Check for significant price moves based on comparison to previous candles
      if hasattr(self.state_tracker, "price_history") and len(self.state_tracker.price_history) >= 4:
          current_bar_size = market_data["price"].get("bar_size", 0)
          current_volume = market_data["volume"]["buy_volume"] + market_data["volume"]["sell_volume"]
          
          # Calculate average of previous 3-4 candles
          prev_bar_sizes = self.state_tracker.bar_size_history[-4:-1] if hasattr(self.state_tracker, "bar_size_history") else []
          prev_volumes = self.state_tracker.volume_history[-4:-1] if hasattr(self.state_tracker, "volume_history") else []
          
          if prev_bar_sizes and prev_volumes:  # Make sure we have data
              avg_prev_bar_size = sum(prev_bar_sizes) / len(prev_bar_sizes)
              avg_prev_volume = sum(prev_volumes) / len(prev_volumes)
              
              # Check if current bar is significantly larger
              if (current_bar_size > avg_prev_bar_size * 3 or 
                  current_volume > avg_prev_volume * 3):
                  events.append({
                      "type": "significant_move",
                      "importance": "high",
                      "details": {
                          "change": market_data["price"]["change"],
                          "direction": "up" if market_data["price"]["change"] > 0 else "down",
                          "bar_size_ratio": current_bar_size / avg_prev_bar_size if avg_prev_bar_size > 0 else float('inf'),
                          "volume_ratio": current_volume / avg_prev_volume if avg_prev_volume > 0 else float('inf')
                      }
                  })
          
      # Check for very large price moves regardless of previous candles
      if abs(market_data["price"]["change"]) > 100:  # More than 100 points is always significant
          events.append({
              "type": "significant_move",
              "importance": "high",
              "details": {
                  "change": market_data["price"]["change"],
                  "direction": "up" if market_data["price"]["change"] > 0 else "down",
                  "magnitude": abs(market_data["price"]["change"])
              }
          })

      # Check for daily high/low breakouts with improved detection
      current_price = market_data["price"]["current"]
      daily_high = market_data["key_levels"].get("daily_high", float('inf'))
      daily_low = market_data["key_levels"].get("daily_low", float('-inf'))
      daily_open = market_data["key_levels"].get("daily_open", 0)
      swing_high = market_data["key_levels"].get("swing_high", 0)
      swing_low = market_data["key_levels"].get("swing_low", 0)
      
      if hasattr(self.state_tracker, "price_history") and len(self.state_tracker.price_history) >= 2:
          previous_price = self.state_tracker.price_history[-2]
          
          # Detect breakout of daily high/low
          if previous_price < daily_high and current_price >= daily_high:
              events.append({
                  "type": "breakout",
                  "importance": "high",
                  "details": {
                      "level_type": "daily_high",
                      "level_value": daily_high,
                      "direction": "up"
                  }
              })
          elif previous_price > daily_low and current_price <= daily_low:
              events.append({
                  "type": "breakout",
                  "importance": "high",
                  "details": {
                      "level_type": "daily_low",
                      "level_value": daily_low,
                      "direction": "down"
                  }
              })
              
          # Detect crossing of swing high/low (medium importance)
          if swing_high and (previous_price < swing_high and current_price >= swing_high):
              events.append({
                  "type": "level_cross",
                  "importance": "medium",
                  "details": {
                      "level_type": "swing_high",
                      "level_value": swing_high,
                      "direction": "up"
                  }
              })
          elif swing_low and (previous_price > swing_low and current_price <= swing_low):
              events.append({
                  "type": "level_cross",
                  "importance": "medium",
                  "details": {
                      "level_type": "swing_low",
                      "level_value": swing_low,
                      "direction": "down"
                  }
              })
              
          # Detect crossing of daily open (added as per discussion)
          if (previous_price < daily_open and current_price >= daily_open) or (previous_price > daily_open and current_price <= daily_open):
              events.append({
                  "type": "level_cross",
                  "importance": "medium",
                  "details": {
                      "level_type": "daily_open",
                      "level_value": daily_open,
                      "direction": "up" if current_price > daily_open else "down"
                  }
              })
          
      # Detect trend reversals
      if hasattr(self.state_tracker, "price_history") and len(self.state_tracker.price_history) >= 5:
          recent_prices = self.state_tracker.price_history[-5:]
          price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
          
          # Check for reversal patterns
          if len(price_changes) >= 3:
              # Bullish reversal: previous bearish movement followed by recent bullish movement
              if all(pc < 0 for pc in price_changes[:-2]) and all(pc > 0 for pc in price_changes[-2:]):
                  events.append({
                      "type": "reversal",
                      "importance": "high",
                      "details": {
                          "direction": "bullish",
                          "from_price": recent_prices[-3],
                          "to_price": current_price
                      }
                  })
              # Bearish reversal: previous bullish movement followed by recent bearish movement
              elif all(pc > 0 for pc in price_changes[:-2]) and all(pc < 0 for pc in price_changes[-2:]):
                  events.append({
                      "type": "reversal",
                      "importance": "high",
                      "details": {
                          "direction": "bearish",
                          "from_price": recent_prices[-3],
                          "to_price": current_price
                      }
                  })
          
      # Check for new daily high/low
      current_price = market_data["price"]["current"]
      daily_high = market_data["key_levels"].get("daily_high", float('inf'))
      daily_low = market_data["key_levels"].get("daily_low", float('-inf'))

      if current_price > daily_high and daily_high > 0:
          events.append({
              "type": "new_daily_high",
              "importance": "high",
              "details": {
                  "value": current_price,
                  "previous_high": daily_high
              }
          })
      elif current_price < daily_low and daily_low > 0:
          events.append({
              "type": "new_daily_low",
              "importance": "high",
              "details": {
                  "value": current_price,
                  "previous_low": daily_low
              }
          })
          
      # Update last price
      self.last_price = market_data["price"]["current"]
      
      return events

  def _get_level_value(self, market_data, level_name):
      """Get the value of a key level by name"""
      if level_name in market_data["key_levels"]:
          return market_data["key_levels"][level_name]
      return None
  
  def _needs_immediate_update(self, quick_data):
    """Determine if quick data needs immediate commentary"""
    # If this is the first quick data, we don't have a reference point
    if not self.last_quick_data:
        self.last_quick_data = quick_data
        return False
    
    current_price = quick_data["price"]["current"]
    last_price = self.last_quick_data["price"]["current"]
    price_change = abs(current_price - last_price)
    
    # Only trigger for truly exceptional moves (50+ points)
    if price_change > 50:
        print(f"Exceptional price change detected in quick update: {price_change:.2f} points")
        return True
    
    # Only trigger for new daily extremes
    daily_high = quick_data["key_levels"].get("daily_high", float('-inf'))
    daily_low = quick_data["key_levels"].get("daily_low", float('inf'))
    
    if current_price > daily_high + 5:  # New high with buffer
        print(f"New daily high: {current_price:.2f} vs previous {daily_high:.2f}")
        return True
    
    if current_price < daily_low - 5:  # New low with buffer
        print(f"New daily low: {current_price:.2f} vs previous {daily_low:.2f}")
        return True
    
    # Only trigger for key level breakouts, not just approaches
    for level_name, level_value in quick_data["key_levels"].items():
        # Skip non-numeric values
        if not isinstance(level_value, (int, float)):
            continue
        
        # Only check important levels
        if level_name not in ["daily_open", "vwap", "prior_day_high", "prior_day_low", "r1", "s1"]:
            continue
        
        # Check for actual breakout, not just crossing
        if (last_price < level_value - 5 and current_price > level_value + 5) or \
           (last_price > level_value + 5 and current_price < level_value - 5):
            print(f"Significant breakout of key level {level_name} at {level_value:.2f}")
            return True
    
    # Update last_quick_data for next comparison
    self.last_quick_data = quick_data
    return False

  def _merge_quick_data(self, quick_data):
    """Merge quick update data with last full market data"""
    if not self.last_full_market_data:
        # If we don't have full market data yet, use quick data with minimal structure
        minimal_data = {
            "price": quick_data["price"],
            "key_levels": quick_data["key_levels"],
            "trend": {"direction": "unknown", "atr": 10},
            "market_state": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add volume data if missing
        if "volume" not in minimal_data:
            minimal_data["volume"] = {"buy_volume": 0, "sell_volume": 0, "volume_oscillator": 0}
            
        print(f"Created minimal data structure from quick update")
        return minimal_data
    
    # Start with a copy of the last full market data
    merged_data = self.last_full_market_data.copy()
    
    # Calculate price change - FIXED to use the quick data values
    old_price = self.last_quick_data["price"]["current"] if self.last_quick_data else merged_data["price"]["current"]
    new_price = quick_data["price"]["current"]
    price_change = new_price - old_price
    
    # Update price information
    merged_data["price"]["current"] = new_price
    merged_data["price"]["high"] = max(merged_data["price"].get("high", 0), quick_data["price"].get("high", 0))
    merged_data["price"]["low"] = min(merged_data["price"].get("low", float('inf')), quick_data["price"].get("low", float('inf')))
    merged_data["price"]["change"] = price_change
    merged_data["price"]["percent_change"] = (price_change / old_price) * 100 if old_price != 0 else 0
    
    # Improved large move detection
    atr = merged_data["trend"].get("atr", 10)
    min_points_threshold = 30  # Minimum absolute threshold
    relative_threshold = 1.5 * atr  # Relative to market volatility
    
    # Use the larger of the two thresholds
    effective_threshold = max(min_points_threshold, relative_threshold)
    
    # Flag large moves based on the effective threshold
    merged_data["price"]["is_large_move"] = abs(price_change) > effective_threshold
    
    # Add context about market volatility regime
    if "volatility" not in merged_data:
        merged_data["volatility"] = {}
        
    merged_data["volatility"]["regime"] = "high" if atr > 20 else "normal" if atr > 10 else "low"
    
    # Update key levels
    for k, v in quick_data["key_levels"].items():
        merged_data["key_levels"][k] = v
    
    # Ensure daily_open is preserved
    if "daily_open" in quick_data["key_levels"]:
        merged_data["key_levels"]["daily_open"] = quick_data["key_levels"]["daily_open"]
    
    # Update timestamp
    merged_data["timestamp"] = datetime.now().isoformat()
    
    print(f"Merged data: Price change: {price_change:.2f}, Is large move: {merged_data['price']['is_large_move']}")
    
    return merged_data

  def _should_generate_quick_commentary(self, events):
    """Determine if quick update should generate commentary"""
    # Always generate commentary for high importance events
    for event in events:
        if event["importance"] == "high":
            return True, "significant"
    
    # Generate commentary for new daily highs/lows
    for event in events:
        if event["type"] in ["new_daily_high", "new_daily_low"]:
            return True, "new_extreme"
    
    # Generate commentary for key level interactions
    for event in events:
        if event["type"] == "key_level":
            return True, "key_level"
    
    # Generate commentary for significant price changes (removed 100-point threshold)
    if "price" in self.last_quick_data and "current" in self.last_quick_data["price"]:
        last_price = self.last_quick_data["price"]["current"]
        current_price = self.last_full_market_data["price"]["current"]
        price_change = current_price - last_price
        
        if abs(price_change) > 30:  # Changed to 30 points to align with other thresholds
            print(f"Large price change detected in quick update: {price_change:.2f} points")
            return True, "large_move"
    
    return False, None
  
  def _should_generate_commentary(self, events, state_changed):
    """Determine if commentary should be generated based on refined strategy"""
    current_time = datetime.now()
    
    # Check if NQAnalysis.cs has already generated commentary
    stream_output_path = os.path.join("..", "data", "stream_output.json")
    if os.path.exists(stream_output_path):
        try:
            with open(stream_output_path) as f:
                stream_data = json.load(f)
                # Check if the commentary is recent (within last 5 seconds)
                commentary_time = datetime.fromisoformat(stream_data.get("timestamp", ""))
                if (current_time - commentary_time).total_seconds() < 5:
                    # Recent commentary exists from NQAnalysis
                    return False, None
        except (json.JSONDecodeError, ValueError):
            pass  # File might be being written to
    
    # 1. Significant Events (always comment)
    
    # If there's a breakout event, always generate commentary
    for event in events:
        if event["type"] == "breakout":
            return True, "breakout"
    
    # If there's a reversal event, always generate commentary
    for event in events:
        if event["type"] == "reversal":
            return True, "reversal"
    
    # If there's a new daily high/low, always generate commentary
    for event in events:
        if event["type"] in ["new_daily_high", "new_daily_low"]:
            return True, "new_extreme"
    
    # If there's a high importance event
    for event in events:
        if event["importance"] == "high":
            # Only if enough time has passed since last significant event
            if current_time - self.last_significant_event_time > timedelta(seconds=30):
                return True, "significant"
    
    # 2. Regular Updates (every 5 minutes if nothing significant)
    
    # If state has changed
    if state_changed:
        return True, "state_change"
    
    # Regular update interval (5 minutes)
    regular_interval = timedelta(minutes=5)
    if current_time - self.last_commentary_time > regular_interval:
        return True, "regular"
    
    # 3. Generic But Useful Comments (for quieter periods)
    
    # If it's been 2 minutes since last commentary and there's a medium importance event
    medium_update_interval = timedelta(minutes=2)
    if current_time - self.last_commentary_time > medium_update_interval:
        for event in events:
            if event["importance"] == "medium":
                return True, "medium"
    
    return False, None
  
  def start_monitoring(self, data_path, quick_update_path="../data/quick_update.json", update_interval=1.0):
    """Start continuous monitoring of market data with quick updates"""
    print(f"Starting Price Action Monitor (monitoring {data_path} and {quick_update_path})")
    print(f"Regular updates every 5 minutes")
    print(f"Quick updates will be processed immediately when significant changes occur")
    print(f"Database: {self.db_path}")
    
    # Simplified debug print function that only prints essential information
    def _debug_print_key_levels(self, data, source="unknown"):
        """Print only essential key levels for debugging"""
        if "key_levels" in data:
            if "daily_open" in data["key_levels"]:
                print(f"Daily open: {data['key_levels']['daily_open']}")
    
    self._debug_print_key_levels = _debug_print_key_levels.__get__(self)
    
    try:
        while True:
            try:
                current_time = datetime.now()
                
                # Check main data
                if os.path.exists(data_path):
                    current_modified_time = os.path.getmtime(data_path)
                    
                    if current_modified_time != self.last_modified_time:
                        self.last_modified_time = current_modified_time
                        
                        # Load market data with better error handling
                        try:
                            with open(data_path, 'r') as f:
                                file_content = f.read().strip()
                                if not file_content:
                                    print(f"Warning: Empty file at {data_path}")
                                    continue
                                market_data = json.loads(file_content)
                            
                            # Process the data
                            commentary = self.process_market_data(market_data)
                            
                            # Print commentary if generated
                            if commentary:
                                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {commentary}")
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from {data_path}: {e}")
                            time.sleep(1)  # Wait a bit before trying again
                            continue
                        except Exception as e:
                            print(f"Error reading market data: {e}")
                            time.sleep(1)
                            continue
                
                # Check quick update file
                if os.path.exists(quick_update_path):
                    quick_update_modified_time = os.path.getmtime(quick_update_path)
                    
                    if quick_update_modified_time != self.last_quick_update_time:
                        self.last_quick_update_time = quick_update_modified_time
                        
                        # Load quick update data with better error handling
                        try:
                            with open(quick_update_path, 'r') as f:
                                file_content = f.read().strip()
                                if not file_content:
                                    print(f"Warning: Empty file at {quick_update_path}")
                                    continue
                                quick_data = json.loads(file_content)
                            
                            # Check for significant events that need immediate commentary
                            if self._needs_immediate_update(quick_data):
                                # Merge quick data with last full market data
                                merged_data = self._merge_quick_data(quick_data)
                                
                                # Process the merged data
                                commentary = self.process_market_data(merged_data, is_quick_update=True)
                                
                                # Print commentary if generated
                                if commentary:
                                    print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] [QUICK] {commentary}")
                            else:
                                print(f"Quick update detected but not significant enough for commentary.")
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from {quick_update_path}: {e}")
                            time.sleep(1)  # Wait a bit before trying again
                            continue
                        except Exception as e:
                            print(f"Error reading quick update data: {e}")
                            time.sleep(1)
                            continue
                
                # Check if we need a regular update based on time
                time_since_last = current_time - self.last_commentary_time
                if time_since_last > timedelta(minutes=5) and self.last_full_market_data:
                    print(f"Checking for regular update (time since last: {time_since_last.total_seconds():.1f}s)")
                    # Force a regular update
                    commentary = self.process_market_data(self.last_full_market_data)
                    if commentary:
                        print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] [REGULAR] {commentary}")
                
                # Wait before checking again
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
                
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
    except Exception as e:
        print(f"Error running monitor: {e}")
        import traceback
        traceback.print_exc()

