import os
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
import sys
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from continuous_monitor import PriceActionMonitor

class CommentaryGenerator:
  def __init__(self):
      # Load commentary templates
      templates_path = os.path.join(os.path.dirname(__file__), "..", "data", "commentary_templates.json")
      try:
          with open(templates_path) as f:
              self.templates = json.load(f)
      except FileNotFoundError:
          self.templates = {"market_states": {}}
          print(f"Warning: Could not load commentary templates from {templates_path}")
      
      self.last_commentary_type = None
      self.last_market_state = None
      self.last_key_level_interaction = None
      self.last_significant_event = None
      self.last_commentary_time = datetime.now()
      self.commentary_history = []

  def generate_new_state_commentary(self, market_data: Dict[str, Any], state_tracker: Any) -> Tuple[str, int]:
      """Generate comprehensive market commentary."""
      trend_analysis = self._get_trend_analysis(market_data, state_tracker)
      
      # Determine the primary focus based on market data
      primary_focus = self._determine_primary_focus(market_data, state_tracker)
      current_state = state_tracker.current_state
      
      commentary_parts = []
      template_id = 1  # Default template ID
      
      # Check if we should add context/follow-up from previous commentary
      follow_up = self._generate_follow_up_context(market_data, state_tracker, current_state)
      if follow_up:
          commentary_parts.append(follow_up)
      
      # Generate commentary based on primary focus
      if primary_focus == "key_level":
          commentary_parts.append(self._generate_key_level_commentary(market_data, trend_analysis))
      elif primary_focus == "significant_move":
          commentary_parts.append(self._generate_significant_move_commentary(market_data, trend_analysis))
      elif primary_focus == "pattern":
          commentary_parts.append(self._generate_pattern_commentary(market_data, trend_analysis, state_tracker))
      elif primary_focus == "trend_change":
          commentary_parts.append(self._generate_trend_change_commentary(market_data, state_tracker))
      else:
          # Regular update - include market state
          commentary_parts.append(self._generate_trend_commentary(market_data, trend_analysis))
      
      # Add secondary context
      if primary_focus != "key_level":
          level_commentary = self._generate_level_commentary(market_data, trend_analysis)
          if level_commentary:
              commentary_parts.append(level_commentary)
      
      if primary_focus != "momentum":
          momentum_commentary = self._generate_momentum_commentary(trend_analysis)
          if momentum_commentary:
              commentary_parts.append(momentum_commentary)
      
      # Add indicator commentary
      indicator_commentary = self._generate_indicator_commentary(trend_analysis)
      if indicator_commentary:
          commentary_parts.append(indicator_commentary)
      
      # Add swing point analysis occasionally
      if random.random() < 0.3:
          swing_commentary = self._generate_swing_commentary(trend_analysis)
          if swing_commentary:
              commentary_parts.append(swing_commentary)
      
      # Combine all parts
      full_commentary = " ".join(commentary_parts)
      
      # Store in history
      self.commentary_history.append({
          "timestamp": datetime.now(),
          "commentary": full_commentary,
          "primary_focus": primary_focus,
          "market_state": current_state
      })
      
      # Update last state variables
      self.last_commentary_type = primary_focus
      self.last_market_state = current_state
      
      # If this was a key level interaction, store it
      if primary_focus == "key_level":
          self.last_key_level_interaction = {
              "timestamp": datetime.now(),
              "level_type": trend_analysis["key_levels"]["recent_interactions"][0]["level_type"] if trend_analysis["key_levels"]["recent_interactions"] else None,
              "level_value": trend_analysis["key_levels"]["recent_interactions"][0]["level_value"] if trend_analysis["key_levels"]["recent_interactions"] else None,
              "interaction_type": trend_analysis["key_levels"]["recent_interactions"][0]["interaction_type"] if trend_analysis["key_levels"]["recent_interactions"] else None
          }
      
      # If this was a significant move, store it
      if primary_focus == "significant_move":
          self.last_significant_event = {
              "timestamp": datetime.now(),
              "type": "significant_move",
              "direction": "up" if market_data["price"]["change"] > 0 else "down",
              "magnitude": abs(market_data["price"]["change"])
          }
      
      self.last_commentary_time = datetime.now()
      
      # Limit history size
      if len(self.commentary_history) > 50:
          self.commentary_history.pop(0)
      
      return full_commentary, template_id

  def _generate_indicator_commentary(self, trend_analysis: Dict[str, Any]) -> str:
      """Generate commentary based on technical indicators."""
      if "indicators" not in trend_analysis:
          return ""
          
      indicators = trend_analysis["indicators"]
      comments = []
      
      # RSI commentary
      if indicators.get("rsi") is not None:
          rsi = indicators["rsi"]
          if rsi > 70:
              comments.append("RSI is in overbought territory.")
          elif rsi < 30:
              comments.append("RSI is in oversold territory.")
      
      # MACD commentary
      if indicators.get("macd") is not None:
          macd = indicators["macd"]
          if isinstance(macd, dict) and "signal" in macd and "histogram" in macd:
              if macd["histogram"] > 0 and macd["histogram"] > macd.get("prev_histogram", 0):
                  comments.append("MACD histogram is increasing in positive territory.")
              elif macd["histogram"] < 0 and macd["histogram"] < macd.get("prev_histogram", 0):
                  comments.append("MACD histogram is decreasing in negative territory.")
      
      # Stochastic commentary
      if indicators.get("stoch") is not None:
          stoch = indicators["stoch"]
          if isinstance(stoch, dict) and "k" in stoch and "d" in stoch:
              if stoch["k"] > 80 and stoch["d"] > 80:
                  comments.append("Stochastic oscillator is in overbought territory.")
              elif stoch["k"] < 20 and stoch["d"] < 20:
                  comments.append("Stochastic oscillator is in oversold territory.")
      
      return " ".join(comments)

  def _generate_follow_up_context(self, market_data: Dict[str, Any], state_tracker: Any, current_state: str) -> str:
      """Generate follow-up context based on previous commentary and current state."""
      # If this is the first commentary, no follow-up needed
      if not self.last_market_state or not self.commentary_history:
          return ""
      
      # If it's been too long since the last commentary, don't provide follow-up
      time_since_last = datetime.now() - self.last_commentary_time
      if time_since_last > timedelta(minutes=15):
          return ""
      
      # Check for trend change
      if self.last_market_state != current_state:
          templates = self.templates.get("context_follow_up", {}).get("trend_change", [])
          if templates:
              template = random.choice(templates)
              return template.replace("{previous_trend}", self.last_market_state).replace("{current_trend}", current_state).replace("{key_level}", self._get_key_level_to_watch(market_data))
      
      # Check for approaching a key level after a significant event
      if self.last_significant_event and time_since_last < timedelta(minutes=10):
          # Find the closest key level
          closest_level = self._find_closest_key_level(market_data)
          if closest_level:
              templates = self.templates.get("context_follow_up", {}).get("level_approach", [])
              if templates:
                  template = random.choice(templates)
                  recent_event = f"{self.last_significant_event['direction']} move"
                  return template.replace("{recent_event}", recent_event).replace("{next_level}", f"{closest_level[0]} at {closest_level[1]:.2f}")
      
      # Default: trend continuation
      if current_state in ["uptrend", "downtrend", "rally", "selloff"]:
          templates = self.templates.get("context_follow_up", {}).get("trend_continuation", [])
          if templates:
              template = random.choice(templates)
              trend_duration = self._get_trend_duration(state_tracker)
              trend_direction = "bullish" if current_state in ["uptrend", "rally"] else "bearish"
              next_level = self._get_next_level_in_trend_direction(market_data, trend_direction)
              return template.replace("{trend_direction}", trend_direction).replace("{trend_duration}", trend_duration).replace("{key_level}", self._get_key_level_to_watch(market_data)).replace("{next_level}", next_level)
      
      return ""

  def _get_trend_analysis(self, market_data: Dict[str, Any], state_tracker: Any) -> Dict[str, Any]:
      """Get trend analysis data."""
      # Ensure volume data is present
      volume_data = market_data.get("volume", {})
      buy_volume = volume_data.get("buy_volume", 0.0)
      sell_volume = volume_data.get("sell_volume", 0.0)
      volume_oscillator = volume_data.get("volume_oscillator", 0.0)
      
      # Ensure trend data is present
      trend_data = market_data.get("trend", {})
      atr = trend_data.get("atr", 0.0)
      adx = trend_data.get("adx", 0.0)
      
      # Ensure key levels data is present
      key_levels = market_data.get("key_levels", {})
      daily_low = key_levels.get("daily_low", 0.0)
      daily_high = key_levels.get("daily_high", 0.0)
      
      # Get momentum analysis from state tracker if available
      momentum_analysis = state_tracker.get_momentum_analysis() if hasattr(state_tracker, "get_momentum_analysis") else {}
      
      # Find closest support and resistance
      current_price = market_data["price"]["current"]
      closest_support = None
      closest_support_distance = float('inf')
      closest_resistance = None
      closest_resistance_distance = float('inf')
      
      for k, v in key_levels.items():
          if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
              continue
          
          if v < current_price:
              distance = current_price - v
              if distance < closest_support_distance:
                  closest_support_distance = distance
                  closest_support = v
          elif v > current_price:
              distance = v - current_price
              if distance < closest_resistance_distance:
                  closest_resistance_distance = distance
                  closest_resistance = v
      
      # Get indicator data if available
      indicators = {}
      if "indicators" in market_data:
          indicators = market_data["indicators"]
      else:
          # Try to extract from other parts of market_data
          rsi_value = market_data.get("rsi", None)
          macd_value = market_data.get("macd", None)
          stoch_value = market_data.get("stoch", None)
          
          if rsi_value is not None or macd_value is not None or stoch_value is not None:
              indicators = {
                  "rsi": rsi_value,
                  "macd": macd_value,
                  "stoch": stoch_value
              }

      return {
          "metrics": {
              "strength": adx,
              "consistency": random.randint(0, 100),  # Replace with actual calculation
              "momentum": momentum_analysis.get("strength", random.randint(-100, 100)) * (1 if momentum_analysis.get("direction") == "up" else -1 if momentum_analysis.get("direction") == "down" else 0),
          },
          "key_levels": {
              "recent_interactions": self._get_key_level_interactions(market_data, state_tracker),
              "closest_support": closest_support,
              "closest_resistance": closest_resistance,
              "daily_high": daily_high,
              "daily_low": daily_low
          },
          "volatility": {
              "trend": "high" if atr > 20 else "low",
              "atr": atr,
          },
          "swing_points": {
              "recent_high": {
                  "price": market_data["price"]["high"],
                  "timestamp": datetime.now(),
              },
              "recent_low": {
                  "price": daily_low,
                  "timestamp": datetime.now(),
              },
          },
          "momentum": momentum_analysis,
          "distance_ema": trend_data.get("distance_from_ema", 0),
          "indicators": indicators
      }

  def _get_trend_duration(self, state_tracker: Any) -> str:
      """Get the duration of the current trend in a human-readable format."""
      if not hasattr(state_tracker, "state_duration") or not state_tracker.state_duration:
          return "recently"
      
      seconds = state_tracker.state_duration.total_seconds()
      if seconds < 60:
          return f"{int(seconds)} seconds"
      elif seconds < 3600:
          return f"{int(seconds / 60)} minutes"
      else:
          return f"{int(seconds / 3600)} hours"

  def _get_key_level_to_watch(self, market_data: Dict[str, Any]) -> str:
      """Get the most important key level to watch based on current price."""
      current_price = market_data["price"]["current"]
      
      # Find nearest support and resistance
      levels = []
      for k, v in market_data["key_levels"].items():
          if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
              continue
          
          distance = abs(current_price - v)
          level_type = "support" if v < current_price else "resistance"
          levels.append((k, v, distance, level_type))
      
      if not levels:
          return "key levels"
      
      # Sort by distance
      levels.sort(key=lambda x: x[2])
      closest = levels[0]
      
      return f"{closest[0]} {closest[3]} at {closest[1]:.2f}"

  def _find_closest_key_level(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
      """Find the closest key level to the current price."""
      current_price = market_data["price"]["current"]
      
      closest_level = None
      closest_distance = float('inf')
      
      for k, v in market_data["key_levels"].items():
          if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
              continue
          
          distance = abs(current_price - v)
          if distance < closest_distance:
              closest_distance = distance
              closest_level = (k, v)
      
      return closest_level

  def _get_next_level_in_trend_direction(self, market_data: Dict[str, Any], trend_direction: str) -> str:
      """Get the next key level in the trend direction."""
      current_price = market_data["price"]["current"]
      
      if trend_direction == "bullish":
          # Find the closest resistance above current price
          closest_resistance = None
          closest_distance = float('inf')
          
          for k, v in market_data["key_levels"].items():
              if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
                  continue
              
              if v > current_price:
                  distance = v - current_price
                  if distance < closest_distance:
                      closest_distance = distance
                      closest_resistance = (k, v)
          
          if closest_resistance:
              return f"{closest_resistance[0]} at {closest_resistance[1]:.2f}"
      else:
          # Find the closest support below current price
          closest_support = None
          closest_distance = float('inf')
          
          for k, v in market_data["key_levels"].items():
              if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
                  continue
              
              if v < current_price:
                  distance = current_price - v
                  if distance < closest_distance:
                      closest_distance = distance
                      closest_support = (k, v)
          
          if closest_support:
              return f"{closest_support[0]} at {closest_support[1]:.2f}"
      
      return "next key level"

  def _determine_primary_focus(self, market_data: Dict[str, Any], state_tracker: Any) -> str:
      """Determine the primary focus for commentary"""
      # Check for key level interactions
      if market_data["key_levels"]["interaction"] != "none":
          return "key_level"
      
      # Check for VWAP crossing
      if self._detect_vwap_cross(market_data, state_tracker):
          return "vwap_cross"
      
      # Check for significant price moves
      if market_data["price"].get("is_large_move", False) or abs(market_data["price"]["change"]) > market_data["trend"]["atr"]:
          return "significant_move"
      
      # Check for pattern completion
      pattern_analysis = state_tracker.get_pattern_analysis() if hasattr(state_tracker, "get_pattern_analysis") else None
      if pattern_analysis and pattern_analysis.get("pattern_detected", False):
          return "pattern"
      
      # Check for trend change
      if state_tracker.current_state != state_tracker.previous_state and state_tracker.previous_state is not None:
          return "trend_change"
      
      # Default to market state
      return "market_state"

  def _detect_vwap_cross(self, market_data: Dict[str, Any], state_tracker: Any) -> bool:
      """Detect if price has crossed the VWAP."""
      if not hasattr(state_tracker, "price_history") or len(state_tracker.price_history) < 2:
          return False
      
      current_price = market_data["price"]["current"]
      previous_price = state_tracker.price_history[-2] if len(state_tracker.price_history) > 1 else current_price
      vwap = market_data["key_levels"].get("vwap", None)
      
      if vwap is None:
          return False
      
      # Check for cross
      if (previous_price < vwap and current_price >= vwap) or (previous_price > vwap and current_price <= vwap):
          return True
      
      return False

  def _get_key_level_interactions(self, market_data: Dict[str, Any], state_tracker: Any) -> List[Dict[str, Any]]:
      """Get recent key level interactions"""
      interactions = []
      
      # Check if state_tracker has key_level_interactions
      if hasattr(state_tracker, "key_level_interactions") and state_tracker.key_level_interactions:
          # Use the most recent interactions from state_tracker
          for interaction in state_tracker.key_level_interactions[-3:]:
              interactions.append({
                  "interaction_type": interaction["type"],
                  "price": interaction["price"],
                  "level_value": interaction["level_value"],
                  "level_type": interaction["level_name"],
              })
      else:
          # Fallback to market_data
          if market_data["key_levels"]["interaction"] != "none":
              level_type = None
              level_value = None
              
              if market_data["key_levels"]["breakout_level"]:
                  level_type = market_data["key_levels"]["breakout_level"]
                  interaction_type = "breakout"
              elif market_data["key_levels"]["rejection_level"]:
                  level_type = market_data["key_levels"]["rejection_level"]
                  interaction_type = "rejection"
              
              if level_type:
                  # Find the corresponding value
                  for k, v in market_data["key_levels"].items():
                      if k == level_type:
                          level_value = v
                          break
                  
                  if level_value:
                      interactions.append({
                          "interaction_type": interaction_type,
                          "price": market_data["price"]["current"],
                          "level_value": level_value,
                          "level_type": level_type,
                      })
          
          # Check for VWAP cross
          if self._detect_vwap_cross(market_data, state_tracker):
              vwap_value = market_data["key_levels"].get("vwap", 0.0)
              interactions.append({
                  "interaction_type": "vwap_cross",
                  "price": market_data["price"]["current"],
                  "level_value": vwap_value,
                  "level_type": "VWAP"
              })
      
      return interactions

  def _generate_trend_commentary(self, market_data: Dict[str, Any], trend_analysis: Dict[str, Any]) -> str:
      """Generate commentary about the current trend."""
      market_state = market_data.get("market_state", market_data.get("trend", {}).get("direction", "sideways"))
      
      if market_state not in self.templates.get("market_states", {}):
          market_state = "sideways"  # Default fallback
      
      templates = self.templates.get("market_states", {}).get(market_state, ["Market conditions are neutral."])
      template = random.choice(templates)
      
      # Format with key levels if needed
      if "{close_support}" in template or "{close_resistance}" in template:
          close_support = trend_analysis["key_levels"]["closest_support"] or market_data["key_levels"].get("s1", 0)
          close_resistance = trend_analysis["key_levels"]["closest_resistance"] or market_data["key_levels"].get("r1", 0)
          template = template.replace("{close_support}", f"{close_support:.2f}")
          template = template.replace("{close_resistance}", f"{close_resistance:.2f}")
      
      # Format with distance from EMA if needed
      if "{distance_ema}" in template:
          distance_ema = abs(trend_analysis.get("distance_ema", 0))
          template = template.replace("{distance_ema}", f"{distance_ema:.2f}")
      
      # Format with points and percent if needed
      if "{points}" in template or "{percent}" in template:
          points = abs(market_data["price"]["change"])
          percent = abs(market_data["price"]["percent_change"])
          template = template.replace("{points}", f"{points:.2f}")
          template = template.replace("{percent}", f"{percent:.2f}")
      
      return template

  def _generate_key_level_commentary(self, market_data: Dict[str, Any], trend_analysis: Dict[str, Any]) -> str:
      """Generate commentary about key level interactions."""
      interactions = trend_analysis["key_levels"]["recent_interactions"]
      
      if not interactions:
          return self._generate_trend_commentary(market_data, trend_analysis)
      
      # Use the most recent interaction
      interaction = interactions[0]
      interaction_type = interaction["interaction_type"]
      level_type = interaction["level_type"]
      level_value = interaction["level_value"]
      
      if interaction_type not in self.templates.get("key_level_interactions", {}):
          interaction_type = "test"  # Default fallback
      
      templates = self.templates.get("key_level_interactions", {}).get(interaction_type, ["Price is interacting with a key level."])
      template = random.choice(templates)
      
      # Format template
      direction = "above" if market_data["price"]["current"] > level_value else "below"
      level_function = "resistance" if market_data["price"]["current"] < level_value else "support"
      
      # Find next level in the appropriate direction
      next_level = None
      if interaction_type == "breakout":
          if direction == "above":
              # Find next resistance
              for k, v in market_data["key_levels"].items():
                  if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
                      continue
                  if v > level_value and (next_level is None or v < next_level[1]):
                      next_level = (k, v)
          else:
              # Find next support
              for k, v in market_data["key_levels"].items():
                  if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
                      continue
                  if v < level_value and (next_level is None or v > next_level[1]):
                      next_level = (k, v)
      
      # Find next support for rejection
      next_support = None
      if interaction_type == "rejection":
          for k, v in market_data["key_levels"].items():
              if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
                  continue
              if v < level_value and (next_support is None or v > next_support[1]):
                  next_support = (k, v)
      
      # Replace template placeholders
      template = template.replace("{level_type}", level_type)
      template = template.replace("{level_value}", f"{level_value:.2f}")
      template = template.replace("{direction}", direction)
      template = template.replace("{level_function}", level_function)
      
      if "{next_level}" in template and next_level:
          template = template.replace("{next_level}", f"{next_level[0]} at {next_level[1]:.2f}")
      elif "{next_level}" in template:
          template = template.replace("{next_level}", "the next key level")
      
      if "{next_support}" in template and next_support:
          template = template.replace("{next_support}", f"{next_support[0]} at {next_support[1]:.2f}")
      elif "{next_support}" in template:
          template = template.replace("{next_support}", "the next support level")
      
      return template

  def _generate_level_commentary(self, market_data: Dict[str, Any], trend_analysis: Dict[str, Any]) -> str:
      """Generate commentary about nearby key levels."""
      current_price = market_data["price"]["current"]
      closest_support = trend_analysis["key_levels"]["closest_support"]
      closest_resistance = trend_analysis["key_levels"]["closest_resistance"]
      
      if closest_support and closest_resistance:
          support_distance = current_price - closest_support
          resistance_distance = closest_resistance - current_price
          
          # If we're very close to either level, mention it
          if support_distance < market_data["trend"]["atr"] * 0.3 or resistance_distance < market_data["trend"]["atr"] * 0.3:
              if support_distance < resistance_distance:
                  return f"Price is very close to support at {closest_support:.2f}, just {support_distance:.2f} points away."
              else:
                  return f"Price is very close to resistance at {closest_resistance:.2f}, just {resistance_distance:.2f} points away."
          
          # Otherwise mention both
          return f"Trading between support at {closest_support:.2f} and resistance at {closest_resistance:.2f}."
      elif closest_support:
          return f"Next support at {closest_support:.2f}, {(current_price - closest_support):.2f} points below current price."
      elif closest_resistance:
          return f"Next resistance at {closest_resistance:.2f}, {(closest_resistance - current_price):.2f} points above current price."
      
      return ""

  def _generate_momentum_commentary(self, trend_analysis: Dict[str, Any]) -> str:
      """Generate commentary about momentum."""
      momentum = trend_analysis.get("momentum", {})
      
      if not momentum:
          return ""
      
      direction = momentum.get("direction")
      strength = momentum.get("strength", 0.0)
      duration = momentum.get("duration", 0)
      exhaustion = momentum.get("exhaustion", False)
      
      if not direction or direction == "flat":
          return ""
      
      if exhaustion:
          momentum_type = "decreasing"
      elif strength > 0.7:
          momentum_type = "increasing"
      else:
          return ""
      
      if momentum_type not in self.templates.get("momentum", {}):
          return ""
      
      templates = self.templates.get("momentum", {}).get(momentum_type, [])
      if not templates:
          return ""
      
      template = random.choice(templates)
      template = template.replace("{direction}", direction)
      
      if "{adx_value}" in template:
          template = template.replace("{adx_value}", f"{trend_analysis['metrics']['strength']:.1f}")
      
      return template

  def _generate_swing_commentary(self, trend_analysis: Dict[str, Any]) -> str:
      """Generate commentary about swing points."""
      swing_high = trend_analysis["swing_points"]["recent_high"]
      swing_low = trend_analysis["swing_points"]["recent_low"]
      
      if random.random() < 0.5:
          return f"Recent swing high at {swing_high['price']:.2f}."
      else:
          return f"Recent swing low at {swing_low['price']:.2f}."

  def _generate_significant_move_commentary(self, market_data: Dict[str, Any], trend_analysis: Dict[str, Any]) -> str:
      """Generate commentary about significant price moves."""
      price_change = market_data["price"]["change"]
      percent_change = market_data["price"]["percent_change"]
      
      direction = "up" if price_change > 0 else "down"
      
      if direction not in self.templates.get("significant_moves", {}):
          # Fallback to generic commentary
          magnitude = "large" if abs(price_change) > market_data["trend"]["atr"] * 2 else "significant"
          return f"{magnitude.capitalize()} move {direction}: Price has moved {abs(price_change):.2f} points ({abs(percent_change):.2f}%) in a single bar."
      
      templates = self.templates.get("significant_moves", {}).get(direction, [])
      if not templates:
          # Fallback to generic commentary
          magnitude = "large" if abs(price_change) > market_data["trend"]["atr"] * 2 else "significant"
          return f"{magnitude.capitalize()} move {direction}: Price has moved {abs(price_change):.2f} points ({abs(percent_change):.2f}%) in a single bar."
      
      template = random.choice(templates)
      
      # Format template
      template = template.replace("{points}", f"{abs(price_change):.2f}")
      template = template.replace("{percent}", f"{abs(percent_change):.2f}")
      
      # Add volume description
      volume_description = "increasing"
      if "volume" in market_data:
          vol_osc = market_data["volume"].get("volume_oscillator", 0)
          if vol_osc > 200:
              volume_description = "surging significantly"
          elif vol_osc > 100:
              volume_description = "increasing strongly"
          elif vol_osc > 0:
              volume_description = "increasing"
          elif vol_osc > -100:
              volume_description = "decreasing"
          else:
              volume_description = "dropping significantly"
      
      template = template.replace("{volume_description}", volume_description)
      
      # Add next support/resistance
      if direction == "up":
          next_resistance = None
          for k, v in market_data["key_levels"].items():
              if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
                  continue
              if v > market_data["price"]["current"] and (next_resistance is None or v < next_resistance[1]):
                  next_resistance = (k, v)
          
          if next_resistance:
              template = template.replace("{next_resistance}", f"{next_resistance[0]} at {next_resistance[1]:.2f}")
          else:
              template = template.replace("{next_resistance}", "the next resistance level")
      else:
          next_support = None
          for k, v in market_data["key_levels"].items():
              if not isinstance(v, (int, float)) or k in ["interaction", "breakout_level", "rejection_level"]:
                  continue
              if v < market_data["price"]["current"] and (next_support is None or v > next_support[1]):
                  next_support = (k, v)
          
          if next_support:
              template = template.replace("{next_support}", f"{next_support[0]} at {next_support[1]:.2f}")
          else:
              template = template.replace("{next_support}", "the next support level")
      
      return template

  def _generate_pattern_commentary(self, market_data: Dict[str, Any], trend_analysis: Dict[str, Any], state_tracker: Any) -> str:
      """Generate commentary about pattern completions."""
      pattern_analysis = state_tracker.get_pattern_analysis() if hasattr(state_tracker, "get_pattern_analysis") else None
      
      if not pattern_analysis or not pattern_analysis.get("pattern_detected", False):
          return self._generate_trend_commentary(market_data, trend_analysis)
      
      pattern_type = pattern_analysis["pattern_type"]
      
      if pattern_type not in self.templates.get("patterns", {}):
          return self._generate_trend_commentary(market_data, trend_analysis)
      
      templates = self.templates.get("patterns", {}).get(pattern_type, [])
      if not templates:
          return self._generate_trend_commentary(market_data, trend_analysis)
      
      template = random.choice(templates)
      
      # Format with price level if needed
      if "{price_level}" in template:
          price_level = pattern_analysis["details"].get("price_levels", [market_data["price"]["current"]])[0]
          template = template.replace("{price_level}", f"{price_level:.2f}")
      
      # Format with neckline if needed
      if "{neckline}" in template:
          # For double top/bottom, neckline is typically the middle point between the two highs/lows
          if pattern_type in ["double_top", "double_bottom"] and "price_levels" in pattern_analysis["details"]:
              levels = pattern_analysis["details"]["price_levels"]
              if len(levels) >= 2:
                  if pattern_type == "double_top":
                      # For double top, neckline is the low between the two highs
                      neckline = min(state_tracker.price_history[-5:]) if hasattr(state_tracker, "price_history") else market_data["price"]["low"]
                  else:
                      # For double bottom, neckline is the high between the two lows
                      neckline = max(state_tracker.price_history[-5:]) if hasattr(state_tracker, "price_history") else market_data["price"]["high"]
                  
                  template = template.replace("{neckline}", f"{neckline:.2f}")
              else:
                  template = template.replace("{neckline}", "the pattern neckline")
          else:
              template = template.replace("{neckline}", "the pattern neckline")
      
      # Format with head level if needed
      if "{head_level}" in template and pattern_type == "head_shoulders":
          # For head and shoulders, head level is typically the highest point
          head_level = max(state_tracker.price_history[-7:]) if hasattr(state_tracker, "price_history") else market_data["price"]["high"]
          template = template.replace("{head_level}", f"{head_level:.2f}")
      
      # Format with target if needed
      if "{target}" in template:
          # Simple target calculation
          if pattern_type == "head_shoulders" and "{neckline}" in template and "{head_level}" in template:
              neckline = float(template.split("{neckline}")[1].split()[0])
              head_level = float(template.split("{head_level}")[1].split()[0])
              target = neckline - (head_level - neckline)
              template = template.replace("{target}", f"{target:.2f}")
          else:
              template = template.replace("{target}", "the pattern target")
      
      return template

  def _generate_trend_change_commentary(self, market_data: Dict[str, Any], state_tracker: Any) -> str:
      """Generate commentary about trend changes."""
      new_trend = state_tracker.current_state
      previous_trend = state_tracker.previous_state
      
      if not previous_trend:
          return self._generate_trend_commentary(market_data, self._get_trend_analysis(market_data, state_tracker))
      
      if new_trend == "uptrend" and previous_trend != "uptrend":
          return f"Trend change detected: Price action has shifted to bullish. Previous {previous_trend} has ended after {state_tracker.state_duration.total_seconds() // 60:.0f} minutes."
      elif new_trend == "downtrend" and previous_trend != "downtrend":
          return f"Trend change detected: Price action has shifted to bearish. Previous {previous_trend} has ended after {state_tracker.state_duration.total_seconds() // 60:.0f} minutes."
      elif new_trend == "sideways" and previous_trend != "sideways":
          return f"Trend change detected: Price action has shifted to sideways consolidation. Previous {previous_trend} has ended after {state_tracker.state_duration.total_seconds() // 60:.0f} minutes."
      
      return self._generate_trend_commentary(market_data, self._get_trend_analysis(market_data, state_tracker))

def main():
    print(f"Starting Market Commentary System at {datetime.now()}")

    # Create and start the monitor
    monitor = PriceActionMonitor()

    # Path to the market data file
    data_path = os.path.join("..", "data", "NQAnalysisData.json")
    quick_update_path = os.path.join("..", "data", "quick_update.json")

    # Check if the data files exist
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found at {data_path}")
        print("Creating data directory if it doesn't exist...")
        os.makedirs(os.path.join("..", "data"), exist_ok=True)
        print(f"Please ensure the NQAnalysisData.json file is placed in {os.path.join('..', 'data')}")

    # Start monitoring
    try:
        print(f"Starting to monitor {data_path} and {quick_update_path}")
        print(f"Regular updates every {monitor.config.get('regular_update_interval_minutes', 5)} minutes")
        print(f"Database: {monitor.db_path}")
        
        # Use the new monitoring method
        monitor.start_monitoring(data_path, quick_update_path)
            
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
    except Exception as e:
        print(f"Error running monitor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
  main()

