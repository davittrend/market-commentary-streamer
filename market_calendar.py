import json
import os
import time
import random
from datetime import datetime

# Paths
JSON_FILE_PATH = r"D:\NQcomments\data\state.json"
TEMPLATES_FILE_PATH = r"D:\NQcomments\data\commentary_templates.json"
OUTPUT_FILE_PATH = r"D:\NQcomments\data\stream_output.json"

# Load commentary templates
def load_templates(file_path):
    """Load commentary templates from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Templates file not found: {file_path}")
    with open(file_path, "r") as file:
        return json.load(file)

# Load JSON data
def load_json(file_path):
    """Load JSON data from the file."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as file:
        return json.load(file)

# Generate market commentary
def generate_market_commentary(data, templates, last_state, state_duration):
    """Generate commentary based on the market state."""
    market_state = data.get("market_state", "neutral")
    key_levels = data.get("key_levels", {})
    indicators = data.get("indicators", {})
    triggers = data.get("triggers", {})

    # Check if the state has changed
    if market_state != last_state:
        state_duration = 0  # Reset duration
        transition_comment = f"The market has transitioned from {last_state} to {market_state}."
    else:
        state_duration += 1  # Increment duration
        transition_comment = ""

    # Select a random template for the current market state
    template = random.choice(templates["market_states"].get(market_state, ["No commentary available."]))

    # Replace placeholders with actual values
    commentary = template.format(
        nearest_support=key_levels.get("nearest_support", "N/A"),
        nearest_resistance=key_levels.get("nearest_resistance", "N/A"),
        current_day_low=key_levels.get("current_day_low", "N/A"),
        current_day_high=key_levels.get("current_day_high", "N/A"),
        breakout_direction="up" if data["price"] > key_levels.get("nearest_resistance", 0) else "down",
        next_target=key_levels.get("pivot_r2", "N/A") if data["price"] > key_levels.get("nearest_resistance", 0) else key_levels.get("pivot_s2", "N/A")
    )

    # Add follow-up commentary if the state persists
    if state_duration > 0:
        follow_up_template = random.choice(templates["market_states"].get(f"{market_state}_follow_up", [
            f"The market is still in a {market_state}."
        ]))
        follow_up_commentary = follow_up_template.format(
            duration=state_duration,
            nearest_support=key_levels.get("nearest_support", "N/A"),
            nearest_resistance=key_levels.get("nearest_resistance", "N/A")
        )
    else:
        follow_up_commentary = ""

    # Combine comments
    full_commentary = f"{transition_comment} {commentary} {follow_up_commentary}".strip()

    return full_commentary, market_state, state_duration

# Generate generic topics commentary
def generate_generic_commentary(templates):
    """Generate commentary on generic topics."""
    topic = random.choice(list(templates["generic_topics"].keys()))
    template = random.choice(templates["generic_topics"][topic])

    # Replace placeholders with actual values
    if topic == "market_news":
        commentary = template.format(
            news_headline="Interest rates remain unchanged",
            news_event="the latest Fed announcement"
        )
    elif topic == "prop_firms":
        commentary = template.format(
            prop_firm_fact="prop firms allow traders to trade with higher capital",
            prop_firm_name="FTMO or The5%ers"
        )
    else:
        commentary = "No generic commentary available."

    return commentary

# Main function
def main():
    """Main function to monitor the JSON file and generate commentary."""
    templates = load_templates(TEMPLATES_FILE_PATH)
    last_modified_time = 0
    last_state = None
    state_duration = 0

    while True:
        # Check if the file has been updated
        if os.path.exists(JSON_FILE_PATH):
            current_modified_time = os.path.getmtime(JSON_FILE_PATH)
            if current_modified_time != last_modified_time:
                last_modified_time = current_modified_time

                # Load and process the JSON data
                data = load_json(JSON_FILE_PATH)
                if data:
                    # Generate market commentary
                    market_commentary, last_state, state_duration = generate_market_commentary(
                        data, templates, last_state, state_duration
                    )
                    print(f"[{data['timestamp']}] Market Commentary: {market_commentary}")

                    # Generate generic topics commentary
                    generic_commentary = generate_generic_commentary(templates)
                    print(f"[{data['timestamp']}] Generic Commentary: {generic_commentary}")

                    # Save output to a file
                    with open(OUTPUT_FILE_PATH, "a") as output_file:
                        output_file.write(f"[{data['timestamp']}] Market Commentary: {market_commentary}\n")
                        output_file.write(f"[{data['timestamp']}] Generic Commentary: {generic_commentary}\n")

        # Wait for 5 seconds before checking again
        time.sleep(5)

if __name__ == "__main__":
    main()