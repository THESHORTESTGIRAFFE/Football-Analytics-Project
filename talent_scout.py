import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv # Make sure python-dotenv is installed
import sys
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time # For API rate limiting if needed

# --- 0. Configuration & Setup ---
# ========================================================================================
# IMPORTANT:
# 1. Create a file named `.env` in the same directory as this script.
#    Inside `.env`, add: BALLDONTLIE_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
#    Replace YOUR_ACTUAL_API_KEY_HERE with your real key.
# 2. Ensure your 'src' folder (containing client.py) is accessible.
#    The path below assumes 'src' is a sibling directory to this script.
#    Adjust 'sys.path.append' if your 'src' folder is elsewhere.
# ========================================================================================

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("BALLDONTLIE_API_KEY")

if not API_KEY:
    print("Error: BALLDONTLIE_API_KEY not found in .env file.")
    print("Please create a .env file in the same directory as this script with: BALLDONTLIE_API_KEY=\"YOUR_API_KEY\"")
    sys.exit(1)

# Add the 'src' directory to Python's path to import SportsClient
script_dir = os.path.dirname(__file__)
src_path = os.path.join(script_dir, 'src')
if os.path.exists(src_path):
    sys.path.append(src_path)
else:
    print(f"Error: 'src' directory not found at {src_path}.")
    print("Please ensure 'src' folder (containing client.py) is correctly placed or adjust sys.path.append.")
    sys.exit(1)

try:
    # Import SportsClient as defined in your client.py
    from client import SportsClient
except ImportError:
    print("Error: Could not import SportsClient from 'src/client.py'.")
    print("Ensure 'client.py' exists in the 'src' folder and the 'src' path is correctly added,")
    print("and that 'SportsClient' class is defined within it as shown previously.")
    sys.exit(1)

# --- Define Constants ---
MIN_MINUTES_PLAYED = 900 # Minimum minutes played for a player to be considered
N_TOP_PLAYERS_DISPLAY = 10 # Number of top players to display for each role
SEASON_FOR_API = "2023" # Most recent completed season for typical data availability


# --- 1. Data Acquisition (Balldontlie API) ---
# ========================================================================================
print(f"--- 1. Data Acquisition from Balldontlie API (EPL, {SEASON_FOR_API}) ---")

try:
    client = SportsClient(api_key=API_KEY)
    print("Balldontlie SportsClient initialized.")

    # Access the EPL-specific client
    epl_client = client.epl
    print("EPL client accessed.")

    # 1. Get all players for the season to get their IDs and basic info
    print(f"Fetching all players for {SEASON_FOR_API} season...")
    all_players_data_objects = [] # Store raw objects here initially
    cursor = None
    page = 0
    MAX_RETRIES = 5
    INITIAL_DELAY = 5 # seconds - Increased for more robust handling

    while True:
        page += 1
        print(f"DEBUG: Calling players.list (Page {page}) with season={SEASON_FOR_API}, cursor={cursor}")
        
        retries = 0
        success = False
        while retries < MAX_RETRIES:
            try:
                players_response = epl_client.players.list(season=SEASON_FOR_API, cursor=cursor)
                success = True
                break # Exit retry loop if successful
            except Exception as e:
                if "Too Many Requests" in str(e):
                    delay = INITIAL_DELAY * (2 ** retries) # Exponential backoff
                    print(f"DEBUG: Too Many Requests. Retrying in {delay:.2f} seconds... (Attempt {retries + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                    retries += 1
                else:
                    # If it's a different error, re-raise or handle as a fatal error
                    print(f"DEBUG: An unexpected error occurred: {e}")
                    raise # Re-raise unexpected errors
        
        if not success:
            print(f"Failed to fetch players after {MAX_RETRIES} retries. Breaking loop.")
            break

        # --- Debug prints (keeping them for verification) ---
        print(f"DEBUG (Page {page}): Type of players_response: {type(players_response)}")
        has_data_attr = hasattr(players_response, 'data')
        has_meta_attr = hasattr(players_response, 'meta')
        print(f"DEBUG (Page {page}): Has 'data' attribute? {has_data_attr}")
        print(f"DEBUG (Page {page}): Has 'meta' attribute? {has_meta_attr}")
        print(f"DEBUG (Page {page}): Content of players_response (first 500 chars): {str(players_response)[:500]}")
        # --- End Debug prints ---

        # Check if the response is valid and contains the 'data' attribute
        if not has_data_attr or players_response.data is None:
            print(f"No valid 'data' attribute or data is None in players_response for page {page}. Breaking loop.")
            print(f"DEBUG: Response causing break on page {page}: {players_response}")
            break

        current_page_data = players_response.data # Access 'data' as an attribute
        if not current_page_data: # If 'data' list is empty, it means no more players
            print(f"No players found on page {page}. Breaking loop.")
            break

        all_players_data_objects.extend(current_page_data)
        
        meta_info = players_response.meta # Access 'meta' as an attribute
        cursor = meta_info.next_cursor if hasattr(meta_info, 'next_cursor') else None

        if not cursor:
            print(f"DEBUG: No next cursor found after page {page}, breaking loop.")
            break
        
        print(f"DEBUG: Fetched {len(current_page_data)} players on page {page}. Next cursor: {cursor}")
        time.sleep(1.0) # Increased sleep for pagination


    # Convert list of custom objects (EPLPlayer) to a list of dictionaries for DataFrame
    processed_players_data = []
    for player_obj in all_players_data_objects:
        player_dict = {attr: getattr(player_obj, attr) for attr in dir(player_obj) 
                       if not attr.startswith('__') and not callable(getattr(player_obj, attr))}
        processed_players_data.append(player_dict)

    print(f"Fetched {len(processed_players_data)} player records.")
    df_players_info = pd.DataFrame(processed_players_data)

    if df_players_info.empty:
        print("No player information loaded. Exiting.")
        sys.exit(1)


    # 2. Iterate through players to get their season stats
    print("Fetching season stats for each player (this may take a while)...")
    all_player_season_stats = []
    for index, player_row in df_players_info.iterrows():
        player_id = player_row['id']
        team_id = player_row['team_ids'][0] if 'team_ids' in player_row and player_row['team_ids'] else None

        if team_id is None:
            continue

        retries = 0
        success = False
        while retries < MAX_RETRIES:
            try:
                player_stats_response = epl_client.players.get_season_stats(player_id=player_id, season=SEASON_FOR_API)
                success = True
                break
            except Exception as e:
                if "Too Many Requests" in str(e):
                    delay = INITIAL_DELAY * (2 ** retries)
                    print(f"DEBUG: Too Many Requests for player {player_id}. Retrying in {delay:.2f} seconds... (Attempt {retries + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                    retries += 1
                else:
                    print(f"Could not fetch stats for player ID {player_id} ({player_row['name']}): {e}")
                    break

        if not success:
            print(f"Failed to fetch stats for player {player_id} after {MAX_RETRIES} retries. Skipping.")
            continue

        if player_stats_response:
            stats_dict = {'id': player_id}
            for stat in player_stats_response:
                # Ensure 'name' and 'value' keys exist in stat dictionary
                if 'name' in stat and 'value' in stat:
                    stats_dict[stat['name']] = stat['value']
                else:
                    print(f"DEBUG: Malformed stat response for player {player_id}: {stat}")
            all_player_season_stats.append(stats_dict)
        time.sleep(0.2) # Increased sleep between individual player stat calls

    df_stats = pd.DataFrame(all_player_season_stats)

    df = pd.merge(df_players_info, df_stats, on='id', how='left')

    print(f"Initial DataFrame shape after merging: {df.shape}")
    print("Initial DataFrame head:\n", df.head())
    print("\nInitial DataFrame columns:\n", df.columns.tolist())

except Exception as e:
    print(f"Error during API data acquisition: {e}")
    sys.exit(1)


# --- 2. Data Preprocessing ---
# ========================================================================================
print("\n--- 2. Data Preprocessing ---")

# --- CUSTOMIZATION POINT 1: Verify Column Names from your API Response ---
# These are based on the Balldontlie EPL API documentation for 'player' and 'season_stats'
PLAYER_NAME_COL = "name" # From /players endpoint
# CLUB_NAME_COL: Team name is not directly returned with player info or season stats in this API.
# It requires fetching teams via client.epl.teams.get_all() and merging based on 'team_ids'.
# For now, we'll assign N/A or you can implement team fetching and merging if critical for display.
CLUB_NAME_COL = "team_name_placeholder" # Placeholder that will be N/A unless merged

MINUTES_PLAYED_COL = "mins_played" # Confirmed in /players/:id/season_stats

# Ensure critical columns exist
for col in [PLAYER_NAME_COL, MINUTES_PLAYED_COL]:
    if col not in df.columns:
        print(f"Error: Critical column '{col}' not found in DataFrame. Please check API response and documentation.")
        sys.exit(1)

# Handle CLUB_NAME_COL: If not present, create a placeholder.
if CLUB_NAME_COL not in df.columns:
    print(f"Warning: '{CLUB_NAME_COL}' column not found directly. Player results will not show team names. "
          "To include team names, you would need to fetch team data via client.epl.teams.get_all() and merge by 'team_ids'.")
    df[CLUB_NAME_COL] = 'N/A' # Default to N/A

# Filter out players with insufficient minutes
original_player_count = df.shape[0]
# Ensure minutes played column is numeric, coerce errors to NaN and then drop
df[MINUTES_PLAYED_COL] = pd.to_numeric(df[MINUTES_PLAYED_COL], errors='coerce')
df_filtered = df[df[MINUTES_PLAYED_COL].notna() & (df[MINUTES_PLAYED_COL] >= MIN_MINUTES_PLAYED)].copy()

print(f"Filtered players with < {MIN_MINUTES_PLAYED} minutes or missing minute data. Removed: {original_player_count - df_filtered.shape[0]} players.")
print(f"Remaining players: {df_filtered.shape[0]}")

# Handle potential NaN values in performance metrics by filling with 0
# Select only numeric columns for fillna
performance_numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
# Exclude identifier columns from filling with 0 if they can be NaN meaningfully
cols_to_exclude_from_fillna = ['id', 'age', MINUTES_PLAYED_COL]
for col in cols_to_exclude_from_fillna:
    if col in performance_numeric_cols:
        performance_numeric_cols.remove(col)

df_filtered[performance_numeric_cols] = df_filtered[performance_numeric_cols].fillna(0)
print("NaN values in performance metrics filled with 0.")

# --- CUSTOMIZATION POINT 2: Create "Per 90" Metrics and other derived stats ---
# Based on available stats from Balldontlie EPL API docs and common derivations:
METRICS_TO_CONVERT_TO_P90 = {
    "goals": "Goals_P90",
    "goal_assist": "Assists_P90",
    "total_tackle": "Tackles_P90",
    "fouls": "Fouls_P90",
    "dispossessed": "Dispossessed_P90",
    "total_scoring_att": "Shots_P90",
    "total_offside": "Offsides_P90",
    "touches": "Touches_P90",
    # Assuming these are available as stat names from the /season_stats endpoint based on 'game_stats' list in API docs.
    "interception": "Interceptions_P90",
    "accurate_pass": "Accurate_Passes_P90",
    "big_chance_created": "Big_Chance_Created_P90",
    "successful_dribbles": "Successful_Dribbles_P90",
}

for raw_col, p90_col in METRICS_TO_CONVERT_TO_P90.items():
    if raw_col in df_filtered.columns:
        df_filtered[p90_col] = df_filtered[raw_col] / (df_filtered[MINUTES_PLAYED_COL].replace(0, np.nan) / 90)
        df_filtered[p90_col] = df_filtered[p90_col].fillna(0).replace([np.inf, -np.inf], 0) # Handle potential inf from 0/0
    else:
        print(f"Warning: Raw column '{raw_col}' not found for P90 conversion. This P90 metric will be skipped.")
        df_filtered[p90_col] = 0.0 # Create column with zeros so it doesn't break later logic if used in ROLES

# Calculate percentages and rates
# These require specific raw counts to be present
if 'ontarget_att_assist' in df_filtered.columns and 'total_scoring_att' in df_filtered.columns:
    df_filtered['Shots_on_Target_Pct'] = (df_filtered['ontarget_att_assist'] / df_filtered['total_scoring_att']).fillna(0) * 100
    df_filtered['Shots_on_Target_Pct'].replace([np.inf, -np.inf], 0, inplace=True)
    print("Calculated Shots_on_Target_Pct.")
else:
    df_filtered['Shots_on_Target_Pct'] = 0.0
    print("Warning: Could not calculate Shots_on_Target_Pct. Missing 'ontarget_att_assist' or 'total_scoring_att'. Set to 0.")

if 'goals' in df_filtered.columns and 'total_scoring_att' in df_filtered.columns:
    df_filtered['Shot_Conversion_Rate'] = (df_filtered['goals'] / df_filtered['total_scoring_att']).fillna(0) * 100
    df_filtered['Shot_Conversion_Rate'].replace([np.inf, -np.inf], 0, inplace=True)
    print("Calculated Shot_Conversion_Rate.")
else:
    df_filtered['Shot_Conversion_Rate'] = 0.0
    print("Warning: Could not calculate Shot_Conversion_Rate. Missing 'goals' or 'total_scoring_att'. Set to 0.")

if 'accurate_pass' in df_filtered.columns and 'total_pass' in df_filtered.columns:
    df_filtered['Pass_Completion_Pct'] = (df_filtered['accurate_pass'] / df_filtered['total_pass']).fillna(0) * 100
    df_filtered['Pass_Completion_Pct'].replace([np.inf, -np.inf], 0, inplace=True)
    print("Calculated Pass_Completion_Pct.")
else:
    df_filtered['Pass_Completion_Pct'] = 0.0
    print("Warning: Could not calculate Pass_Completion_Pct. Missing 'accurate_pass' or 'total_pass'. Set to 0.")


# Rename the minutes column for consistency
df_filtered.rename(columns={MINUTES_PLAYED_COL: 'Minutes_Played_Standard'}, inplace=True)

print("\nDataFrame after preprocessing and P90 conversions. Head:")
print(df_filtered.head())
print("\nDataFrame columns after P90 conversions and derivations:")
print(df_filtered.columns.tolist())


# --- 3. Role Definitions & Scoring Logic ---
# ========================================================================================
print("\n--- 3. Defining Roles and Scoring Logic ---")

# --- CUSTOMIZATION POINT 3: Define ROLES with EXACT P90/Percentage Column Names ---
# Based on the available stats from the Balldontlie EPL API documentation and derivations
ROLES = {
    "Modern Holding Midfielder": {
        "metrics": [
            "Tackles_P90",
            "Interceptions_P90",
            "Pass_Completion_Pct",
            "Dispossessed_P90",
            "Fouls_P90",
            "Total_Passes_P90" # Use raw total passes for general involvement
        ],
        "inverse_metrics": [
            "Dispossessed_P90",
            "Fouls_P90"
        ],
        "description": "Breaks up play, retains possession, and dictates tempo from deep. Focuses on control and defensive solidity."
    },
    "Goal Poaching Striker": {
        "metrics": [
            "Goals_P90",
            "Shots_P90",
            "Shots_on_Target_Pct",
            "Shot_Conversion_Rate",
            "Offsides_P90",
            "Touches_P90"
        ],
        "inverse_metrics": [
            "Offsides_P90"
        ],
        "description": "A forward whose primary contribution is scoring goals, relying on intelligent movement and clinical finishing."
    },
    "Creative Attacking Midfielder": {
        "metrics": [
            "Assists_P90",
            "Big_Chance_Created_P90",
            "Successful_Dribbles_P90",
            "Total_Passes_P90"
        ],
        "inverse_metrics": [],
        "description": "Creates chances for others, often from central or wide attacking areas, with incisive passing and dribbling."
    }
}

def calculate_role_score(player_df_segment, role_name, role_info, player_col, club_col):
    metrics = role_info['metrics']
    inverse_metrics = role_info['inverse_metrics']

    available_metrics = [m for m in metrics if m in player_df_segment.columns]
    missing_metrics = [m for m in metrics if m not in player_df_segment.columns]

    if missing_metrics:
        print(f"  Warning: Missing metrics for '{role_name}': {missing_metrics}. Scoring with available metrics.")
        if not available_metrics:
            print(f"  No valid metrics available for '{role_name}'. Skipping scoring for this role.")
            return pd.DataFrame()
        metrics = available_metrics

    # Ensure all selected metrics are numeric and handle non-numeric values
    for m in metrics:
        if not pd.api.types.is_numeric_dtype(player_df_segment[m]):
            print(f"  Warning: Metric '{m}' for role '{role_name}' is not numeric. Coercing to numeric.")
            player_df_segment[m] = pd.to_numeric(player_df_segment[m], errors='coerce').fillna(0)

    # Filter out rows where all available metrics are zero, as MinMaxScaler would cause issues (constant feature)
    # And such players wouldn't be meaningful for scoring anyway.
    temp_df_for_scaling = player_df_segment[metrics].copy()
    
    # Drop rows where all metric values are 0 for the scaling process (for the selected metrics)
    # Use the original index of the player_df_segment to ensure scores are mapped correctly later
    non_zero_metrics_indices = temp_df_for_scaling[(temp_df_for_scaling != 0).any(axis=1)].index
    temp_df_for_scaling = temp_df_for_scaling.loc[non_zero_metrics_indices]

    if temp_df_for_scaling.empty:
        print(f"  No players with non-zero values for '{role_name}' metrics after filtering. Skipping scaling.")
        return pd.DataFrame() # Return empty if no meaningful data to scale

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(temp_df_for_scaling)
    scaled_df = pd.DataFrame(scaled_features, columns=metrics, index=temp_df_for_scaling.index)

    for col in inverse_metrics:
        if col in scaled_df.columns:
            scaled_df[col] = 1 - scaled_df[col]

    # Calculate the total role score by summing the scaled metrics
    # Ensure scores are applied back to the original df_filtered indices
    player_df_segment.loc[scaled_df.index, f'{role_name}_score'] = scaled_df.sum(axis=1)

    display_cols = [player_col, club_col, f'{role_name}_score'] + metrics
    # Only return players that were actually scored (i.e., had non-zero metrics)
    return player_df_segment.loc[scaled_df.index, display_cols].sort_values(by=f'{role_name}_score', ascending=False)


# --- 4. Apply Scoring & Display Results ---
# ========================================================================================
print("\n--- 4. Applying Scoring and Displaying Top Players ---")

top_players_by_role = {}

for role_name, role_info in ROLES.items():
    print(f"\nProcessing '{role_name}'...")
    ranked_players = calculate_role_score(df_filtered.copy(), role_name, role_info, PLAYER_NAME_COL, CLUB_NAME_COL)

    if not ranked_players.empty:
        top_players_by_role[role_name] = ranked_players.head(N_TOP_PLAYERS_DISPLAY)
        print(f"Top {N_TOP_PLAYERS_DISPLAY} {role_name}s:")
        print(top_players_by_role[role_name].to_string(index=False))
    else:
        print(f"No top players found or scored for '{role_name}'.")


# --- 5. (Optional) Visualization Generation ---
# ========================================================================================
print("\n--- 5. Generating Visualizations (Optional) ---")

# Example Radar Chart Function
def create_radar_chart(player_data, player_name, metrics, title, filename=None):
    categories = metrics
    values = player_data[metrics].iloc[0].tolist()

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=player_name)
    ax.fill(angles, values, 'b', alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2","0.4","0.6","0.8","1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    ax.set_title(title, va='bottom', fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)

    if filename:
        plt.savefig(f'./{filename}.png', bbox_inches='tight')
        print(f"Saved radar chart: {filename}.png")
    plt.close()

# Generate radar charts for the top player of each role
for role_name, top_players_df in top_players_by_role.items():
    if not top_players_df.empty:
        top_player = top_players_df.iloc[0]
        player_name = top_player[PLAYER_NAME_COL]

        role_info = ROLES[role_name]
        metrics_for_plotting = role_info['metrics']
        inverse_metrics_for_plotting = role_info['inverse_metrics']

        valid_metrics_for_plotting = [m for m in metrics_for_plotting if m in df_filtered.columns]
        if valid_metrics_for_plotting:
            # Re-scale the player's metrics against all players in the filtered dataset
            temp_df_for_scaling_all = df_filtered[valid_metrics_for_plotting].copy()
            
            # Filter out players that only have zero values for these metrics to avoid MinMaxScaler errors
            temp_df_for_scaling_all_non_zero = temp_df_for_scaling_all[(temp_df_for_scaling_all != 0).any(axis=1)]

            if not temp_df_for_scaling_all_non_zero.empty:
                scaler_plot = MinMaxScaler()
                scaled_all_players_for_plot = scaler_plot.fit_transform(temp_df_for_scaling_all_non_zero)
                scaled_df_all_players_for_plot = pd.DataFrame(scaled_all_players_for_plot,
                                                            columns=valid_metrics_for_plotting,
                                                            index=temp_df_for_scaling_all_non_zero.index)

                # Get the original index of the top player to retrieve their scaled values
                player_original_row_index_df_filtered = df_filtered[df_filtered[PLAYER_NAME_COL] == player_name].index
                
                if not player_original_row_index_df_filtered.empty:
                    player_original_row_index = player_original_row_index_df_filtered[0]
                    
                    if player_original_row_index in scaled_df_all_players_for_plot.index:
                        player_scaled_values = scaled_df_all_players_for_plot.loc[player_original_row_index].copy()

                        for col in inverse_metrics_for_plotting:
                            if col in player_scaled_values.index:
                                player_scaled_values[col] = 1 - player_scaled_values[col]

                        plot_df_player = pd.DataFrame([player_scaled_values.tolist()], columns=player_scaled_values.index)

                        # Clean player name for filename (remove dots, etc.)
                        clean_player_name = player_name.replace(' ', '_').replace('.', '').replace("'", "")
                        clean_role_name = role_name.replace(' ', '_').replace('.', '').replace("'", "")
                        create_radar_chart(plot_df_player, player_name, player_scaled_values.index.tolist(),
                                           f"{player_name} as a {role_name}",
                                           filename=f"{clean_player_name}_{clean_role_name}_radar")
                    else:
                         print(f"  Warning: Top player {player_name} not found in scaled data for radar chart (may have all zero metrics for role).")
                else:
                    print(f"  Warning: Could not find original row for {player_name} for radar chart.")
            else:
                print(f"  No non-zero data for plotting metrics for '{role_name}'. Skipping radar chart.")
        else:
            print(f"  Cannot create radar chart for {player_name}: no valid metrics for plotting.")


# Example Scatter Plot (Strikers: Shots P90 vs. Goals P90)
if 'Goal Poaching Striker' in top_players_by_role:
    if "Goals_P90" in df_filtered.columns and "Shots_P90" in df_filtered.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_filtered, x="Shots_P90", y="Goals_P90", alpha=0.6, label='All Players')

        top_strikers = top_players_by_role['Goal Poaching Striker']
        sns.scatterplot(data=top_strikers, x="Shots_P90", y="Goals_P90", color='red', s=100, label='Top Strikers', zorder=5)

        for i, row in top_strikers.iterrows():
            plt.text(row["Shots_P90"] * 1.02, row["Goals_P90"] * 1.02, row[PLAYER_NAME_COL], fontsize=9)

        plt.title('Goals P90 vs. Shots P90 for Strikers')
        plt.xlabel('Shots P90')
        plt.ylabel('Goals P90')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig('./striker_shots_goals_scatter.png', bbox_inches='tight')
        print("Saved striker_shots_goals_scatter.png")
        plt.close()
    else:
        print("  Cannot create striker scatter plot: Missing 'Goals_P90' or 'Shots_P90' columns.")

print("\n--- Script Finished ---")
