import pandas as pd
import numpy as np
import os

def process_files(folder_path):
    merged_df = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') and not filename.endswith('_info.csv'):
            match_id = filename[:-4]
            match_df = pd.read_csv(os.path.join(folder_path, filename))
            info_filename = f"{match_id}_info.csv"
            try:
                info_df = pd.read_csv(os.path.join(folder_path, info_filename), error_bad_lines=False)
            except pd.errors.EmptyDataError:
                print(f"Skipping {info_filename} due to empty data.")
                continue

            # Add information from info_df to match_df
            match_df['toss_winner'] = info_df.loc[0, 'info.toss.winner']
            match_df['toss_decision'] = info_df.loc[0, 'info.toss.decision']
            match_df['player_of_match'] = info_df.loc[0, 'info.player_of_match']
            match_df['winner'] = info_df.loc[0, 'info.outcome.winner']

            # Create new columns for player IDs
            player_ids = info_df.set_index('info.players')['info.registry.people'].to_dict()
            match_df['striker_id'] = match_df['striker'].map(player_ids)
            match_df['non_striker_id'] = match_df['non_striker'].map(player_ids)
            match_df['bowler_id'] = match_df['bowler'].map(player_ids)

            # Add match_id as a column
            match_df['match_id'] = match_id

            merged_df = pd.concat([merged_df, match_df], ignore_index=True)

    return merged_df

# Specify the folder path where the CSV files are located
folder_path = './new_datasets/'

# Call the function to process the files and get the merged dataframe
result_df = process_files(folder_path)

# Print the resulting dataframe
print(result_df)