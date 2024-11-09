import os
import pandas as pd


def load_data():
    # Check if files exist before loading
    if not os.path.exists("../data/raw/results.csv"):
        raise FileNotFoundError("results.csv not found in data/raw/")
    if not os.path.exists("../data/raw/shootouts.csv"):
        raise FileNotFoundError("shootouts.csv not found in data/raw/")
    if not os.path.exists("../data/raw/goalscorers.csv"):
        raise FileNotFoundError("goalscorers.csv not found in data/raw/")

    # Load datasets
    results_df = pd.read_csv("../data/raw/results.csv")
    shootouts_df = pd.read_csv("../data/raw/shootouts.csv")
    goalscorers_df = pd.read_csv("../data/raw/goalscorers.csv")

    return results_df, shootouts_df, goalscorers_df

# Example usage (optional)
# if __name__ == "__main__":
#     results_df, shootouts_df, goalscorers_df = load_data()
#     print("Data loaded successfully")

def preprocess_results(results_df, shootouts_df):
    # Create the 'outcome' column based on scores
    def determine_outcome(row):
        if row['home_score'] > row['away_score']:
            return 'home_win'
        elif row['home_score'] < row['away_score']:
            return 'away_win'
        else:
            return 'draw'

    results_df['outcome'] = results_df.apply(determine_outcome, axis=1)

    # Merge shootouts to assign winners for drawn matches
    results_df = pd.merge(results_df, shootouts_df[['date', 'home_team', 'away_team', 'winner']],
                          on=['date', 'home_team', 'away_team'], how='left')

    # Update outcome based on shootout results
    def update_outcome(row):
        if row['outcome'] == 'draw' and pd.notna(row['winner']):
            return 'home_win' if row['winner'] == row['home_team'] else 'away_win'
        return row['outcome']

    results_df['outcome'] = results_df.apply(update_outcome, axis=1)

    return results_df

# Example usage
results_df, shootouts_df, _ = load_data()
results_df = preprocess_results(results_df, shootouts_df)
# print(results_df.info())
