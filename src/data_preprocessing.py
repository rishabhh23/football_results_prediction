import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)


def prepare_training_data(df, team_stats):
    data = []
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_stats = team_stats[home_team]
        away_stats = team_stats[away_team]

        # 1 for home win, 0 for draw, -1 for away win
        outcome = 1 if row['home_score'] > row['away_score'] else (0 if row['home_score'] == row['away_score'] else -1)

        data.append([
            home_stats['win_rate'], away_stats['win_rate'],
            home_stats['avg_score'], away_stats['avg_score'],
            outcome
        ])

    model_df = pd.DataFrame(data, columns=['home_win_rate', 'away_win_rate', 'home_avg_score', 'away_avg_score', 'outcome'])
    model_df = model_df[model_df['outcome'] != 0]
    return model_df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def calculate_team_stats(df):
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    stats = {}

    for team in teams:
        home_games = df[df['home_team'] == team]
        away_games = df[df['away_team'] == team]

        total_games = len(home_games) + len(away_games)
        total_wins = len(home_games[home_games['home_score'] > home_games['away_score']]) + len(
            away_games[away_games['away_score'] > away_games['home_score']])
        total_draws = len(home_games[home_games['home_score'] == home_games['away_score']]) + len(
            away_games[away_games['home_score'] == away_games['away_score']])

        win_rate = total_wins / total_games if total_games > 0 else 0
        avg_home_score = home_games['home_score'].mean() if not home_games.empty else 0
        avg_away_score = away_games['away_score'].mean() if not away_games.empty else 0
        avg_score = (avg_home_score + avg_away_score) / 2

        stats[team] = {
            'win_rate': win_rate,
            'avg_score': avg_score,
            'total_games': total_games,
            'draw_rate': total_draws / total_games if total_games > 0 else 0
        }

    return stats
