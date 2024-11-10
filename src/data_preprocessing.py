import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)


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
