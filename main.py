from src.data_preprocessing import load_data, calculate_team_stats
from src.model_training import train_model

# Load data and calculate team stats
data_path = "C:/Users/rish2/OneDrive/Desktop/project1/data/raw/results.csv"
df = load_data(data_path)
team_stats = calculate_team_stats(df)

# Train model
train_model(data_path)
