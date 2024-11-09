# runs the entire pipeline and flow of the application
from src.data_preprocessing import load_data, preprocess_results
from src.model_training import train_model

# Load and preprocess data
results_df, shootouts_df, goalscorers_df = load_data()
results_df = preprocess_results(results_df, shootouts_df)

# Train and evaluate the model
train_model(results_df)
