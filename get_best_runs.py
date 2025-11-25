import mlflow

# Set the experiment ID
EXPERIMENT_ID = "268599232806679177"

# Search for the best XGBoost run
best_xgboost_run = mlflow.search_runs(
    experiment_ids=[EXPERIMENT_ID],
    filter_string="tags.mlflow.runName LIKE 'XGBoost%'",
    order_by=["metrics.r2 DESC"],
    max_results=1,
)

# Search for the best GradientBoosting run
best_gb_run = mlflow.search_runs(
    experiment_ids=[EXPERIMENT_ID],
    filter_string="tags.mlflow.runName LIKE 'GradientBoosting%'",
    order_by=["metrics.r2 DESC"],
    max_results=1,
)

print("--- Best XGBoost Run ---")
for _, row in best_xgboost_run.iterrows():
    print({k: v for k, v in row.items() if k.startswith('params.')})

print("\n--- Best GradientBoosting Run ---")
for _, row in best_gb_run.iterrows():
    print({k: v for k, v in row.items() if k.startswith('params.')})
