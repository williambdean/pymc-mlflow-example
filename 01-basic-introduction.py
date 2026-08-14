"""Basic MLflow logging

Comprised of:

1. Pointing to the MLflow server
2. Pointing to an experiment
3. Starting a run
4. Logging parameters
5. Logging metrics
6. Logging artifacts

"""

import mlflow

from utils import mlflow_set_tracking_uri

# Start up
# Point to the MLflow server
mlflow_set_tracking_uri()


# Point to an experiment
experiment_name = "01-basis-introduction"
mlflow.set_experiment(experiment_name=experiment_name)


# Start a run
# mlflow.start_run() ... mlflow.end_run()
with mlflow.start_run(
    description="Some more information about the run",
    log_system_metrics=True,
) as run:
    # Information about the run
    # Various parameters describing the model
    mlflow.log_param("configuration", "something")
    mlflow.log_param("model-complexity", 9000)
    mlflow.log_param("tune", 1000)

    # Run some code
    # Can be anything (or nothing)

    # Log some metrics
    mlflow.log_metric("model-performance", 0.99)

    # Log various artifacts
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3, 2, 1])
    ax.set(xlabel="x", ylabel="y", title="Some plot")
    mlflow.log_figure(fig, "figure.png")

    mlflow.log_text("This is some text", "text.txt")
