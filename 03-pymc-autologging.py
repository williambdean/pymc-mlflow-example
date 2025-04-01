"""PyMC-Marketing autologging.

Use standard PyMC code after setting up the autologging allows
for the automatic logging of various artifacts and metrics.

"""

import sys

import mlflow

from utils import (
    mlflow_set_tracking_uri,
    define_normal_model,
    define_student_t_model,
    generate_normal_data,
)

import pymc as pm
import numpy as np

import pymc_marketing.mlflow

if __name__ == "__main__":
    seed = sum(map(ord, "Logging PyMC model"))
    rng = np.random.default_rng(seed)

    data = generate_normal_data(n=100, rng=rng, mu=2.5, sigma=3.5)

    nuts_sampler, likelihood = sys.argv[1:]
    define_model = {
        "normal": define_normal_model,
        "student_t": define_student_t_model,
    }[likelihood]

    # Only MLflow related setup
    pymc_marketing.mlflow.autolog()

    mlflow_set_tracking_uri()
    mlflow.set_experiment("03-pymc-autologging")

    with mlflow.start_run():
        model = define_model(data)
        idata = pm.sample(nuts_sampler=nuts_sampler, model=model)

        pymc_marketing.mlflow.log_inference_data(idata)
