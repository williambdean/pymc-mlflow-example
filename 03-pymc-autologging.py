"""PyMC-Marketing autologging.

Use standard PyMC code after setting up the autologging allows
for the automatic logging of various artifacts and metrics.

"""

import sys

import mlflow

from utils import (
    define_gamma_model,
    define_normal_model,
    define_student_t_model,
    generate_normal_data,
    mlflow_set_tracking_uri,
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
        "gamma": define_gamma_model,
    }[likelihood]

    # Only MLflow related setup
    pymc_marketing.mlflow.autolog()

    mlflow_set_tracking_uri()
    mlflow.set_experiment("03-pymc-autologging")

    sample_kwargs = {"nuts_sampler": nuts_sampler}

    if nuts_sampler == "pymc":
        callback = pymc_marketing.mlflow.create_log_callback(
            stats=["energy", "model_logp", "step_size"],
            parameters=["mu", "sigma_log__"],
            take_every=100,
        )
        sample_kwargs["callback"] = callback

    with mlflow.start_run():
        model = define_model(data)
        idata = pm.sample(model=model, **sample_kwargs)

        pymc_marketing.mlflow.log_inference_data(idata)
