"""PyMC-Marketing autologging.

Use standard PyMC code after setting up the autologging allows
for the automatic logging of various artifacts and metrics.

"""

import mlflow

from utils import (
    define_gamma_model,
    define_normal_model,
    define_student_t_model,
    generate_normal_data,
    mlflow_set_tracking_uri,
)
import arviz as az

import pymc as pm
from pymc.testing import mock_sample
import matplotlib.pyplot as plt

import numpy as np

import pymc_marketing.mlflow

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "nuts_sampler",
    type=str,
    choices=["pymc", "nutpie", "numpyro"],
    help="NUTS sampler to use.",
)
parser.add_argument(
    "likelihood",
    type=str,
    choices=["normal", "student_t", "gamma"],
    help="Likelihood to use.",
)
# Option for --mock or --no-mock with default of --no-mock
parser.add_argument(
    "--mock",
    action="store_true",
    help="Use mock sampling instead of actual sampling",
)


if __name__ == "__main__":
    seed = sum(map(ord, "Logging PyMC model"))
    rng = np.random.default_rng(seed)

    data = generate_normal_data(n=100, rng=rng, mu=2.5, sigma=3.5)

    args = parser.parse_args()

    define_model = {
        "normal": define_normal_model,
        "student_t": define_student_t_model,
        "gamma": define_gamma_model,
    }[args.likelihood]

    # Only MLflow related setup
    pymc_marketing.mlflow.autolog()

    mlflow_set_tracking_uri()
    mlflow.set_experiment("03-pymc-autologging")

    sample_kwargs = {"nuts_sampler": args.nuts_sampler}

    if args.mock:
        pm.sample = mock_sample

    if args.nuts_sampler == "pymc":
        callback = pymc_marketing.mlflow.create_log_callback(
            stats=["energy", "model_logp", "step_size"],
            parameters=["mu", "sigma_log__"],
            take_every=100,
        )
        sample_kwargs["callback"] = callback

    with mlflow.start_run():
        mlflow.log_param("mock", args.mock)
        model = define_model(data)
        idata = pm.sample(model=model, **sample_kwargs)

        pymc_marketing.mlflow.log_inference_data(idata)

        az.plot_forest(
            idata,
            var_names=["mu", "sigma"],
        )
        mlflow.log_figure(plt.gcf(), "forest.png")
