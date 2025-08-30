import mlflow

import pymc as pm
import numpy as np
import numpy.typing as npt


def mlflow_set_tracking_uri() -> None:
    local_file = "mlruns.db"
    uri = f"sqlite:///{local_file}"
    mlflow.set_tracking_uri(uri=uri)


def define_normal_model(data: npt.NDArray[np.float64]) -> pm.Model:
    coords = {"idx": np.arange(data.size)}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        observed = pm.Data("data", data, dims="idx")
        pm.Normal("obs", mu=mu, sigma=sigma, observed=observed, dims="idx")

    return model


def define_student_t_model(data: npt.NDArray[np.float64]) -> pm.Model:
    coords = {"idx": np.arange(data.size)}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        nu = pm.Exponential("nu", lam=1 / 30)

        observed = pm.Data("data", data, dims="idx")
        pm.StudentT("obs", mu=mu, sigma=sigma, nu=nu, observed=observed, dims="idx")

    return model


def define_gamma_model(data: npt.NDArray[np.float64]) -> pm.Model:
    coords = {"idx": np.arange(data.size)}
    with pm.Model(coords=coords) as model:
        mu = pm.HalfNormal("mu", sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        observed = pm.Data("data", data, dims="idx")
        pm.Gamma("obs", alpha=mu, beta=sigma, observed=observed, dims="idx")

    return model


def generate_normal_data(
    n: int,
    rng: np.random.Generator,
    mu,
    sigma,
) -> npt.NDArray[np.float64]:
    dist = pm.Normal.dist(mu=mu, sigma=sigma, shape=n)
    return pm.draw(dist, random_seed=rng)
