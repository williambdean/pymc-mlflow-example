from itertools import product
from pathlib import Path

import mlflow

import yaml

import pandas as pd

import pymc_marketing.mlflow
from pymc_marketing.mmm import (
    MMM,
    adstock_from_dict,
    saturation_from_dict,
)

from utils import mlflow_set_tracking_uri

HERE = Path(__file__).parent


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_data() -> pd.DataFrame:
    data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_example.csv"
    return pd.read_csv(data_url, parse_dates=["date_week"])


def run_experiment(X, y, adstock_config, saturation_config, yearly_seasonality):
    adstock = adstock_from_dict(adstock_config)
    saturation = saturation_from_dict(saturation_config)

    mmm = MMM(
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
        date_column="date_week",
        channel_columns=["x1", "x2"],
        control_columns=[
            "event_1",
            "event_2",
            "t",
        ],
    )

    with mlflow.start_run():
        idata = mmm.fit(X, y, nuts_sampler="numpyro")

        for transform in [mmm.adstock, mmm.saturation, mmm.yearly_fourier]:
            curve = transform.sample_curve(idata.posterior)
            fig, _ = transform.plot_curve(curve)
            mlflow.log_figure(fig, f"{transform.prefix}_curve.png")

        pymc_marketing.mlflow.log_mmm(mmm=mmm)


def run_experiments(X, y, combinations):
    for adstock_config, saturation_config, yearly_seasonality in combinations:
        run_experiment(X, y, adstock_config, saturation_config, yearly_seasonality)


def main():
    data = read_data()

    X = data.drop("y", axis=1)
    y = data["y"]

    mlflow_set_tracking_uri()
    mlflow.set_experiment("04-pymc-marketing-mmm")

    pymc_marketing.mlflow.autolog()

    config_file = HERE / "run-config.yaml"
    config = load_config(path=config_file)

    combinations = list(
        product(
            config["adstocks"],
            config["saturations"],
            config["yearly_seasonality"],
        )
    )
    print(f"Running a combination of {len(combinations)} MMM models")

    run_experiments(X, y, combinations)


if __name__ == "__main__":
    main()
