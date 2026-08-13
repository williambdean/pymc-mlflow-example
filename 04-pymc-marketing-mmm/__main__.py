import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import mlflow

import yaml

import pandas as pd

import pymc_marketing.mlflow
from pymc_marketing.mmm import MMM
from pymc_marketing.serialization import serialization

from utils import mlflow_set_tracking_uri

HERE = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=Path,
    default=HERE / "run-config.yaml",
    help="Path to the run configuration YAML file.",
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_data() -> pd.DataFrame:
    data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/1.0.0/data/mmm_example.csv"
    return pd.read_csv(data_url, parse_dates=["date_week"])


@dataclass
class Data:
    X: pd.DataFrame
    y: pd.Series

    @classmethod
    def from_df(cls, df, target_col):
        return cls(
            X=df.drop(target_col, axis=1),
            y=df[target_col],
        )


@dataclass
class Split:
    train: Data
    test: Data


def run_experiment(split: Split, adstock_config, saturation_config, yearly_seasonality):
    adstock = serialization.deserialize(adstock_config)
    saturation = serialization.deserialize(saturation_config)

    mmm = MMM(
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
        date_column="date_week",
        target_column="y",
        channel_columns=["x1", "x2"],
        control_columns=[
            "event_1",
            "event_2",
            "t",
        ],
    )

    with mlflow.start_run():
        # The 1.0 MMM max-scales the target internally and no longer
        # inverse-transforms predictions; register original-scale
        # deterministics before fitting so metrics use the data scale.
        mmm.build_model(split.train.X, split.train.y)
        mmm.add_original_scale_contribution_variable(
            var=[
                "channel_contribution",
                "control_contribution",
                "intercept_contribution",
                "yearly_seasonality_contribution",
                "y",
            ]
        )

        idata = mmm.fit(split.train.X, split.train.y, nuts_sampler="numpyro")
        posterior = idata["posterior"].to_dataset()

        for transform in [mmm.adstock, mmm.saturation, mmm.yearly_fourier]:
            curve = transform.sample_curve(posterior)
            fig, _ = transform.plot_curve(curve)
            mlflow.log_figure(fig, f"{transform.prefix}_curve.png")

        # metrics expect posterior predictive samples with shape (date, sample)
        in_predictions = posterior["y_original_scale"].stack(sample=("chain", "draw"))
        out_predictions = mmm.sample_posterior_predictive(
            X=split.test.X,
            include_last_observations=True,
            var_names=["y_original_scale"],
        ).y_original_scale

        metrics_to_calculate = ["r_squared", "rmse"]
        pymc_marketing.mlflow.log_mmm_evaluation_metrics(
            y_true=split.train.y,
            y_pred=in_predictions,
            prefix="in-sample",
            metrics_to_calculate=metrics_to_calculate,
        )
        pymc_marketing.mlflow.log_mmm_evaluation_metrics(
            y_true=split.test.y,
            y_pred=out_predictions,
            prefix="out-sample",
            metrics_to_calculate=metrics_to_calculate,
        )

        pymc_marketing.mlflow.log_mmm(mmm=mmm)


def run_experiments(split: Split, combinations):
    for adstock_config, saturation_config, yearly_seasonality in combinations:
        run_experiment(split, adstock_config, saturation_config, yearly_seasonality)


def main():
    args = parser.parse_args()

    data = read_data()

    cutoff = "2021-01-01"

    idx_train = data["date_week"] < cutoff

    data_train = data.loc[idx_train]
    data_test = data.loc[~idx_train]

    split = Split(
        train=Data.from_df(data_train, target_col="y"),
        test=Data.from_df(data_test, target_col="y"),
    )

    mlflow_set_tracking_uri()
    mlflow.set_experiment("04-pymc-marketing-mmm")

    pymc_marketing.mlflow.autolog()

    config = load_config(path=args.config)

    combinations = list(
        product(
            config["adstocks"],
            config["saturations"],
            config["yearly_seasonality"],
        )
    )
    print(f"Running a combination of {len(combinations)} MMM models")

    run_experiments(split=split, combinations=combinations)


if __name__ == "__main__":
    main()
