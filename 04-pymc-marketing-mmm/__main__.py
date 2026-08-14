import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import mlflow
import pandas as pd
import pymc_marketing.mlflow
import yaml
from mlflow import MlflowClient
from pymc_marketing.mmm import MMM
from pymc_marketing.serialization import serialization

from utils import mlflow_set_tracking_uri

HERE = Path(__file__).parent

REGISTERED_MODEL_NAME = "pymc-marketing-mmm"
SELECTION_METRIC = "out-sample_r_squared_mean"
CHAMPION_ALIAS = "champion"
MODEL_ARTIFACT_PATH = "model"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=Path,
    default=HERE / "run-config.yaml",
    help="Path to the run configuration YAML file.",
)


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


def run_experiment(
    split: Split, adstock_config, saturation_config, yearly_seasonality
) -> str:
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

    with mlflow.start_run() as run:
        mmm.build_model(split.train.X, split.train.y)
        mmm.add_original_scale_contribution_variable(
            var=[
                "channel_contribution",
                "control_contribution",
                "intercept_contribution",
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

        pymc_marketing.mlflow.log_mmm(mmm=mmm, artifact_path=MODEL_ARTIFACT_PATH)

        return run.info.run_id


def select_best_run(metrics_by_run: dict[str, dict[str, float]]) -> str | None:
    clean = {
        run_id: metrics
        for run_id, metrics in metrics_by_run.items()
        if metrics.get("total_divergences") == 0 and SELECTION_METRIC in metrics
    }
    return max(clean, key=lambda run_id: clean[run_id][SELECTION_METRIC], default=None)


def promote_best_model(run_ids: list[str]) -> None:
    metrics_by_run = {run_id: mlflow.get_run(run_id).data.metrics for run_id in run_ids}

    best_run_id = select_best_run(metrics_by_run)
    if best_run_id is None:
        print("No fit with zero divergences; nothing registered")
        return

    best_score = metrics_by_run[best_run_id][SELECTION_METRIC]
    version = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/{MODEL_ARTIFACT_PATH}",
        name=REGISTERED_MODEL_NAME,
    ).version

    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version, "validation_status", "approved"
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version, SELECTION_METRIC, best_score
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "total_divergences", 0)

    print(
        f"Promoted {REGISTERED_MODEL_NAME} version {version} "
        f"({SELECTION_METRIC}={best_score:.4f}) to @{CHAMPION_ALIAS}"
    )


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

    config = yaml.safe_load(args.config.read_text())

    combinations = list(
        product(
            config["adstocks"],
            config["saturations"],
            config["yearly_seasonality"],
        )
    )
    print(f"Running a combination of {len(combinations)} MMM models")

    run_ids = [run_experiment(split, *combination) for combination in combinations]

    promote_best_model(run_ids)


if __name__ == "__main__":
    main()
