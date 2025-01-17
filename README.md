# PyMC with MLflow

These are simple examples of using PyMC with MLflow, taking advantage of the
`pymc_marketing.mlflow` module.

This focuses on logging parameters, metrics, and artifacts to MLflow.

![Autologging](./images/autolog.png)

Suggestions or Questions? [Comment on this Issue](https://github.com/pymc-labs/pymc-marketing/issues/938)

## Scripts

There are four scripts: 

1. [Non-PyMC example showing how to log parameters, metrics, and artifacts to MLflow](./01-basic-introduction.py)
2. [PyMC example which logs some PyMC related metrics to MLflow](./02-pymc-context.py)
3. [Logging that and more with `pymc_marketing.mlflow` module](./03-pymc-autologging.py)
4. [Autologging of Marketing Mix Model with `pymc_marketing.mlflow` module](./04-pymc-marketing-mmm)

Kick them off with `make experiments`. View with `make serve`. Clean up with `make clean_up`.

Use the `environment.yml` file to create the conda environment. i.e. `conda env create -f environment.yml`.

There are some helper functions in the `utils.py` file which help setup mlflow and define some reused PyMC models.

## Resources

- [`pymc_marketing.mlflow` module](https://www.pymc-marketing.io/en/latest/api/generated/pymc_marketing.mlflow.html)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
