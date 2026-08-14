## PyMC-Marketing Marketing Mix Model MLflow Logging

The [`__main__.py`](__main__.py) script kicks off [various model
configurations](./run-config.yaml) and logs each as a separate run in MLflow.
The script leverages the `pymc_marketing.mlflow.autolog` function as well as
various `mlflow` log functions in order to customize the logging of the model.

The configurations use the `pymc-marketing>=1.0` serialization format: each
adstock / saturation entry carries a `__type__` key with the fully-qualified
class path, and is deserialized with `pymc_marketing.serialization`. A
different configuration file can be passed with `--config <path>`.

## Model registry

Registration is only wired up in this example, since it is the one that fits a
grid of competing models and therefore has something to choose between. Nothing
about it is specific to MMMs though: any of the other scripts could register
its model by passing `registered_model_name` to the relevant log function, or
by calling `mlflow.register_model` on a finished run and setting an alias.

Every fit is logged as an MLflow model, but only the winner reaches the
registry. Selection is validity first, score second: configurations that
sampled with divergences are discarded, and the highest
`out-sample_r_squared_mean` among the remaining ones is registered as a new
version of `pymc-marketing-mmm`. Filtering on `total_divergences` matters
because the metric alone cannot separate these models — a typical batch spans
0.001 in R-squared while one configuration diverges thousands of times.

The promoted version gets the `champion` alias and a
`validation_status=approved` tag, alongside the score and divergence count as
evidence. Aliases and tags replace the model stages that MLflow deprecated in
2.9: the alias is a movable pointer to what is deployed, while tags are durable
facts about a version. Load it with:

```python
mlflow.pyfunc.load_model("models:/pymc-marketing-mmm@champion")
```

Below are a subset of the metrics & parameters:

![Autologging](./../images/mmm-autolog.png)

And the artifacts that are saved off:

![Autolog Artifacts](./../images/mmm-autolog-artifacts.png)
