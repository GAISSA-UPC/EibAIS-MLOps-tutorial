# EibAIS MLOps tutorial

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This is a toy project to demonstrate the use of MLOps practices in ML projects used in the tutorial [Software Engineering for ML Systems](https://conf.researchr.org/track/cibse-2025/cibse-2025-eibais#Tutorial-3) at the [EibAIS 2025](https://conf.researchr.org/track/cibse-2025/cibse-2025-eibais#About) school.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Guides to implement some of the MLOps practices
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src   <- Source code for use in this project.
    │
    ├── config.py                       <- Store useful variables and configuration
    │
    ├── data                            <- Scripts to process data                
    │   ├── download_raw_dataset.py     <- Download raw dataset
    │   ├── preprocess.py               <- Preprocess raw dataset
    │   ├── split_data.py               <- Split raw dataset into train, validation, and test sets
    │   ├── gx_context_configuration.py <- Great Expectations context configuration
    │   └── validate_data.py            <- Validate quality of the data with Great Expectations
    │
    └── modeling                
        ├── __init__.py 
        ├── evaluate_model.py           <- Code to evaluate the latest trained model
        └── train.py                    <- Code to train models
```

--------

## Guides
- [Data versioning with DVC](docs/dvc-demo.md)
- [Experiment tracking with MLflow](docs/mlflow-demo.md)
- [Data quality with Great Expectations](docs/great-expectations-demo.md)
- [Model deployment with FastAPI](docs/fastapi-demo.md)
