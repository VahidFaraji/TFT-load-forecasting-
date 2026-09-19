# TFT-Electricity-load-forecasting
Multi-horizon electricity load forecasting using Temporal Fusion Transformers (TFT) with support for exogenous variables, rolling evaluation, multi-seed experiments, and computational cost analysis.

This repository contains the TFT implementation used in my MSc thesis at Lund University. 
The corresponding N-HiTS and N-BEATSx experiments are available in the [NeuralForecast repository](https://github.com/VahidFaraji/neuralforecast-electricity-load-forecasting).
ؤ
## Features

* Multi-horizon forecasting (24, 48, 96, 192, 336, 720 hours)
* Support for future known exogenous variables
* Fixed and rolling-origin evaluation
* Multi-seed experiments
* Automatic experiment management
* Forecast accuracy and computational cost analysis
* Support for ECL and PJM datasets

## Repository Structure

```text
src/            Source code
experiments/    Experiment outputs
figures/        Figures and visualizations
docs/           Additional documentation
```



## Outputs

The pipeline automatically generates:

* `metrics.json`
* `metrics_runs.csv`
* `timings_runs.csv`
* `timings_summary.csv`
* `cost_benefit.csv`

## Citation

If you use this repository in academic work, please cite the corresponding publication.
