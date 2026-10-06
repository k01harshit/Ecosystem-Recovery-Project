# Ecosystem Collapse & Recovery Prediction

A deep learning benchmarking suite for predicting species recovery after ecosystem collapse using **Graph Neural Networks (GNNs)**.

The project trains models—primarily **H2GCN (Heterophily-aware Graph Convolutional Network)**—on synthetic food-webs and evaluates their **zero-shot transferability on 39 published empirical food webs** from marine, lake, stream, and estuarine ecosystems.

## Overview

Ecosystem dynamics are simulated using a tri-trophic **Rosenzweig–MacArthur consumer–resource ODE model**. Each food web undergoes:

1. Initial equilibrium simulation
2. Severe species collapse
3. Basal-resource subsidy intervention ($\delta$)
4. Post-intervention recovery simulation

The task is formulated as a **node-level binary classification problem**, where the model predicts whether a species recovers above a population-density threshold of **0.1 at $t=2000$**.

## Key Features

- **Zero-shot transfer:** Synthetic training → 39 unseen empirical food webs
- **Heterophily-aware learning:** H2GCN for trophic food-web structure
- **Comprehensive baselines:** H2GCN, GCN, GAT, GraphSAGE, MLP, Random Forest, Logistic Regression, and heuristic models
- **Ecological metrics:** ROC-AUC, PR-AUC, F1, MCC, Balanced Accuracy, Cohen's Kappa, Sensitivity, and Specificity
- **Feature analysis:** Ablation and isolation experiments for structural and intervention features
- **Automated outputs:** Figures, CSV tables, and an ecology-oriented markdown report

## Installation

```bash
pip install numpy pandas scipy networkx scikit-learn torch torch_geometric matplotlib seaborn
```

Place the 39 empirical food webs in:

```text
real_foodwebs/
```

## Usage

### Full Benchmark

```bash
python ecosystem_final02.py
```

### Quick Test

**Linux/macOS:**
```bash
ECO_QUICK=1 python ecosystem_final02.py
```

**Windows PowerShell:**
```powershell
$env:ECO_QUICK="1"; python ecosystem_final02.py
```

## Outputs

Results are automatically generated in:

```text
ecosystem_plots/
```

including:

- Model performance and zero-shot transfer plots
- ROC/PR curves
- Feature ablation and isolation analysis
- Hyperparameter and scalability analysis
- Empirical network recovery results
- Raw CSV results in `ecosystem_plots/tables/`
- Automated report: `ecosystem_plots/paper_report.md`

## Data Provenance

The 39 empirical food webs are derived primarily from the R packages **`igraphdata`** and **`cheddar`**, based on published ecological datasets. Network-specific references are recorded in:

```text
ecosystem_plots/tables/empirical_network_citations.csv
```

## Objective

The project investigates whether **GNNs can learn generalizable ecosystem recovery patterns from synthetic food-web simulations and transfer these predictions to real-world ecosystems**.
