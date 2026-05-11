# CSCI 567 Final Project

**Project: Molecular Representation Matters**

A comparison of molecular representations — RDKit descriptors, Morgan fingerprints + TF-IDF, ChemBERTa embeddings, and a Graph Convolutional Network — for multi-task prediction of five polymer properties on the [NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) Kaggle competition.

## Dataset

Source: [NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) on Kaggle.

Files used:

- `train.csv` — polymers (SMILES) labelled with up to five targets.
- `test.csv` — held-out polymers.
- `train_supplement/dataset1.csv` — supplementary `Tc` labels.
- `train_supplement/dataset3.csv` — supplementary `Tg` labels.
- `train_supplement/dataset4.csv` — supplementary `FFV` labels.

Targets and the competition's weighted-MAE metric:

| Target  | Description            | Weight |
| ------- | ---------------------- | -----: |
| Tg      | Glass transition temp. |    1.0 |
| FFV     | Fractional free volume |   10.0 |
| Tc      | Critical temperature   |    1.0 |
| Density | Polymer density        |    1.0 |
| Rg      | Radius of gyration     |    1.0 |

## Notebooks

- [polymer-rdkit-lgbm.ipynb](polymer-rdkit-lgbm.ipynb) — Computes ~200 RDKit molecular descriptors per polymer and trains a 5-fold out-of-fold LightGBM regressor for each target.
- [polymer-morgan-tfidf.ipynb](polymer-morgan-tfidf.ipynb) — Three feature variants on SMILES — 2048-bit Morgan fingerprints, 4096-D TF-IDF character n-grams, and their concatenation — each fed to a 5-fold LightGBM. Merges in the supplementary `Tc` / `Tg` / `FFV` datasets.
- [chemberta-workbook.ipynb](chemberta-workbook.ipynb) — Extracts 384-D embeddings from DeepChem ChemBERTa-77M (both MLM and MTR checkpoints) and runs them through Ridge and LightGBM with 5-fold CV.
- [gcn-5fold-oof.ipynb](gcn-5fold-oof.ipynb) — A 2-layer Graph Convolutional Network on RDKit-built molecular graphs, trained with a masked multi-task loss and 5-fold OOF validation. Predictions and summary stats are written to `./gcn_outputs/`.

## Running on Kaggle (recommended)

The notebooks load data from `/kaggle/input/competitions/neurips-open-polymer-prediction-2025/` and run unmodified on Kaggle:

1. Open a new Kaggle Notebook and **File → Upload** the `.ipynb` you want to run.
2. In the right sidebar, **Add Input → Competitions → NeurIPS Open Polymer Prediction 2025**.
3. Enable a GPU accelerator for [chemberta-workbook.ipynb](chemberta-workbook.ipynb) and [gcn-5fold-oof.ipynb](gcn-5fold-oof.ipynb); CPU is fine for the LightGBM notebooks.
4. **Run All**.

## Running locally (optional)

Requires Python 3.10+ (a fresh `venv` or `conda` env is recommended).

1. Install dependencies:

   ```
   pip install pandas numpy scikit-learn lightgbm rdkit torch transformers tqdm scipy
   ```

2. Download the dataset with the Kaggle CLI (requires a configured `~/.kaggle/kaggle.json` token and accepting the competition rules on Kaggle):

   ```
   kaggle competitions download -c neurips-open-polymer-prediction-2025 -p ./data
   unzip ./data/neurips-open-polymer-prediction-2025.zip -d ./data
   ```

3. Point each notebook at the local data directory. Near the top of each notebook there is a single path variable — change it from the Kaggle path to `./data`:

   | Notebook                                                 | Variable   |
   | -------------------------------------------------------- | ---------- |
   | [polymer-rdkit-lgbm.ipynb](polymer-rdkit-lgbm.ipynb)     | `DATA_DIR` |
   | [polymer-morgan-tfidf.ipynb](polymer-morgan-tfidf.ipynb) | `BASE`     |
   | [chemberta-workbook.ipynb](chemberta-workbook.ipynb)     | `DATA_DIR` |
   | [gcn-5fold-oof.ipynb](gcn-5fold-oof.ipynb)               | `DATA_DIR` |

   The RDKit / ChemBERTa notebooks also contain a Kaggle-only `pip install` cell that points to a `.whl` under `/kaggle/input/datasets/...`; skip or comment out that cell when running locally (RDKit is already installed via the `pip install` step above).

## Repo layout

```
.
├── README.md
├── chemberta-workbook.ipynb
├── gcn-5fold-oof.ipynb
├── polymer-morgan-tfidf.ipynb
└── polymer-rdkit-lgbm.ipynb
```
