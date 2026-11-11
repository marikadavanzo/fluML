# fluML
### Transformer-based modelling and forecasting of influenza A(H3N2) haemagglutinin evolution

`fluML` is a research framework for investigating and forecasting the temporal evolution of influenza A(H3N2) haemagglutinin (HA) sequences. It combines protein sequence representations with temporal, geographical, phylogenetic and epitope-related information to model amino-acid variation and identify patterns of evolutionary change.

The repository contains a sequence-to-sequence Transformer architecture, training and inference scripts, prediction-generation and ensemble procedures, evaluation tools, and exploratory analyses of sequence evolution and antigenic characteristics.

The project focuses on **amino-acid-level evolutionary forecasting**, with particular attention to position-specific changes, emerging substitutions, conserved regions and antigenically relevant sites.

> **Research status:** This repository contains research code associated with an ongoing study. The manuscript is in preparation. The code and analyses should therefore be considered research-stage implementations rather than a packaged, fully reproducible software release.

---

## Table of contents

1. [Research objectives](#research-objectives)
2. [Framework overview](#framework-overview)
3. [Repository structure](#repository-structure)
4. [Modelling approach](#modelling-approach)
5. [Data requirements](#data-requirements)
6. [Installation and environment](#installation-and-environment)
7. [Running the main workflow](#running-the-main-workflow)
8. [Prediction evaluation](#prediction-evaluation)
9. [Exploratory and downstream analyses](#exploratory-and-downstream-analyses)
10. [Interpretability analyses](#interpretability-analyses)
11. [Outputs](#outputs)
12. [Reproducibility and limitations](#reproducibility-and-limitations)
13. [Citation and contact](#citation-and-contact)

---

## Research objectives

Influenza A(H3N2) undergoes continuous genetic evolution, particularly in the haemagglutinin protein, which is a major target of host antibody responses. Antigenic drift is driven by the accumulation and changing frequencies of amino-acid substitutions, including changes at antigenically relevant positions.

The objective of `fluML` is to investigate whether machine-learning models can learn temporal patterns of HA sequence evolution and forecast amino-acid-level changes in subsequent periods.

The framework addresses several related questions:

- Can historical HA sequence data be used to forecast subsequent amino-acid patterns?
- Which positions exhibit relatively stable or variable evolutionary behaviour?
- Can recent changes in amino-acid frequencies provide useful information about emerging substitutions?
- How do temporal and geographical patterns contribute to sequence forecasting?
- How do model predictions compare with consensus-based, vaccine-strain and other reference approaches?
- To what extent do predicted changes overlap with recognised HA antigenic sites?

The framework is intended to support the computational investigation of viral evolution. Sequence-based predictions should not be interpreted as direct measurements of antigenic phenotype or as independent evidence of vaccine effectiveness.

---

## Framework overview

The repository contains four main components.

**1. Sequence modelling**

A Transformer-based sequence-to-sequence architecture is implemented in TensorFlow/Keras. The model uses an ESM-2 tokenizer and combines sequence information with additional biological and spatiotemporal context.

**2. Prediction generation**

The prediction pipeline generates candidate HA sequences using trained models and reference or consensus-based approaches. It also includes procedures for incorporating recent evolutionary trends, position-specific constraints and ensemble strategies.

**3. Quantitative evaluation**

Predicted sequences are compared with observed sequences from subsequent periods. Evaluation includes positional accuracy and sequence-distance metrics, with separate consideration of all positions, antigenic sites and positions identified as shifted.

**4. Evolutionary and interpretability analyses**

The notebooks investigate historical amino-acid substitutions, sequence divergence, HI titre relationships, model comparisons and selected attention, divergence and epistasis-related outputs.

---

## Repository structure

```text
fluML/
│
├── models_ym_clade.py
├── predict_TrLs_clade.py
├── evaluation_new.py
├── evaluation_hamming.py
│
├── Distance_vs_titre.ipynb
├── Distance_vs_titre_hamming.ipynb
├── Distances_with_train_consensus.ipynb
├── antigenic_distance.ipynb
├── mutations.ipynb
├── interpretability_figures.ipynb
├── wilcoxin.ipynb
├── wilcoxin_hamming.ipynb
├── transformer_architecture_A4.ipynb
│
└── README.md
```

### Main Python scripts

| File | Description |
|---|---|
| `models_ym_clade.py` | Implements the sequence-to-sequence modelling framework, including data preparation, tokenisation, Transformer and LSTM architectures, training utilities, biological masks and evaluation metrics. |
| `predict_TrLs_clade.py` | Implements prediction and inference procedures, including consensus-based approaches, mutation-trend adjustments, position-specific constraints and ensemble prediction strategies. |
| `evaluation_new.py` | Evaluates saved predictions using BLOSUM62-based distances and positional accuracy, including comparisons across shifted, epitope and all HA positions. |
| `evaluation_hamming.py` | Evaluates predictions using Hamming and Levenshtein distances, positional accuracy and comparisons with seasonal vaccine strains and consensus sequences. |

### Jupyter notebooks

| Notebook | Purpose |
|---|---|
| `Distance_vs_titre.ipynb` | Processes haemagglutination-inhibition (HI) titre data, matches virus and serum isolates to sequence metadata, and investigates relationships between HI titre categories and Hamming sequence distances. |
| `Distance_vs_titre_hamming.ipynb` | Examines HI titre relationships using BLOSUM62-based sequence distances, including full-sequence and epitope-specific comparisons. |
| `Distances_with_train_consensus.ipynb` | Constructs historical training consensus sequences and compares them with target-season sequences, including geographical-zone-specific analyses and Hamming/BLOSUM-based divergence measures. |
| `antigenic_distance.ipynb` | Collects and compares predicted sequences and consensus references, summarising sequence-distance-based performance across models and seasons. |
| `mutations.ipynb` | Analyses seasonal amino-acid substitutions and consensus changes, including their distribution across HA positions and antigenic sites. |
| `interpretability_figures.ipynb` | Produces visualisations and analyses of model attention, geographical and clade divergence, epistasis-related outputs and predicted mutation characteristics. |
| `wilcoxin.ipynb` | Compares model performance using BLOSUM62-based distance and accuracy metrics, including seasonal summaries and paired statistical comparisons. |
| `wilcoxin_hamming.ipynb` | Performs corresponding model comparisons using Hamming-distance-based metrics. |
| `transformer_architecture_A4.ipynb` | Generates diagrams of the Transformer architecture and the overall spatiotemporal modelling pipeline. |

*Note:* Some notebooks contain exploratory analyses and alternative implementations. They are not necessarily designed to run as one sequential pipeline.

---

## Modelling approach

### 1. Sequence representation and preprocessing

The modelling framework operates on influenza A(H3N2) HA amino-acid sequences.

The data-processing functions in `models_ym_clade.py` and the analysis notebooks handle sequence loading, metadata extraction, sequence preparation and tokenisation.

The modelling workflow uses the ESM-2 tokenizer:

- Pretrained tokenizer: `facebook/esm2_t12_35M_UR50D`
- Protein sequence tokenisation through Hugging Face Transformers.
- Sequence padding and masking for batch processing.
- Position-specific sequence information for downstream modelling.

The associated research dataset includes complete HA sequences and metadata such as collection date, geographical transmission zone and clade.

For the forecasting analyses, the mature HA sequence is considered after removal of the 16-amino-acid signal peptide. The precise preprocessing requirements should be checked against the dataset and the relevant analysis before reproducing individual experiments.

### 2. Biological and spatiotemporal context

In addition to sequence tokens, the framework incorporates information designed to capture evolutionary and biological context.

Depending on the model configuration, these features include:

- **Temporal information:** collection dates and target-period information.
- **Geographical information:** transmission-zone representations.
- **Clade information:** sequence-clade context.
- **Position-specific amino-acid frequencies:** historical distributions of amino acids at individual HA positions.
- **Temporal frequency trends:** changes in amino-acid frequencies over recent and historical periods.
- **Epitope annotations:** information relating to recognised HA antigenic sites.
- **Reference sequences:** consensus or recent reference representations.
- **Position-specific variability:** entropy-related measures used to distinguish relatively conserved and variable positions.

The code also contains procedures for constructing phylogenetic recency weights and position-specific evolutionary constraints.

### 3. Transformer architecture

The main sequence-to-sequence architecture is implemented in `models_ym_clade.py`.

The implementation includes:

- Encoder-decoder Transformer components.
- Multi-head attention.
- Positional and sequence representations.
- Padding and look-ahead masks.
- Position-context gating.
- Spatiotemporal feature modulation through a FiLM-related layer.
- Time-decay components.
- Position-weighted training objectives.
- Masked sequence accuracy and mutation-oriented metrics.

The model is designed to use historical HA sequences to generate amino-acid predictions conditioned on the available contextual information.

The Python implementation also includes an LSTM-based alternative for model comparison.

An architecture diagram can be generated using `transformer_architecture_A4.ipynb`.

### 4. Position-specific evolutionary constraints

The prediction pipeline includes additional procedures intended to incorporate the heterogeneous evolutionary behaviour of HA positions.

Examples implemented in `predict_TrLs_clade.py` include:

- Recency-weighted amino-acid frequencies.
- Identification of relatively stable positions.
- Entropy-based guards against locking positions that exhibit recent variability.
- Detection of increasing amino-acid frequencies between historical and recent windows.
- Position-specific mutation-trend adjustments.
- Epistasis-related estimates based on weighted mutual information between amino-acid positions.
- Recent consensus and reference-sequence construction.
- Monte Carlo dropout-based positional uncertainty estimation.

These procedures are used to inform candidate sequence generation and prediction strategies. They should not be interpreted as proof of causal epistatic interactions or experimentally confirmed antigenic effects.

### 5. Prediction strategies

The prediction code includes multiple approaches for generating candidate sequences and comparing model outputs.

These include:

- Baseline consensus sequences.
- Recent and pooled consensus strategies.
- Position-specific mutation-trend approaches.
- Transformer-based predictions.
- Ensemble and majority-vote strategies.
- Unified ensemble consensus procedures.

The exact configuration and available prediction strategies depend on the implementation and arguments used in the scripts.

---

## Data requirements

The repository contains analysis code but does not distribute the underlying influenza sequence and HI titre datasets or the complete model-generated output directories.

The following external inputs are referenced by the scripts and notebooks.

### Sequence data

A master HA FASTA file is required, for example:

```text
h3n2_clean.fasta
```

The notebooks expect FASTA records containing metadata fields separated by `|`. A representative header structure is:

```text
sequence_id|A_H3N2|clade|YYYY-MM-DD|country|transmission_zone
```

The sequence identifier, subtype, clade, collection date, country and geographical transmission zone are used in different analyses.

The actual header format must be consistent with the parser used by the selected script or notebook.

### HI titre data

The HI titre analyses reference:

```text
D_H3N2(H3N2).csv
combined_data.csv
```

These files are used to match virus and serum strain names to sequence identifiers and to examine relationships between HI titre measurements and sequence divergence.

The notebooks include procedures for categorising censored or non-standard titre values and identifying unmatched virus or serum strains.

### Vaccine reference sequences

The evaluation scripts can use seasonal vaccine-strain FASTA files, organised in a directory such as:

```text
Vaccine_strains/
```

### Prediction and evaluation outputs

The evaluation and interpretability analyses expect generated prediction files and analysis outputs in the directory structures specified by the scripts.

Examples include:

- Predicted FASTA sequences.
- Prediction CSV files.
- Position-attention matrices.
- Zone- and clade-attention matrices.
- Zone- and clade-divergence matrices.
- Epistasis-related output tables.
- Model-specific seasonal evaluation results.

These outputs are generated by the modelling and prediction workflows; they are not included in the repository.

### Data access and acknowledgements

Influenza sequence data may be subject to database-specific access conditions, data-use agreements and acknowledgement requirements.

Users must obtain the necessary permissions and comply with the relevant database policies before using or redistributing sequence data. In particular, GISAID data must be handled in accordance with its access and acknowledgement requirements.

---

## Installation and environment

The main modelling scripts use Python and TensorFlow/Keras, while the analysis notebooks use common scientific Python packages.

The code imports or uses the following packages across the repository:

**Core scientific computing**
- NumPy
- pandas
- SciPy
- scikit-learn

**Deep learning and sequence modelling**
- TensorFlow
- Keras (through TensorFlow)
- Hugging Face Transformers

**Bioinformatics**
- Biopython
- python-Levenshtein

**Visualisation and statistical analysis**
- Matplotlib
- Seaborn
- scikit-posthocs
- PtitPrince
- NetworkX

**Additional utilities**
- python-dateutil
- tqdm

The precise package versions and compatible TensorFlow/CUDA configuration have not yet been formalised in a pinned environment specification. GPU execution depends on a compatible TensorFlow, CUDA and hardware configuration.

A general starting point for setting up an environment is:

```bash
git clone https://github.com/marikadavanzo/fluML.git
cd fluML

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

Install the required packages in versions compatible with your Python and TensorFlow environment. The main model script sets `TF_USE_LEGACY_KERAS=1`, so the corresponding legacy Keras compatibility requirements should also be respected.

The ESM-2 tokenizer is loaded through Hugging Face Transformers. Access to the pretrained tokenizer may require an internet connection on first use.

**Important:** This is not yet a one-command installation. A fully pinned `requirements.txt` or environment file should be added after testing the scripts in a clean environment.

---

## Running the main workflow

The following commands reflect the command-line interfaces implemented in the Python scripts. They assume that the required input data and directory structures have already been prepared.

### 1. Train the sequence model

The main modelling script is:

```bash
python models_ym_clade.py h3n2_clean.fasta \
    --start_target_year 2014 \
    --end_target_year 2019
```

The script accepts:

| Argument | Description |
|---|---|
| `fasta_file` | Input HA FASTA file. |
| `--start_target_year` | First target year in the modelling period. |
| `--end_target_year` | Final target year in the modelling period. |
| `--start-month` | Optional manual starting month. |
| `--end-month` | Optional manual ending month. |
| `--epochs` | Number of training epochs. |
| `--batch-size` | Training batch size. |

The default target-year range is defined in the script and can be overridden using the command-line arguments.

The implementation includes seasonal and rolling-window data preparation. Consult the script configuration when reproducing a specific forecasting experiment, as the temporal training and evaluation windows are part of the experimental design.

### 2. Generate predictions

The prediction and inference procedures are implemented in:

```text
predict_TrLs_clade.py
```

The script exposes the following command-line arguments:

| Argument | Description |
|---|---|
| `--base_dir` | Master backtest directory. |
| `--fasta_file` | Master FASTA file used for data generation. |
| `--test_sample_index` | Index of the test sample to process; defaults to `0`. |

Example:

```bash
python predict_TrLs_clade.py \
    --base_dir backtest_seasonal_h3n2 \
    --fasta_file h3n2_clean.fasta \
    --test_sample_index 0
```

The script contains the inference and prediction-generation procedures. It expects the appropriate backtest inputs and model artefacts to be available in the configured directory structure.

### 3. Evaluate BLOSUM62-based distances and positional accuracy

Use:

```bash
python evaluation_new.py \
    --base_dir backtest_seasonal_h3n2 \
    --fasta_file h3n2_clean.fasta \
    --preds_dir introspection_out \
    --vaccine_dir Vaccine_strains
```

The evaluation script accepts:

- `--base_dir`: master backtest directory.
- `--fasta_file`: master HA FASTA file.
- `--zone_idx`: geographical-zone field index in the FASTA header.
- `--preds_dir`: directory containing prediction CSV files.
- `--vaccine_dir`: directory containing seasonal vaccine-strain FASTA files.

The script evaluates saved predictions using positional accuracy and BLOSUM62-based sequence distances.

### 4. Evaluate Hamming and Levenshtein distances

Use:

```bash
python evaluation_hamming.py \
    --base_dir backtest_seasonal_h3n2 \
    --fasta_file h3n2_clean.fasta \
    --vaccine_dir Vaccine_strains
```

The script evaluates sequence predictions using Hamming and Levenshtein distances, positional accuracy and reference comparisons.

The exact expected directory layout and prediction file naming conventions should be inspected in the relevant script before running it on a new dataset.

---

## Prediction evaluation

The evaluation framework compares predicted HA sequences with observed sequences from subsequent periods.

It includes complementary measures of sequence similarity and positional agreement.

### Sequence-distance measures

**Hamming distance**

Counts the number of positions at which two aligned sequences differ. It measures the number of amino-acid mismatches without distinguishing between conservative and non-conservative substitutions.

**Levenshtein distance**

Measures the minimum number of insertions, deletions and substitutions required to transform one sequence into another.

**BLOSUM62-based distance**

Uses the BLOSUM62 substitution matrix to account for differences in amino-acid substitution scores. The implementation converts substitution scores into a distance-like quantity, with lower values representing greater sequence similarity.

### Position-specific evaluation

The evaluation scripts distinguish between:

- **All positions:** the complete evaluated HA sequence.
- **Epitope positions:** positions belonging to the predefined HA antigenic sites A–E.
- **Shifted positions:** positions identified as having undergone a change according to the relevant historical and target-period comparison.

This separation allows model performance to be examined both globally and at positions of particular biological interest.

### Accuracy measures

The scripts also calculate positional agreement between predicted and observed amino acids.

Where applicable, evaluation includes population-weighted positional accuracy and comparisons with reference consensus sequences and seasonal vaccine strains.

### Statistical comparisons

The statistical-analysis notebooks compare model performance across seasons and evaluation metrics.

The BLOSUM62- and Hamming-based analyses use paired Wilcoxon signed-rank tests where the corresponding paired observations are available. They also produce seasonal summaries and statistical-significance tables.

---

## Exploratory and downstream analyses

### Evolutionary drift and consensus comparisons

`Distances_with_train_consensus.ipynb` investigates sequence divergence relative to consensus sequences constructed from historical training data.

The analyses include:

- Construction of consensus sequences from historical periods.
- Comparisons with target-season sequences.
- Unique-sequence analyses within geographical zones.
- Full-sequence and epitope-specific distance calculations.
- Geographical comparisons of evolutionary divergence.
- Positional agreement and sequence-match analyses.

### Antigenic distance and model comparisons

`antigenic_distance.ipynb` collects predicted FASTA sequences and compares them with consensus references.

It generates model-level and seasonal comparisons using the available prediction and evaluation outputs.

### HI titre and sequence-distance relationships

`Distance_vs_titre.ipynb` and `Distance_vs_titre_hamming.ipynb` investigate relationships between HI titres and sequence divergence.

The analyses include:

- Parsing HI titre records.
- Handling censored and non-standard titre values.
- Matching virus and serum isolates to sequence metadata.
- Identifying unmatched isolates.
- Calculating full-sequence and epitope-specific distances.
- Exploring the relationship between sequence divergence and HI titre categories.

The two notebooks use different sequence-distance approaches and should not be treated as identical analyses.

### Seasonal mutation analysis

`mutations.ipynb` investigates amino-acid substitutions and consensus changes across influenza seasons.

The analyses include:

- Parsing HA sequences and assigning seasons.
- Calculating amino-acid frequency changes.
- Identifying seasonal consensus substitutions.
- Mapping changes to HA positions.
- Comparing substitutions with recognised antigenic sites.
- Investigating changes over selected temporal windows.

---

## Interpretability analyses

`interpretability_figures.ipynb` contains analyses of selected model-generated outputs and evolutionary features.

The notebook includes visualisation and processing procedures for:

**Position attention**

Analysis of attention-related matrices to investigate how the model distributes attention across sequence positions.

**Geographical and clade-related attention**

Examination of attention outputs associated with transmission zones and viral clades.

**Zone and clade divergence**

Hierarchical clustering and visualisation of divergence matrices across geographical and clade categories.

**Epistasis-related patterns**

Analysis and visualisation of amino-acid position pairs identified through the model's epistasis-related procedures.

**Mutation characteristics**

Additional analyses investigate historical amino-acid distributions, consensus changes, conservative versus non-conservative substitutions and their relationship with antigenic sites.

These analyses provide exploratory views of model behaviour and sequence evolution. Attention weights and statistical associations should not automatically be interpreted as causal explanations of viral evolution or antigenicity.

---

## Outputs

Depending on the selected workflow and input data, the scripts and notebooks can generate:

- Predicted HA sequences in FASTA format.
- Consensus and reference sequences.
- Position-specific amino-acid summaries.
- Sequence-distance tables.
- Positional accuracy tables.
- Seasonal model-comparison summaries.
- Statistical-significance summaries.
- Mutation and consensus-change tables.
- Attention and divergence matrices.
- Epistasis-related tables.
- Figures illustrating model architecture, evolutionary drift and model performance.

The precise output names and locations depend on the script configuration and the input directory structure.

---

## Reproducibility and limitations

The repository is intended to document and support an ongoing research project. Some important considerations apply when using or reproducing the analyses.

1. **External datasets are required.** The master FASTA file, HI titre data, vaccine reference sequences and model-generated prediction files are not distributed with this repository.

2. **Directory structures matter.** Several scripts and notebooks reference specific filenames and folder structures. These may need to be adapted to the location of the user's own data.

3. **The environment is not yet pinned.** Package versions and hardware requirements should be matched to the original computational environment before attempting exact reproduction.

4. **The analyses are not all packaged as a single pipeline.** Some notebooks contain exploratory analyses, intermediate processing and alternative implementations. They may require outputs generated in earlier stages.

5. **Forecasting performance is not equivalent to antigenic prediction.** Sequence similarity, positional accuracy and overlap with recognised epitope sites are useful computational measures, but they do not directly establish antigenic similarity or predict experimental HI titres.

6. **Interpretability outputs require careful interpretation.** Attention, mutual-information and divergence analyses provide information about model behaviour and sequence associations, not necessarily causal biological mechanisms.

7. **Independent evaluation is important.** Temporal separation between training and evaluation data, control of sampling biases and avoidance of information leakage are essential when interpreting forecasting performance.

For research use, users should document the dataset version, preprocessing choices, target seasons, model configuration, random seeds and evaluation protocol associated with each experiment.

---

## Citation and contact

The code is associated with an ongoing research project on machine-learning-based forecasting of influenza A(H3N2) haemagglutinin evolution.

**Manuscript:** In preparation.

If you use or adapt this code in your research, please contact the repository author for citation details and information about the associated manuscript.

**Author:** Marika D'Avanzo & Aung Pone Myint
**Repository:** https://github.com/marikadavanzo/fluML

For questions about the modelling framework, data processing or evaluation procedures, please open a GitHub issue or contact the author.

---

## Acknowledgements

This work builds on publicly available influenza sequence resources, protein-language-model representations and established bioinformatics and machine-learning tools. Users should acknowledge the original data providers, software packages and pretrained models according to their respective terms and citation requirements.
