# SmolVLM Vision Challenge

Parameter-efficient fine-tuning of `SmolVLM-500M-Instruct` for scientific visual multiple-choice reasoning, built for the [*Pixels to Predictions: DL Vision Challenge*](https://www.kaggle.com/competitions/pixels-to-predictions/overview) on a ScienceQA-derived dataset.

### Authors

- Helin Wang &nbsp;(NetID: `hw4103`)
- Sichen Li &nbsp;(NetID: `sl12693`)

## Method overview

We adapt the official `SmolVLM-500M-Instruct` backbone with a parameter-efficient pipeline built around four design choices.

1. **LoRA + DoRA adapters** on attention (`q_proj`, `v_proj`) **and** MLP (`gate_proj`, `up_proj`, `down_proj`) projections of the language model. Rank `r=4`, `alpha=8`, `dropout=0.05`. Total trainable parameters: 2,162,688 (≈2.16M, 0.42% of the 509M base model).
2. **Answer-only supervision.** During fine-tuning the prompt prefix is masked with `-100`, so cross-entropy loss is applied **only** to the final answer-letter token, not to the entire input.
3. **Length-normalized log-likelihood inference.** For each candidate letter `c` we forward `prefix + " {c}"` and pick the `argmax` of the per-token mean log-probability: `s_c = (1/|T_c|) Σ_t log p_θ(t | prompt, image, t_<)`.
4. **Metadata-aware prompt.** `subject` / `grade` / `topic` metadata and `lecture` / `hint` context are optionally injected into a structured prompt with an explicit `"The correct answer is:"` answer prefix, which constrains the model to a single-letter output that is trivially compatible with the log-likelihood scoring above.

Training uses fp16 mixed precision, gradient checkpointing, batch size 1 with gradient accumulation 8 (effective batch 8), learning rate `2e-4`, and 2 epochs (778 optimizer steps total) on a single NVIDIA A100-40GB (≈3.8 hours end-to-end).

## Repository layout

```
.
├── README.md                          this file
└── smolvlm-vision-challenge.ipynb     main notebook: training + eval + submission
```

## Setup

### Install dependencies

```bash
pip install -q transformers==4.57.6 peft==0.18.1 bitsandbytes accelerate datasets pillow
```

The notebook uses these exact pinned versions.

### Data layout

Place the competition data so that `DATA_DIR / image_path` resolves correctly, e.g.:

```
data/
├── train.csv
├── val.csv
├── test.csv
└── images/
    ├── train/train_00000.png
    ├── val/val_00000.png
    └── test/test_00000.png
```

The notebook's `DATA_DIR` is set to a Google Drive path for Colab. **For local runs, change the line in the imports cell of [`smolvlm-vision-challenge.ipynb`](smolvlm-vision-challenge.ipynb):**

```python
DATA_DIR = Path("data")  # was: Path("/content/drive/MyDrive/final_dl")
```

---

## How to run

The full pipeline lives in [`smolvlm-vision-challenge.ipynb`](smolvlm-vision-challenge.ipynb).

Execute the cells top-to-bottom:

1. **Install + Imports.** Install dependencies, optionally mount Google Drive, set `DATA_DIR` and `IMG_SIZE = 384`.
2. **Load data.** Read the three CSVs and parse the `choices` JSON column.
3. **Prompt template.** Toggle `USE_METADATA_IN_PROMPT` to switch between metadata-on and metadata-off ablations.
4. **Load processor + model.** SmolVLM-500M-Instruct in fp16 with gradient checkpointing and `use_cache=False`.
5. **Attach LoRA/DoRA adapters.** `use_dora=True` enables DoRA; flip to `False` for the LoRA-only ablation. Trainable parameter count is checked against the 5M budget.
6. **Train.** 2 epochs. The trained adapter is saved to `./smolvlm_lora_out/final` and copied to Drive.
7. **Reload model + evaluate on val.** Runs `score_choices_by_loglikelihood` over the 1,048-example validation split and prints overall accuracy plus the prediction distribution.
8. **Error analysis.** Per-`num_choices` / `subject` / `grade` / `topic` / `category` / `skill` accuracy, confusion matrix, and high-confidence mistakes ranked by score margin.
9. **Generate submission.** Writes `submission.csv` for the test split using the exact same scoring routine as validation.

### Configuration switches (used for ablations)

| Switch | Where | Final value | Purpose |
|---|---|---|---|
| `IMG_SIZE` | imports cell | `384` | Image side length (starter used `224`) |
| `USE_METADATA_IN_PROMPT` | prompt cell | `True` | Add `Subject` / `Grade` / `Topic` block to the prompt |
| `USE_DORA` | LoRA config cell | `True` | Apply DoRA decomposition on top of LoRA |
| Prefix masking in `VQATrainDataset` | dataset cell | enabled | Set `labels[:prefix_len] = -100` for answer-only loss |

---

## Results

**Main validation accuracy** (top-1, 1,048 examples):

| Setting | Ans-only | Metadata | Val Acc |
|---|---|---|---|
| Baseline (zero-shot)  | No  | No  | 0.6355 |
| + Answer-only loss    | Yes | No  | 0.7777 |
| + Metadata in prompt  | Yes | Yes | 0.7834 |
| + DoRA (final)        | Yes | Yes | **0.7882** |

The dominant gain comes from **answer-only supervision** (+14.2 pts over the zero-shot baseline). Adding metadata gives a smaller but consistent improvement (+0.6 pts), and switching from LoRA to DoRA contributes another +0.5 pts.

### Error analysis highlights

- **By number of choices**
  - 2-choice 0.840
  - 3-choice 0.770
  - 4-choice 0.857
  - **5-choice 0.318**
    - 5-choice questions are by far the dominant failure mode.
- **By subject**
  - social science 0.901 ≫ natural science 0.759 ≫ language science 0.607.
- **Letter prediction bias**
  - the model strongly prefers A/B
  - predicted letter E only once across the entire val split (true count = 7), and every E-labeled example is mispredicted.
- **Hardest skills**
  - *Compare ages of fossils in a rock sequence* (0.125)
  - *Use Punnett squares to calculate ratios of offspring* (0.238)
  - *Identify and compare air masses* (0.300)

See Section 5 of the report for the full breakdown, including the training loss curve, the confusion matrix, and per-topic accuracy charts.
