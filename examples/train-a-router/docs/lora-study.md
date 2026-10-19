# LoRA follow-up: select a recipe, then add training examples

This optional study tests the capacity and regularization of the native Laya
router, then increases its training data. It uses the same competition-relative
human difficulty labels as the main example. The [original GPU lab](finetuning.md)
and its results remain a separate baseline.

The useful change was adding clean examples: the same recipe improves from
**57.34% to 65.00% mean accuracy**, or **17.9% fewer errors**, when training grows
from 512 to 2,581 examples. All three declared seeds improve. The paired 95%
interval for the gain is **+3.20 to +12.04 percentage points**. This follow-up
uses the already-published 479-problem regression benchmark. Qwen4B plus ridge
still scores **72.03%** with 512 examples and remains the main recommendation;
this extension demonstrates learning a compact native decision model. See
[results](results.md) for all seeds, comparisons and limitations.

Run every command below from `examples/train-a-router`. The recorded comparison
can be replayed on CPU. Training and fresh model evaluation require the optional
training dependencies and a compatible CUDA GPU. An SIE Cloud inference key does
not provision a training job.

Training and direct `finetune.py` evaluation can start slowly on a fresh CUDA
machine. Transformers 4.57.6 can implicitly compile ModernBERT's embedding and
MLP blocks when Triton is available, even without selecting a Laya compilation
backend. Direct evaluation prints its first result after the scoring pass and
up to 35 single-example timing calls. The guarded `lora_evaluate.py` command
below disables compilation and checks development-score parity first. Storage
reads and dependency imports can also delay startup.

## Replay the recorded comparison

```sh
uv sync --frozen
uv run --frozen python run.py fetch
uv run --frozen python run.py fetch --lora-study
uv run --frozen python lora_report.py \
  --study run-output/lora-study --evidence evidence \
  --output run-output/lora-report.json
```

The first download contains the original experiment in `evidence/`. The second
contains the follow-up in `run-output/lora-study/`: the declared plan, every
candidate's learning curves and selection records, the training-expansion audit,
and the recorded evaluations. This adds about **6 MB** without model weights.
Both downloads check immutable manifest and file
hashes. To choose another destination, pass `fetch --lora-study --evidence` with
that directory. A directory containing another experiment's manifest is rejected
before download.

`lora_report.py` recomputes validation selection, checks the frozen plans and
calibration cutoffs, and recomputes the comparison from recorded per-problem
scores. It needs neither weights nor a GPU. Its stages separate the original
cross-entropy selection, AUC selection with the original recipe, recipe choice
at the same training size, and the increase in training data. Reported seeds are
averaged before resampling problems or competition-year groups; three seeds do
not create three times as many independent test cases.

Generate the follow-up figure from the same verified records:

```sh
uv run --frozen --group plots python plot.py \
  --lora-study run-output/lora-study --evidence evidence \
  --output run-output/lora-figures
```

This writes `lora-study.png` and `lora-study.svg`: validation AUC for every recipe,
then regression accuracy for each comparison stage. Both panels show every seed
and the mean across seeds; the regression panel includes the unchanged ridge
reference. The figure recomputes the strict report rather than reading a saved
`report.json`. Use a new output directory for another rendering. Running
`plot.py` without `--lora-study` continues to generate the original baseline
figures.

To fetch the expanded seed-89 native checkpoint as well:

```sh
uv run --frozen python run.py fetch --lora-study --weights lora
```

This downloads the merged FP32 checkpoint under
`run-output/lora-study/expanded/seed-89/lora/`. Seed 89 is a fixed serving example,
not a seed chosen by test performance. The follow-up fetch accepts only
`--weights lora`; the original `fetch --weights full`, `lora`, and `both` continue
to refer to the original experiment.

## Serve the downloaded router locally

After fetching the follow-up weights, configure and start SIE:

```sh
uv run --frozen python decision.py configure \
  --run run-output/lora-study/expanded/seed-89 --variant lora \
  --models-dir run-output/lora-study-models
SIE_IPC_SOCKET_PATH=/tmp/sie-example-lora-study.sock \
uvx --from 'sie-server[local]==0.10.0' sie-server serve \
  --host 127.0.0.1 --port 8100 --models-dir run-output/lora-study-models \
  --models example/laya-difficulty-lora --preload example/laya-difficulty-lora
```

Wait for the model to load. In another terminal, from the same example folder:

```sh
export SIE_BASE_URL=http://127.0.0.1:8100
unset SIE_API_KEY
uv run --frozen python decision.py route \
  --run run-output/lora-study/expanded/seed-89 --variant lora --contest AMC10 \
  --request 'A rectangle has perimeter 30 and length 9. What is its area?'
uv run --frozen python check_serving.py \
  --run run-output/lora-study/expanded/seed-89 --variant lora \
  --evaluation evidence/test.json --limit 32 \
  --output run-output/lora-study-serving.json
```

The check compares 32 decisions from this checkpoint with its recorded GPU
evaluation and reports any mismatch. It is a serving consistency check, not a
new accuracy measurement. The historical example's serving receipts apply to
its own checkpoints. Stop the server with Ctrl-C. Custom native checkpoints run
on your own SIE server; the [Cloud exercise](../README.md) uses the public Qwen
embedding model.

## Reconstruct the expanded training data

Rebuild from the pinned public Parquet source rather than constructing new
training data from model predictions:

```sh
uv run --frozen --group data python data.py \
  --source run-output/lora-source \
  --output run-output/lora-rebuilt \
  --previously-used evidence/prior-used-ids.json
uv run --frozen --group data python expand_training.py \
  --source run-output/lora-source --evidence evidence \
  --output run-output/lora-expansion
```

The expansion uses all 2,581 eligible pre-2018 duplicate-family representatives.
Its first 512 rows must exactly match the original training file. It preserves
the original human-rating thresholds and removes families that cross temporal
partitions. Validation, calibration, and test populations stay unchanged.
`training-expansion-audit.json` records the source hashes, row identities,
family identities, and generated-file hash.

If the published follow-up was fetched above, compare the regenerated files with
the archived expansion:

```sh
cmp run-output/lora-expansion/train-expanded.jsonl \
  run-output/lora-study/expansion/train-expanded.jsonl
cmp run-output/lora-expansion/training-expansion-audit.json \
  run-output/lora-study/expansion/training-expansion-audit.json
```

Create a directory containing only the four development inputs. On a remote
training machine, transfer this directory; keep the test inputs for the later
evaluation step.

```sh
mkdir run-output/lora-expanded-inputs
cp evidence/validation.jsonl evidence/calibration.jsonl evidence/questions.json \
  run-output/lora-expanded-inputs/
cp run-output/lora-expansion/train-expanded.jsonl \
  run-output/lora-expanded-inputs/train.jsonl
```

## Train one router with the selected recipe

The development screen selected `original`. Once the expanded input directory
above is ready, one command trains a router with those settings:

```sh
uv sync --frozen --group training
uv run --frozen --group training python finetune.py train \
  --data run-output/lora-expanded-inputs --output run-output/my-router \
  --seed 89 --recipe original --variants lora \
  --validation-metric auc --export-dtype float32 --device cuda
```

This trains rank-16 attention adapters and the native decision head for eight
epochs, retains the checkpoint with the best validation AUC, and fits a cutoff
on separate calibration examples. The merged checkpoint is in
`run-output/my-router/lora/`; its selection and cutoff are in `frozen.json`.
Serve it by passing `--run run-output/my-router --variant lora` to the
[local serving commands](serving.md).

After `frozen.json` exists, evaluate that fixed model:

```sh
uv run --frozen --group training python finetune.py evaluate \
  --evaluation evidence/test.json --questions evidence/questions.json \
  --output run-output/my-router --device cuda
```

Seed 89 is the fixed demonstration seed. The published comparison includes all
three declared seeds. For a new application, reserve its own evaluation data
before training and calibrate the route against the outcome you need to predict.

## Reproduce the complete recipe comparison

The full comparison also needs the original 512-example development directory:

```sh
mkdir run-output/lora-small-inputs
cp evidence/train.jsonl evidence/validation.jsonl evidence/calibration.jsonl \
  evidence/questions.json run-output/lora-small-inputs/
uv sync --frozen --group training
uv run --frozen --group training python lora_study.py \
  --data run-output/lora-small-inputs \
  --expanded-data run-output/lora-expanded-inputs \
  --output run-output/my-lora-study --device cuda \
  --timeout-seconds 14400
```

The command sets a four-hour wall-clock limit; choose a limit appropriate to
your hardware before starting. It retains 21 native FP32 checkpoints, requiring
tens of GB of disk space. The output directory must be new. An interrupted study
preserves partial records but does not resume in place; use a fresh output for a
new complete run.

Every candidate starts from the same pinned Laya checkpoint and uses eight
epochs, effective batch size 16, a 512-token input limit, and a 256-token question
budget. Training uses supervised cross-entropy, AdamW with cosine learning-rate
decay, and shuffled choice-option order with aligned targets. The exact recipes
are declared in `finetune.py`:

| Recipe | Encoder attention layers adapted | Rank / alpha | Adapter / head learning rates | Native head updates | Adapter dropout / weight decay / smoothing |
| --- | --- | --- | --- | --- | --- |
| `original` | All | 16 / 32 | 2e-4 / 1e-4 | Entire head | .05 / .01 / 0 |
| `slow` | All | 16 / 32 | 5e-5 / 1e-5 | Entire head | .05 / .01 / 0 |
| `regularized` | All | 8 / 16 | 5e-5 / 1e-5 | Entire head | .1 / .1 / .1 |
| `frozen-head` | All | 8 / 16 | 1e-4 / none | Frozen | .1 / .1 / .1 |
| `scorer-only` | All | 8 / 16 | 1e-4 / 1e-5 | Scorer only | .1 / .1 / .1 |
| `top-layers` | Layers 24–27 | 16 / 32 | 5e-5 / 1e-5 | Entire head | .05 / .01 / 0 |

Adapters target `attn.Wqkv` and `attn.Wo`; no candidate adapts encoder MLP
layers. `scorer-only` still trains the encoder adapters. The five new recipes
freeze the unused action head. Label smoothing applies only to training targets.
Native-head dropout remains active during training, including in frozen-head
runs; evaluation disables dropout. The [source review](laya-adaptation.md)
explains the motivation and differences from public Laya adaptation recipes.

Selection has three separate steps:

1. Train each of the six recipes on 512 examples for seeds **89, 137, and 211**.
   For each run, choose the maximum ROC AUC on the 2018 validation set. Epoch zero
   is eligible; ties retain the earliest epoch. Cross-entropy and training loss
   remain diagnostics.
2. Choose the recipe with the highest mean selected validation AUC across all
   three seeds. Ties retain recipe declaration order, shown in the table. Write
   `recipe-frozen.json` before starting the larger training runs.
3. Train that same recipe on 2,581 examples for all three seeds, with unchanged
   learning settings. For every selected checkpoint, fit its cutoff by balanced
   accuracy on the separate 2019 calibration set. Calibration does not choose the
   recipe or epoch. Write `all-frozen.json` when every run is complete.

The 18 screening runs plus three expanded runs are all retained. No best seed
is selected. The driver checks that both training directories have identical
validation, calibration, and question files, and that the smaller training file
is the exact prefix of the expanded one. It never reads test inputs.

Keeping eight epochs also increases the training budget: 512 examples produce
256 optimizer updates, while 2,581 produce 1,296. This compares the practical
recipe of adding examples and training for the same number of passes; it does
not isolate dataset size at fixed compute.

## Why this study exports FP32 weights

The original Laya serializer writes half-precision weights. In the first
follow-up attempt, that conversion changed validation logit margins beyond the
declared export tolerance. Before recipe selection or test evaluation, the
study was amended to preserve FP32 weights for every candidate. The initial
attempt is retained as an audit; the training recipes and data did not change.

The study passes `--export-dtype float32` to `finetune.py`. This changes the
saved checkpoint precision, not the training objective or native architecture.
Encoder adapters are merged, validation margins are checked before and after
merge and export, and calibration uses the reloaded exported checkpoint. All
follow-up candidates use the same export rule. The original baseline keeps its
original export and selection protocol. Consequently, the historical CE baseline
versus the new original-recipe AUC control changes both checkpoint selection and
export precision. That comparison cannot attribute a difference to selection
alone. Recipe and training-size comparisons within the new study all use FP32.

## Evaluate only after the study is frozen

Wait for `ALL_LORA_RUNS_FROZEN` and `run-output/my-lora-study/all-frozen.json`.
Then make the original evidence available on the evaluation machine and run:

```sh
uv run --frozen --group training python lora_evaluate.py \
  --study run-output/my-lora-study --evidence evidence \
  --device cuda --timeout-seconds 1800
uv run --frozen python lora_report.py \
  --study run-output/my-lora-study --evidence evidence \
  --output run-output/my-lora-report.json
```

The evaluator checks the completed study and development selections, then sets
`TORCH_COMPILE_DISABLE=1` and `TORCHDYNAMO_DISABLE=1` before loading any model.
Each frozen checkpoint must reproduce its selected validation scores and saved
calibration scores within a maximum absolute margin difference of **0.03**, with
**zero changed decisions** at its existing cutoff on both splits. These checks
run in isolated processes under the same overall timeout. The declared checks,
per-checkpoint scores, and completion hashes are saved in `runtime-parity`
receipts. A failed check stops the run before test inputs are opened.

In the recorded run, all six checkpoints passed: maximum development-margin
drift was 0.000174, with no changed decisions. The first evaluation attempt had
produced a plan but no outcome files during a prolonged startup. That plan and
the runtime amendment are retained in the public audit. The startup cause was
not localized; disabling compilation did not eliminate the initial delay.

After every checkpoint passes, it evaluates the original recipe at 512 examples, the
selected recipe at 512, and the selected recipe at 2,581, each for all three
seeds. If the original recipe wins, its 512-example stage is evaluated once.
Existing parity and evaluation outputs are preserved; rerunning an interrupted
attempt in place is refused. The runtime change keeps weights and cutoffs fixed;
latency measurements from eager and compiled execution are different comparisons.
Fresh GPU evaluation needs the
trained weights; downloading only the published core records supports CPU report
replay instead.

The report compares these stages with the original CE-selected native runs and
the unchanged `ridge512` embedding reference. The published test is an **exposed
regression benchmark** in this follow-up. Development selection is separate from
its outcomes, but a new claim about future generalization requires a new test
population. Difficulty accuracy alone does not establish downstream answer
quality or monetary savings; those need paired solver outcomes and serving costs.
