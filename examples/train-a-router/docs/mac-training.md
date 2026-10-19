# Train Laya on Apple Silicon

An Apple Silicon Mac can perform the LoRA adaptation in the main walkthrough
using its GPU through [PyTorch MPS](https://docs.pytorch.org/docs/2.9/notes/mps.html).
Use the existing pinned PyTorch/Laya environment; no model conversion or second
training framework is needed. The Qwen embedding classifiers use CPU after their
vectors have been collected.

## Setup

From `examples/train-a-router`, install the optional training dependencies and
check that PyTorch can use your Apple GPU:

```sh
# Install training dependencies, verify the Apple GPU, and select MPS.
uv sync --frozen --group training
uv run --frozen --group training python -c \
  'import torch; assert torch.backends.mps.is_available(); print(torch.__version__, "MPS available")'
export ROUTER_DEVICE=mps
```

Follow the [main walkthrough](../README.md#prerequisites-and-shared-setup) with
`ROUTER_DEVICE=mps`: prepare the inputs, measure base Laya, train the 512-example
adapter, and then train the 2,581-example adapter. Every native training and
evaluation command accepts the same device setting. Use a new output directory
for each run. Several GB of disk space per run are needed for the base model,
selected training state and merged checkpoint, in addition to the download cache.

Keep the Mac connected to power. Training on smaller Apple GPUs will take longer
than the measured machine below. Available unified memory must also cover macOS
and other applications; GPU allocator measurements are not a machine RAM minimum.
For a CUDA machine, set `ROUTER_DEVICE=cuda` instead.

## Runtime settings

| Setting | Apple GPU (MPS) | CUDA reference |
| --- | --- | --- |
| Training precision | FP32 | Laya's CUDA autocast |
| Micro-batch / gradient accumulation | 1 / 16 | 8 / 2 |
| Effective batch | 16 | 16 |
| Gradient checkpointing | Disabled | Enabled |
| Evaluation batch | 4 | 16 |
| Selected checkpoint export | FP32 | FP32 |

The model, attention targets, rank, learning rates, eight epochs, validation
selection and calibration rules are unchanged. Different precision, batch
grouping and random-number streams can still change the selected epoch and final
accuracy. The three-seed reference results in the main headings are CUDA results.

The pinned Torch 2.9.1 checkpoint implementation does not restore MPS dropout
state correctly in the tested non-reentrant path. A local probe reproduced the
same forward output but different gradients after checkpoint recomputation.
The MPS profile disables this optimization and uses smaller batches to bound
memory. It does not enable CPU fallback or change the checkpoint precision.
The [checkpoint implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/checkpoint.py)
and [MPS RNG implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/mps/__init__.py)
provide the source context for that measured behavior.

An optional hardware regression checks outputs and gradients through the actual
MPS profile, without downloading a model:

```sh
# Check MPS forward outputs and gradients without downloading model weights.
uv run --frozen --group training python -m unittest discover \
  -s tests -p test_mps_training.py -v
```

## Validation scope

The validation machine is an **Apple M3 Max with 128 GiB unified memory**, running
macOS 26.2, Python 3.12.1, Laya 0.3.29, Torch 2.9.1, Transformers 4.57.6 and
PEFT 0.18.1. CPU fallback is unset. The model download was already cached for
the timing measurements.

The complete **512-example, eight-epoch** run used seed 89 and the `original`
recipe, with all 247 validation and 246 calibration examples. Training never
read test inputs. Validation AUC selected epoch five, and calibration set the
cutoff after export. A separate command then evaluated both frozen models on
all 479 test problems:

| Model | Test accuracy | Balanced accuracy | Errors | False routine / false advanced |
| --- | ---: | ---: | ---: | ---: |
| Calibrated base Laya | 53.65% | 54.88% | 222 | 49 / 173 |
| Mac-trained LoRA + native head | 57.62% | 57.95% | 203 | 81 / 122 |

This is one Mac seed, reported separately from the three-seed CUDA study.
Its accuracy gain is 3.97 percentage points; the error mix also changes, so
aggregate accuracy does not imply fewer errors of each type.

- **Elapsed time:** 10.3 minutes for base calibration plus the complete LoRA
  training/export/calibration pipeline; 40 seconds for separate test evaluation.
- **Sampled peak MPS driver allocation:** 9.1 GiB. Peak process RSS was 8.8 GiB;
  these measurements overlap and must not be added. Sampling every 100 ms can
  miss short peaks. Total system memory use was not measured.
- **Merge/export consistency:** maximum validation-margin drift 0.000351 and
  zero changed validation decisions at the final cutoff.
- **Gradient regression:** the optional MPS test above passed on this machine.
- **Local SIE serving:** all 32 fixed-prefix decisions matched this checkpoint's
  own frozen evaluation through SIE 0.10.0 on MPS.

The [published Mac run records](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/d9165fc1f19a57f3ea901a37aba2a26f7fe6b5e3/evaluations/2026-10-19/train-a-router-mps)
include every epoch, selection and calibration records, per-problem test scores,
serving checks, memory measurements and the executed training source. They total
about 258 KB, excluding model weights. The pinned manifest SHA-256 is
`0ddd5ad56cbd52b99b9118f4b7f81302dd3efcd0115ce5faa494cce0aaf83750`.

The larger 2,581-example training run and three-seed comparison were measured on
CUDA. Their MPS commands use the same tested runtime profile, but a complete
2,581-example Mac run and lower-memory Macs have not been measured here. The Mac
timing and accuracy above describe the single 512-example run on this machine.

A 32-example, eight-epoch smoke run completed training, validation selection,
adapter merge, FP32 export, checkpoint reload and calibration in **108 seconds**.
Export changed no validation decisions; maximum logit-margin drift was
0.000043. Its sampled peak MPS driver allocation was **7.0 GiB**. This smoke run
checks the training path, not classification accuracy.
