# What the comparison measures

The target is difficulty **within a competition**, derived from the human
ratings in [Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench).
An AMC8 problem can be advanced relative to AMC8 while an AIME problem is routine
relative to AIME. Both the problem and competition name are available at serving
time. Year, question number, rating, answer and solution are excluded from inputs.

`data.py` pins the source revision and hashes, builds the labels, identifies
duplicate families and writes the split audit. A label is above/below the median
rating for that competition, fitted only on the eligible pre-2018 source
population. Thus 64–512 is the number of **classifier training labels**, not the
total annotation cost of defining this benchmark. Validation and calibration
labels are additional supervision.

Rebuild the exact dataset from the public source with:

```sh
uv run --frozen --group data python data.py \
  --output run-output/rebuilt-data --previously-used evidence/prior-used-ids.json
```

The regenerated `data.json` must have canonical digest
`e5537fef5fee3523b3ebd485fe10684a417a9ece86d795dbb032621883eeaac1`,
reported by the command. `prior-used-ids.json` records earlier development
exposure so those families cannot be called a fresh test.

| Partition | Years | Problems | Role |
| --- | --- | ---: | --- |
| Training | Through 2017 | 512 | Nested budgets of 64, 128, 256, 512 |
| Validation | 2018 | 247 | Ridge regularization and native checkpoint selection |
| Calibration | 2019 | 246 | Decision cutoff, after model selection |
| Test | 2020 onward | 479 | Frozen comparison only |

Canonical text, five-token overlap, and number-masked overlap form connected
duplicate families. Families spanning chronological partitions are removed.
Only one representative per retained family is kept, selected by a fixed hash.
Test families containing any example used in the earlier experiments are removed.
Contest papers cannot span partitions. This removes detected leakage; it does
not establish that paraphrases are all detected or that foundation-model
pretraining excluded these public mathematics problems.

The sampling/target protocol was fixed before new test predictions. All eight
embedding policies and both native training families at seeds 89, 137 and 211
are reported. There is no test-based choice of seed, family, threshold or budget.
The test is now a public regression benchmark; further development should use a
new prospective test population for a new generalization claim.

## Models and selection

**kNN:** frozen Qwen3-Embedding-4B produces a normalized 2,560-dimensional vector.
The five most similar training vectors vote. More labels grow the bank; encoder
weights never change.

**Ridge:** the same vector feeds 2,560 learned weights and a bias. There are no
hidden layers. Training minimizes squared error with L2 regularization;
validation MSE chooses among 0.01, 0.1, 1 and 10. Calibration chooses the cutoff
by balanced accuracy. Scores can fall outside [0, 1] and are not probabilities.

Growing the ridge training set does not add inference calls: serving still uses
one embedding and 2,561 local coefficients. Feedback collection and retraining
are separate, one-time costs. kNN instead keeps a larger example bank. Neither
approach changes the embedding model's per-request computation as labels grow.

**Laya:** a 421M-parameter ModernBERT-large encoder plus a native decision head
scores a typed two-option question without generating text. Full fine-tuning
updates the complete checkpoint. LoRA trains rank-16 adapters on encoder
attention `Wqkv`/`Wo` **and the entire decision head**: 29.3M trainable parameters,
including 2.75M encoder adapter parameters. This is not head-only training.
Adapters are merged into an ordinary Laya checkpoint for serving.

Every native run uses eight epochs and the same data order seed, effective batch
16, 512-token input limit and 256-token question budget. Validation cross-entropy
selects an epoch; a separate calibration partition selects the cutoff by balanced
accuracy on the serialized checkpoint. Full encoder/head learning rates are
2.5e-5/1e-4; LoRA uses 2e-4/1e-4.
The comparison evaluates these declared recipes, not the best possible fully
tuned recipe for each architecture. Three seeds are a stability check, not an
exhaustive hyperparameter search.

## Static and improving versions

Each deployed policy is static: its bank or weights and its cutoff are frozen.
An update collects reviewed examples, retrains on training data, selects using
development data, freezes a new version, and evaluates it. Improvement is a
measured outcome, not an automatic property of adding examples. The nested
budgets illustrate that process without changing the test population.

If human labels are unavailable, a teacher can supply candidate labels, but
teacher agreement alone does not establish correctness. Keep a separately
human-reviewed validation/test sample and measure teacher errors before replacing
the benchmark's labels. No teacher or GEPA supplies labels in this example.

Accuracy and balanced accuracy are reported alongside false-routine and
false-advanced counts. Paired case-bootstrap intervals condition on the trained
models; seed ranges describe variation across the three training runs. Problems
within a contest paper can be correlated; the report also includes a descriptive
paired bootstrap over the 11 competition-year groups. None of these results translates
directly into downstream answer quality or dollars saved: that requires paired
cheap/strong answers and measured serving charges on the target workload.
