# ODSC West: build a router that learns from feedback

**Outcome:** route competition-math problems to routine or advanced handling,
then improve that policy with labeled feedback. Allow 45–60 minutes. Bring Git
and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Everything lives in `examples/request-routing/`; larger data comes from Hugging Face.

## 1. Download and learn on your laptop — no key or GPU

```sh
git clone https://github.com/superlinked/sie.git
cd sie/examples/request-routing/cost
uv sync --frozen
uv run --frozen python workshop.py download
uv run --frozen python workshop.py learn --labels 64 --output run-output/policy64.json
uv run --frozen python workshop.py evaluate --policy run-output/policy64.json
uv run --frozen python workshop.py learn --labels 512 --output run-output/policy512.json
uv run --frozen python workshop.py evaluate --policy run-output/policy512.json
```

You fit 2,560 weights and one bias on frozen **Qwen3-Embedding-4B** vectors.
The encoder is unchanged. Regularization and the routing cutoff use training
leave-one-out predictions; the test set never selects them. The public
Easy2Hard-AMC labels derive from human contest performance. The split has
512 training, 128 validation and 512 test problems.

| Policy | Test accuracy | Errors / 512 |
| --- | ---: | ---: |
| Frozen nearest neighbors, 64 labels | 71.68% | 145 |
| Learned classifier, 64 labels | 79.49% | 105 |
| Learned classifier, 512 labels | **89.65%** | **53** |

The full update cuts errors **63%** versus nearest neighbors; adding feedback to
the same classifier cuts errors **50%**. The main accuracy gain is +17.97 points
(paired 95% interval +14.06 to +22.07). A fresh 512-case confirmation set reproduces the gain: **74.22% → 91.60%**,
with **67% fewer errors**. CPU fitting takes well under a second. This measures
difficulty-routing accuracy, not LLM answer quality.

## 2. Use the policy you trained

Recorded request, zero inference calls:

```sh
uv run --frozen python workshop.py route --policy run-output/policy512.json --recorded-id e2h-amc-eval-1602
```

Create an account at [SIE Console](https://console.superlinked.com), check your
available starter credits and create an API key. Then:

```sh
unset SIE_BASE_URL
export SIE_API_KEY=your-key
uv run --frozen python workshop.py route --policy run-output/policy512.json --request 'Find the number of integer solutions to x^2 + y^2 = 2025.'
```

This makes **one embedding call**, prints its actual credit receipt and applies
your local classifier. Our two single-request checks used 2 credits each.
More feedback adds no inference calls per routed request. The app receives a
route; calling a downstream answering model is a separate application step.
The [economic capstone](./cost/ECONOMICS.md) supplies that full route → answer
pipeline and measures answer correctness and actual credits independently.
For your own SIE server, use `SIE_BASE_URL` and `--self-hosted`; see the
[local-server instructions](./cost/README.md#run-your-own-sie-server).

## 3. Compare and extend

```sh
uv run --frozen python workshop.py compare
```

Compare static kNN, growing example banks, learned classifiers, fixed Qwen4B
reranking and GEPA prompt updates on the same test set. Try 128 and 256 labels
to reconstruct the learning curve. **GEPA did not improve this task**; it is an
optional optimization lab, not the source of the demonstrated gain.
The [post-training extension](./cost/POSTTRAIN.md) is a second successful learning
path: on fresh confirmation, a 421M-parameter Laya model improves **53.3% → 87.1%**
with full tuning, or **88.1%** with LoRA plus head training. Replay it without a GPU:

```sh
uv run --frozen python posttrain_evidence.py
uv run --frozen python replay_posttrain.py
```

Optional GPU instructions reproduce training and serve the released checkpoints
through local SIE. The embedding classifier remains the most accurate recipe.

The optional [teacher-feedback and cost lab](./cost/ECONOMICS.md) also runs end to
end, including real answers. Its primary saves **4.7%** with a **1.95-point answer
accuracy loss**; it did not meet the 25%-saving/1-point-loss target. Keep this
measured tradeoff separate from the successful difficulty-learning exercise.

**Deployment loop:** collect reviewed feedback → embed new examples once →
refit → validate → freeze weights and cutoff together → deploy the accepted policy.
[Results, attribution and pinned public evidence](./cost/DIFFICULTY_POLICY.md).
