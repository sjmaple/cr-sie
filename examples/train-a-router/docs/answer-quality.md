# Measure the savings and the answers people receive

A difficulty classifier is useful only if its decisions improve the application.
Here the application solves mathematics problems. We compare the final answers
and charges for **always Qwen 27B**, **always Qwen 4B**, and the five routers in
the [walkthrough](../README.md).

## Start with the recorded comparison

**CPU; no API key or inference charges.** Run these commands from
`examples/train-a-router`, after the walkthrough's shared setup. This is the
default way to inspect answer quality while you work through the router-training
stages; collecting another complete set of answers is optional.

```sh
# Download verified result files and recompute each router's answer quality and cost.
uv run --frozen python run.py fetch --solver-study
uv run --frozen python solver_policies.py report \
  --frozen run-output/solver-study/policies-frozen.json \
  --test-grades run-output/solver-study/grades-test.json \
  --router-credits run-output/solver-study/embedding-test/router-credits.json \
  --output run-output/my-answer-quality-report.json
```

The command prints tables with answer accuracy, the quality gap and money
saved, first for the original routing rules and then for the rules chosen with
a 1-point calibration target. A separate cost table includes the routing charge
where measured; `unpriced` means that hosting cost is still unknown.

It also saves `run-output/my-answer-quality-report.json`: `baselines` contains
`always_strong` and `always_cheap`; `stages` contains the five routers and all
four declared quality-loss targets. The report is recomputed from recorded
grades. To inspect the underlying model answers and individual API receipts:

```sh
# Add the optional raw answer recordings to the same verified download.
uv run --frozen python run.py fetch --solver-study --solver-records
tar -xzf run-output/solver-study/solver-records.tar.gz -C run-output/solver-study
tar -xzf run-output/solver-study/embedding-records.tar.gz -C run-output/solver-study
```

The raw downloads are compressed archives; the `tar` commands unpack the
hash-verified recordings into the same study folder. Use a new report filename
for a second run. Downloaded files remain in
`run-output/solver-study`; fresh experiments below use separate directories.

You can also repeat the answer grading on CPU. This writes a new file and
checks that it exactly matches the published grades:

```sh
# Regrade the recorded test answers without overwriting the published results.
uv run --frozen --group evaluation python solver.py grade \
  --run run-output/solver-study --split test --output run-output/my-regraded-test.json
cmp run-output/my-regraded-test.json run-output/solver-study/grades-test.json
```

`cmp` prints nothing when the files match. Regrading makes no model calls.

To draw the tradeoff, place savings on the horizontal axis and quality lost on
the vertical axis. Further right saves more; lower preserves more answer quality:

```sh
# Draw the two cost-versus-quality comparisons from your reproduced report.
uv run --frozen --group plots python plot_solver.py \
  --report run-output/my-answer-quality-report.json \
  --output run-output/my-answer-quality-figures
```

The output folder contains PNG and SVG figures. Their savings exclude router
hosting so that every design uses the same cost basis; the report tables also
show total costs where the routing charge was measured.

## What the comparison measures

For every problem, record one answer from each model. Keep their answers,
correctness grades, token usage and SIE charge receipts. A router then selects
which of these two answers the user would have received. Every router shares
the same recordings, so differences come from routing decisions rather than
another random generation or another round of charges.

- **Answer accuracy** is the fraction of correct final mathematical answers.
- **Quality gap** is always-strong accuracy minus routed accuracy, in percentage
  points. We also count lost correct answers and answers rescued by the smaller
  model. A negative gap is possible: the stronger model is not always right.
- **Answer-model cost** is the charge for the selected answer, including its
  final-answer turn when one was needed.
- **Total serving cost** adds the routing decision. Always-strong and
  always-cheap bypass the router and have no routing charge.
- **Savings** compare that cost with always-strong. When a router's hosting
  cost is unknown, the report gives answer-model savings and the maximum
  router cost that would still leave a saving; it does not assume free hosting.

For example, 90% answer accuracy at $10 per 1,000 requests versus 88% at $4
means a **2-point quality gap and 60% savings**. These numbers illustrate the
calculation; they are not benchmark results.

The Cloud embedding measurement uses **one request at a time within each API
call**, even when several calls run concurrently. The 479 test embeddings cost
597 credits: **$0.00597 total, or $0.01246 per 1,000 requests** at the recorded
$0.00001 per credit. Both embedding classifiers have the same routing overhead.
Batching multiple problems into one call has different billing rounding and is
not used as a substitute for this measurement.

## Select the routing rule before testing it

We report two comparisons:

1. **Original difficulty rules.** Keep each router's previously chosen cutoff.
   This tests whether improving its difficulty accuracy also improves the
   delivered answers and cost.
2. **Rules chosen for cost and answer quality.** Use the 246 calibration
   problems to choose the cheapest cutoff within a declared quality-loss
   budget. The primary budget is 1 percentage point; 0, 2 and 5 are reported
   as additional choices. A cutoff determines which scores send a problem
   to the stronger model. Always-strong is available when no cheaper rule
   satisfies the calibration target.

The selected rules and their input hashes are saved before collecting the
479 test answer pairs. Test reporting applies every saved rule. It does not
pick the best-looking test result or change the cutoff to hit the target.
A calibration target is an empirical check, not a promise about new requests.

The comparison uses the recorded seed-89 versions of the router recipes in the
walkthrough. Difficulty-training tables also show averages over three seeds; those
averages do not describe a single deployed router. All 479 problems remain in
the evaluation. They are separate from training, but have been exposed in the
earlier router experiments; a new generalization claim needs new problems.

## Check the answers

Answer keys are joined from the pinned Easy2Hard source using exact problem IDs,
text and file hashes. They stay separate from the inference inputs. Models see
the problem and competition name; no reference answer, solution or difficulty
rating is sent to them.

The grader extracts the explicit final answer and uses pinned **Math-Verify
0.9.0** to compare equivalent mathematical expressions. It does not search a
long solution for any expression that happens to match. It grades the final
value, not proof completeness or prose quality.

Before grading the calibration or test answers, 28 answer conventions were
reviewed against the original problems and solutions, without consulting solver
outputs. These include ordered coordinates, sets, ratios and clock notation.
Seven questions explicitly ask for a percentage and accept the requested value
with or without a percent sign. Elsewhere, `42%` means `0.42`, not `42`.
Two source transcription errors were corrected:
`e2h-amc-eval-1694` to `1010000`, and `e2h-amc-eval-620` to `21000`.
The review retains every problem and records the corrections explicitly.

`solver.py prepare` downloads and verifies the original source, separates
inference inputs from the answer key, and applies these checked review rules
automatically. It writes `gold-review.json` and
`gold-adjudication-details.json`; unresolved answers prevent collection.

## Try two fresh Cloud answer pairs

**CPU client; SIE Cloud generates the answers.** Create an account and API key
in the [SIE Console](https://console.superlinked.com). As checked on
**2026-10-19**, the [Serverless offer](https://superlinked.com/serverless)
advertises **$20 in free credits, with no credit card required**. Check the
balance displayed in your account before starting.

The two-problem pilot below made four initial solver calls and cost
**$0.01615** in the recorded run, including any final-answer turns. It checks
your connection and the collection/grading process; two problems cannot estimate
which router is better. It uses validation problems and does not select routing
cutoffs.

```sh
# Install the answer grader and point this terminal at SIE Cloud with your own key.
uv sync --frozen --group evaluation
export SIE_BASE_URL=https://api.superlinked.com
export SIE_API_KEY=your-key
```

```sh
# Prepare verified inputs, collect two validation pairs, and grade their final answers.
uv run --frozen --group evaluation python solver.py prepare \
  --evidence evidence --output run-output/my-solver-pilot
uv run --frozen --group evaluation python solver.py collect \
  --run run-output/my-solver-pilot --split validation --limit 2 \
  --workers 2 --max-credits 5000 --max-seconds 600
uv run --frozen --group evaluation python solver.py grade \
  --run run-output/my-solver-pilot --split validation
```

The results are in `run-output/my-solver-pilot/grades-validation.json`, with
answers and acquisition receipts under `answers-validation/`. Each model gets
one initial call and, if its answer is truncated or lacks a final answer, at
most one additional final-answer call. Both calls count toward cost.

## Collect a complete Cloud comparison

This optional reproduction collects **725 answer pairs**: 246 calibration and
479 test problems, or 1,450 initial solver calls. Allow roughly **two hours with
four workers**. The current estimate is **$5–7 for answers**, plus the recorded
**$0.00905** for measuring the embedding router on all 725 problems. This fits
the advertised $20 starting credit; the collection receipts determine the actual
bill. Downloaded replay results let you finish the walkthrough without waiting
for these calls.

First complete the walkthrough's shared downloads and embedding-classifier
training. You need `evidence/`, `run-output/lora-study/` and
`run-output/policies/`. The comparison reads the published seed-89 Laya scores
and the matching learned embedding policies. It does not substitute your new
Laya training run silently. Keep the source and prepared settings unchanged
throughout this experiment.

Prepare a fresh study, then collect and grade calibration answers:

```sh
# Create a separate study and measure both solvers on all 246 calibration problems.
uv run --frozen --group evaluation python solver.py prepare \
  --evidence evidence --output run-output/my-solverstudy
uv run --frozen --group evaluation python solver.py collect \
  --run run-output/my-solverstudy --split calibration \
  --workers 4 --max-credits 300000 --max-seconds 3600
uv run --frozen --group evaluation python solver.py grade \
  --run run-output/my-solverstudy --split calibration
```

Measure the embedding decision's actual per-request charge, then choose and
save the cost/quality cutoffs using calibration answers only:

```sh
# Measure calibration routing overhead and freeze every routing rule before test collection.
uv run --frozen python solver_costs.py \
  --run run-output/my-solverstudy --split calibration --workers 4 \
  --output run-output/my-solverstudy/embedding-calibration
uv run --frozen python solver_policies.py freeze \
  --plan run-output/my-solverstudy/plan.json \
  --calibration-grades run-output/my-solverstudy/grades-calibration.json \
  --baseline evidence --study run-output/lora-study --policies run-output/policies \
  --router-credits run-output/my-solverstudy/embedding-calibration/router-credits.json \
  --output run-output/my-solverstudy/policies-frozen.json
```

Only after `policies-frozen.json` exists, collect the test answers and routing
charges, then apply all the saved choices:

```sh
# Measure all 479 test pairs and report the previously frozen policies without retuning.
uv run --frozen --group evaluation python solver.py collect \
  --run run-output/my-solverstudy --split test \
  --workers 4 --max-credits 500000 --max-seconds 7200
uv run --frozen --group evaluation python solver.py grade \
  --run run-output/my-solverstudy --split test
uv run --frozen python solver_costs.py \
  --run run-output/my-solverstudy --split test --workers 4 \
  --output run-output/my-solverstudy/embedding-test
uv run --frozen python solver_policies.py report \
  --frozen run-output/my-solverstudy/policies-frozen.json \
  --test-grades run-output/my-solverstudy/grades-test.json \
  --router-credits run-output/my-solverstudy/embedding-test/router-credits.json \
  --output run-output/my-solverstudy/report.json
```

Credit and time limits stop the collector from starting more tasks; calls
already in flight can finish. Credit limits apply to newly observed charges
in that invocation. If a collection stops, preserve its files and repeat that
`collect` command with `--resume`. Complete answers are reused and earlier
attempts remain recorded. An uncertain previous execution or charge requires
the separate `--retry-unknown` option; it is not retried automatically. Run
`grade` only after collection completes. Use a new study directory when changing
the problem selection or solver settings.

## Use your own SIE deployment for answer quality

**CPU client; your generation server supplies the compute.** This path needs
an existing SIE endpoint exposing `Qwen/Qwen3.5-4B` and
`Qwen/Qwen3.8-27B-FP8` at the model revisions declared in `solver.py`, including
structured final-answer support. The collector checks those revisions before
inference. Setting an endpoint does not start the models; use the
[SIE deployment guide](https://superlinked.com/docs/deployment/) to configure
your server. The local Laya and embedding servers in the walkthrough serve
routers, not this pair of answer models.

Use the same downloaded evidence and learned embedding policies as above.
For a server running on your own computer at port 8080:

```sh
# Select your local SIE endpoint and prepare a separate self-hosted answer study.
export SIE_BASE_URL=http://127.0.0.1:8080
unset SIE_API_KEY  # The local server needs no Cloud key.
uv run --frozen --group evaluation python solver.py prepare \
  --evidence evidence --output run-output/my-local-solverstudy
```

If your deployment requires authentication, set its own API key instead.
Collect calibration answers with `--self-hosted`, which permits responses
without Cloud billing receipts:

```sh
# Collect and grade local calibration answers, then retain the original difficulty cutoffs.
uv run --frozen --group evaluation python solver.py collect \
  --run run-output/my-local-solverstudy --split calibration --self-hosted \
  --workers 4 --max-seconds 3600
uv run --frozen --group evaluation python solver.py grade \
  --run run-output/my-local-solverstudy --split calibration
uv run --frozen python solver_policies.py freeze \
  --plan run-output/my-local-solverstudy/plan.json \
  --calibration-grades run-output/my-local-solverstudy/grades-calibration.json \
  --baseline evidence --study run-output/lora-study --policies run-output/policies \
  --quality-only --output run-output/my-local-solverstudy/policies-frozen.json
```

`--quality-only` saves the original difficulty cutoffs. It skips cost-based
selection because unknown hosting costs cannot identify the cheapest policy.
Then measure the test answers:

```sh
# Collect local test answers and report answer accuracy and quality gaps without invented prices.
uv run --frozen --group evaluation python solver.py collect \
  --run run-output/my-local-solverstudy --split test --self-hosted \
  --workers 4 --max-seconds 7200
uv run --frozen --group evaluation python solver.py grade \
  --run run-output/my-local-solverstudy --split test
uv run --frozen python solver_policies.py report \
  --frozen run-output/my-local-solverstudy/policies-frozen.json \
  --test-grades run-output/my-local-solverstudy/grades-test.json \
  --output run-output/my-local-solverstudy/report.json
```

This report contains no costs, savings, break-even prices or cost-selected
policies. It does not treat missing billing as free computation. For a
financial comparison, measure your hardware bill and useful throughput,
including idle time. A local latency measurement alone is not a Cloud price
or a complete hosting-cost estimate.

## Account for the experiment separately from serving

SIE Cloud receipts provide the billed credits for each completed call. Dollar
conversion uses the recorded credit-pack price; future prices may differ.
The two solvers have fixed output budgets and at most one separately charged
final-answer turn. The baseline describes those settings, not the best result
the larger model could achieve with unlimited computation.

Collecting both answers is an experiment cost. Serving a future request pays
for one selected answer and, where needed, one routing decision. Training,
evaluation collection and production serving therefore have separate totals.
Failed or interrupted collection attempts retain their receipts; an unknown
charge is not counted as a free request.
