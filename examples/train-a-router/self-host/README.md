# Serve the three models on your own GPUs

This recipe exposes one local SIE endpoint for the encoder and both answer
models. The routing commands use the same prompts, model revisions and
conditional final-answer calls as SIE Cloud.

This is an optional inference deployment. The main example can use SIE Cloud
for these calls and needs only one GPU for LoRA training.

You need **Linux, Docker Compose, NVIDIA Container Toolkit and three separate
NVIDIA GPUs**. The validation hardware is three H100 80 GB GPUs. The strong
model's pinned image requires a CUDA 13 compatible driver. This is a larger
deployment than the single GPU used to train Laya's LoRA adapter. Sharing one
GPU between all three workers has not been validated.

The first start downloads about 48 GB of model weights, in addition to the
container images. The named `model-cache` volume retains those weights.
Each worker gets a separate runtime because the answer models need different
SGLang/CUDA versions. The answer workers use released images pinned by digest.
The encoder builds the `sglang-embedding` bundle from an exact public SIE
commit: the released generation bundle does not accept embedding adapters.
This initial CPU image build also takes time. The image digests, build commit
and model revisions are pinned in [compose.yaml](compose.yaml) and [models/](models/).

Run these commands from `examples/train-a-router`:

```sh
# Check the Compose configuration and start all workers, waiting for model readiness.
docker compose -f self-host/compose.yaml config --quiet
docker compose -f self-host/compose.yaml up --build --detach --wait --wait-timeout 1200
```

The default device IDs are 0, 1 and 2. Set `CHEAP_GPU`, `STRONG_GPU` and
`ENCODER_GPU` before startup to choose different, distinct devices. Cold model
loading takes several minutes. Inspect startup with
`docker compose -f self-host/compose.yaml logs --tail 100` if readiness fails.
The health checks require each pinned model profile to be loaded; a running
HTTP process alone does not count as ready. The SDK smoke below additionally
checks real inference and the structured final-answer path.

```sh
# Install the client dependencies and check metadata, embeddings and both answer protocols.
uv sync --frozen --group outcomes
export SIE_BASE_URL=http://127.0.0.1:8080
export SIE_API_KEY=''
uv run --frozen --group outcomes python self-host/smoke.py \
  --output run-output/self-host-smoke
```

The smoke check makes one embedding request and an initial answer plus a
forced structured finalizer for each answer model. The cheap worker preloads
both its speculative and grammar-safe engines, reserving 40% of GPU memory
for each. This avoids loading a second engine during an answer. Token admission
is capped at 8,192 per profile; this recipe is for sequential requests.
It validates immutable model revisions and saves an incomplete receipt
if any step fails. Use a new output directory for another attempt.

After following the main example's training instructions:

```sh
# Route a new request with your saved classifiers, then call only the selected answer model.
uv run --frozen --group outcomes python router.py answer \
  --study evidence --run run-output/my-router --population expanded \
  --method embedding \
  --question 'A rectangle has perimeter 30 and length 9. What is its area?' \
  --target 1 --self-hosted --output run-output/self-hosted-answer
```

`--self-hosted` allows missing Cloud billing receipts; GPU hosting is still
unpriced. It preserves model identity checks. The saved calibration target is
not a guarantee for an individual answer or a new workload.

```sh
# Stop the workers and proxy while preserving the downloaded model cache.
docker compose -f self-host/compose.yaml down
```

The frontend listens only on `127.0.0.1:8080`; worker ports 18081–18083 are also
published only on loopback. SGLang child ports stay on the private Compose
network. This is a trusted local deployment with no API-key authentication.
Docker administrators and other local users are privileged peers. Do not
publish these ports on a network without designing and reviewing authentication.

The small proxy accepts only the three pinned model metadata paths, the
encoder path and chat completion requests for the two answer models. It bounds
request bodies and headers, rejects ambiguous JSON model fields and strips
client credentials before forwarding. There are no automatic request retries.

The GPU smoke passed with both exact released answer images and the pinned
encoder build: all three model revisions, embeddings, both initial/finalizer
pairs, and one public routed request. Application processes ran as UID 1000
with writable caches and the original image dependencies. Modal connected the
isolated workers through authenticated tunnels and required materializing the
encoder Dockerfile's final `FROM` build argument to its intended stage; the
public Compose build uses the original Dockerfile and ordinary build arguments.
A complete Docker Compose GPU stack has not been executed as one deployment.
These checks do not establish throughput or bit-for-bit equivalence with
SIE Cloud.
