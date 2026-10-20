"""Stable evaluation of the existing arithmetic logistic-training objective."""

from __future__ import annotations

import numpy as np

import benefit_outcomes as outcomes

EVALUATION = {
    "objective": "Sum softplus((1-2*y)*logit) + 0.5*lambda*||weights||^2",
    "equivalence": "Algebraically identical Bernoulli NLL; avoids subtracting two large floating-point sums.",
    "unchanged": "Penalty grid, regularization, intercept, folds, stopping tolerances, candidate selection and cost model.",
}


def fit_logistic(matrix, targets, penalty):
    """Fit the unchanged objective using a cancellation-resistant value."""
    import scipy
    from scipy.optimize import minimize
    from scipy.special import expit

    outcomes._require(
        scipy.__version__ == outcomes.PROTOCOL["optimizer"]["scipy"], "Fit requires the declared SciPy version"
    )
    matrix, targets = np.asarray(matrix, dtype=float), np.asarray(targets, dtype=float)
    outcomes._require(
        matrix.ndim == 2
        and targets.shape == (len(matrix),)
        and np.isfinite(matrix).all()
        and np.isin(targets, (0, 1)).all()
        and np.isfinite(penalty)
        and penalty > 0,
        "Need finite vectors, binary labels and a positive fixed penalty",
    )
    mean = matrix.mean(axis=0)
    centered = np.ascontiguousarray(matrix - mean)
    frequency = float(targets.mean())
    outcomes._require(0 < frequency < 1, "Each training fold needs both correctness classes for each solver")
    initial = np.zeros(matrix.shape[1] + 1)
    initial[-1] = np.log(frequency / (1 - frequency))

    def loss(theta):
        weights, bias = theta[:-1], theta[-1]
        logits = centered @ weights + bias
        residual = expit(logits) - targets
        value = np.logaddexp(0, (1 - 2 * targets) * logits).sum() + 0.5 * penalty * (weights @ weights)
        return float(value), np.r_[centered.T @ residual + penalty * weights, residual.sum()]

    options = {key: outcomes.PROTOCOL["optimizer"][key] for key in ("maxiter", "ftol", "gtol", "maxls")}
    result = minimize(loss, initial, method="L-BFGS-B", jac=True, options=options)
    outcomes._require(result.success and np.isfinite(result.x).all(), f"Logistic optimization failed: {result.message}")
    return {
        "weights": result.x[:-1].tolist(),
        "bias": float(result.x[-1] - mean @ result.x[:-1]),
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "objective": float(result.fun),
            "gradient_absolute_max": float(np.abs(result.jac).max()),
            "loss_evaluation": EVALUATION["objective"],
        },
    }


def fit_head(matrix, rows, groups, costs, *, fold_assignments, progress=None):
    """Use the original fixed search and costs with the stable loss value."""
    outcomes.solver_report.summarize_quality(rows, "always_strong")
    matrix = np.asarray(matrix, dtype=float)
    targets = np.array([[int(row[f"{arm}_correct"]) for arm in outcomes.ARMS] for row in rows], dtype=float)
    outcomes._require(
        matrix.ndim == 2 and len(matrix) == len(rows) == len(groups) and np.isfinite(matrix).all(),
        "Training vectors, paired targets and paper groups must align",
    )
    folds = np.asarray(fold_assignments)
    outcomes._require(
        folds.shape == (len(rows),)
        and set(folds.tolist()) == set(range(outcomes.PROTOCOL["folds"]))
        and all(len({int(folds[i]) for i, item in enumerate(groups) if item == group}) == 1 for group in set(groups)),
        "Every training family must stay in one of the declared folds",
    )
    statistics = outcomes.conditional_costs(targets, costs)
    candidates = []
    for penalty in outcomes.PROTOCOL["penalties"]:
        predictions = np.empty_like(targets)
        optimizers = []
        for fold in range(outcomes.PROTOCOL["folds"]):
            held = folds == fold
            for arm, name in enumerate(outcomes.ARMS):
                model = fit_logistic(matrix[~held], targets[~held, arm], penalty)
                predictions[held, arm] = outcomes.probability(matrix[held], model)
                optimizers.append({"fold": fold, "arm": name, **model["optimizer"]})
        losses = outcomes._nll(targets, predictions)
        candidates.append(
            {
                "penalty": penalty,
                "cheap_nll": float(losses[0]),
                "strong_nll": float(losses[1]),
                "summed_binary_logloss": float(losses.sum()),
                "oof_probabilities": predictions.tolist(),
                "optimizer_runs": optimizers,
            }
        )
        if progress is not None:
            progress(f"Training family-fold CV lambda={penalty:g}: summed NLL={losses.sum():.6f}")
    winner = min(candidates, key=lambda item: (item["summed_binary_logloss"], -item["penalty"]))
    models = {name: fit_logistic(matrix, targets[:, arm], winner["penalty"]) for arm, name in enumerate(outcomes.ARMS)}
    return {
        "training_ids": [row["id"] for row in rows],
        "targets": targets.astype(int).tolist(),
        "folds": folds.tolist(),
        "cv": candidates,
        "penalty": winner["penalty"],
        "models": models,
        "training_cost_proxies": costs,
        "conditional_cost_statistics": statistics,
    }
