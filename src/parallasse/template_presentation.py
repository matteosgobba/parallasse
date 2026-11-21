"""Read verified score-template evidence for presentation without running inference."""

import hashlib
import json
from pathlib import Path

EVIDENCE = Path("results/task0014/20261116-template-evaluation-01")
PRESENTATION = Path("results/task0015/20261116-template-note-01")
CUT = "original_cut_delta"
SOURCE = "source_multinomial_profile"
SIMULATION = "simulation_multinomial_profile"
METHODS = (CUT, SOURCE, SIMULATION)
KEYS = tuple(f"{model}-{seed}" for model in ("PFN-K", "PFN-ID") for seed in (17, 29, 43))
LABELS = {CUT: "Cut inversion", SOURCE: "Source templates", SIMULATION: "Simulation aware"}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_evidence():
    """Verify the closed phase and its archived documents, never mutable memory hashes."""
    validation_path = EVIDENCE / "validation.json"
    validation = read_json(validation_path)
    snapshot = read_json(PRESENTATION / "prior_document_snapshot.json")
    if sha256(validation_path) != snapshot["parent_validation_sha256"]:
        raise ValueError("Presentation parent validation changed")
    for group in ("scientific_evidence_sha256", "verification_artifacts_sha256"):
        for path, digest in validation[group].items():
            if sha256(path) != digest:
                raise ValueError(f"Verified scientific evidence changed: {path}")
    for path, digest in validation["documentation_sha256"].items():
        archived = snapshot["documents"][path]
        if archived["sha256"] != digest or sha256(archived["archived_path"]) != digest:
            raise ValueError(f"Prior phase document changed: {path}")
    audit = read_json(EVIDENCE / "full_independent_audit.json")
    metrics = audit["metrics"]
    identity = {
        (r["generator"], r["sample_size"], r["fraction"], r["key"], r["method"]) for r in metrics
    }
    if len(metrics) != 216 or len(identity) != 216:
        raise ValueError("The complete 216-cell table is required")
    if any(r["attempts"] != 1000 or r["failures"] for r in metrics):
        raise ValueError("Presentation denominators differ from the verified result")
    return validation, audit


def metric_lookup(audit):
    return {
        (r["generator"], r["sample_size"], r["fraction"], r["key"], r["method"]): r
        for r in audit["metrics"]
    }


def presentation_data(audit):
    """Copy plotted quantities and ordering into a small auditable display manifest."""
    lookup = metric_lookup(audit)
    columns = [(n, f) for n in (1000, 10000) for f in (0.25, 0.5, 0.75)]
    primary = [lookup["herwig", 10000, 0.5, key, method] for key in KEYS for method in METHODS]
    grids = {
        generator: {
            method: [
                [lookup[generator, n, f, key, method]["coverage"] for n, f in columns]
                for key in KEYS
            ]
            for method in METHODS
        }
        for generator in ("pythia", "herwig")
    }
    minima = [
        min(
            (r for r in audit["metrics"] if r["generator"] == g and r["method"] == SIMULATION),
            key=lambda r: r["coverage"],
        )
        for g in ("pythia", "herwig")
    ]
    return {
        "source": str(EVIDENCE / "full_independent_audit.json"),
        "source_sha256": sha256(EVIDENCE / "full_independent_audit.json"),
        "checkpoint_order": list(KEYS),
        "method_order": list(METHODS),
        "column_order": [{"sample_size": n, "fraction": f} for n, f in columns],
        "primary": primary,
        "coverage_grids": grids,
        "secondary_minima": minima,
        "coverage_error_bars": "Per-cell Wilson MC95, 1000 attempted pseudoexperiments",
        "bias_and_span": "Per-checkpoint descriptive points; no seed pooling or marginal CIs",
        "empirical_reference": 0.93,
        "primary_criterion_met": audit["primary_criterion_met"],
        "scope": "Fixed-template disjoint-pool conditional evaluation within the modelling family",
    }
