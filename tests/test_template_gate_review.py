"""Consequential statistical properties of the proposed qualification design."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import betabinom, binom, binomtest

from parallasse.template_gate_review import (
    assess_proposed_cell,
    exact_lower_bound,
    ideal_operating_characteristics,
    required_hits,
)


def qualification():
    return json.loads(Path("configs/template_validation_revision.json").read_text())[
        "qualification"
    ]


def test_lower_bound_matches_independent_binomial_inversion():
    for hits in (0, 1, 956, 957, 987, 1000):
        expected = (
            binomtest(hits, 1000, alternative="greater")
            .proportion_ci(confidence_level=1 - 0.05 / 144, method="exact")
            .low
        )
        assert exact_lower_bound(hits, 1000, 0.05 / 144) == pytest.approx(expected, abs=1e-12)


def test_exact_qualification_boundary_and_type_one_error():
    assert required_hits(1000, 0.93, 0.05 / 144) == 957
    assert binom.sf(956, 1000, 0.93) <= 0.05 / 144
    assert binom.sf(955, 1000, 0.93) > 0.05 / 144
    assert exact_lower_bound(956, 1000, 0.05 / 144) < 0.93
    assert exact_lower_bound(957, 1000, 0.05 / 144) >= 0.93


def test_conservative_coverage_is_monotone_but_full_span_is_uninformative():
    q = qualification()
    accepted = [assess_proposed_cell(h, 1000, 0, 0.1, q)["qualified"] for h in range(1001)]
    assert accepted == sorted(accepted)
    assert sum(accepted) == 44
    assert not assess_proposed_cell(1000, 1000, 0, 1.0, q)["qualified"]
    assert not assess_proposed_cell(1000, 1000, 0, None, q)["qualified"]


def test_failure_denominator_and_boundary_are_explicit():
    q = qualification()
    assert assess_proposed_cell(957, 1000, 10, 0.25, q)["qualified"]
    assert not assess_proposed_cell(956, 1000, 10, 0.25, q)["qualified"]
    assert not assess_proposed_cell(957, 1000, 11, 0.25, q)["qualified"]
    with pytest.raises(ValueError):
        assess_proposed_cell(999, 1000, 2, 0.1, q)
    with pytest.raises(ValueError):
        assess_proposed_cell(957, 999, 0, 0.1, q)


def test_small_sample_can_be_unable_to_qualify_even_all_successes():
    assert required_hits(10, 0.93, 0.05 / 144) is None
    assert exact_lower_bound(1000, 1000, 0.05 / 144) == pytest.approx((0.05 / 144) ** 0.001)


def test_ideal_single_rank_matches_beta_binomial_mixture():
    outcome = ideal_operating_characteristics(1000, 973, 1, 1000, 916, 984, 0)
    expected = betabinom.cdf(915, 1000, 973, 28) + betabinom.sf(984, 1000, 973, 28)
    assert outcome["one_cell_veto_probability"] == pytest.approx(expected, abs=1e-10)
    assert outcome["mean_selected_content"] == pytest.approx(973 / 1001, abs=1e-11)


def test_maximum_calibration_increases_upper_veto_of_symmetric_gate():
    source = ideal_operating_characteristics(1000, 973, 3, 1000, 916, 984)
    model = ideal_operating_characteristics(1000, 973, 9, 1000, 916, 984)
    assert 0.07 < source["one_cell_upper_veto_probability"] < 0.08
    assert model["one_cell_upper_veto_probability"] > source["one_cell_upper_veto_probability"]
    assert (
        source["any_validation_cell_veto_probability"] > source["one_cell_upper_veto_probability"]
    )


def test_invalid_counts_are_rejected_and_bounds_are_finite():
    for hits, attempts, error in [
        (True, 1000, 0.05),
        (-1, 1000, 0.05),
        (2, 1, 0.05),
        (0, 0, 0.05),
        (1, 2, 0),
    ]:
        with pytest.raises(ValueError):
            exact_lower_bound(hits, attempts, error)
    assert np.isfinite(exact_lower_bound(np.int64(957), 1000, 0.05 / 144))


def test_public_review_reads_path_contract_and_never_scores(tmp_path, monkeypatch):
    import runpy

    script = Path("scripts/review_template_validation.py").resolve()
    namespace = runpy.run_path(str(script))
    review = namespace["review"]
    config = json.loads(Path("configs/template_validation_revision.json").read_text())
    monkeypatch.chdir(tmp_path)
    refs = {
        key: f"inputs/{key}.json"
        for key in (
            "parent_protocol",
            "parent_engine",
            "preserved_failed_run",
            "preserved_failed_gate",
            "source_cutoffs_to_reuse",
            "source_score_reference",
        )
    }
    config.update(refs)
    for path in refs.values():
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    Path(refs["preserved_failed_run"]).write_text(
        json.dumps(
            {
                "status": "source_failed",
                "score_archives": {"pythia_template": {}},
            }
        )
    )
    Path(refs["preserved_failed_gate"]).write_text('{"passed": false}')
    p = Path("configs/template_validation_revision.json")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(config))
    for path in (
        "src/parallasse/template_gate_review.py",
        "scripts/review_template_validation.py",
        "tests/test_template_gate_review.py",
        "requirements-lock.txt",
        "pyproject.toml",
    ):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic snapshot fixture")
    monkeypatch.setitem(review.__globals__, "load_engine", lambda path: {})
    result = review("synthetic-review")
    assert result["status"] == "completed_review"
    assert result["new_scores"] == result["new_validation_draws"] == 0
    assert result["access_granted"] is False
    design = json.loads(
        Path("results/task0013_review/synthetic-review/revision_design_freeze.json").read_text()
    )
    assert design["state"] == "design_only_not_executable"
    with pytest.raises(FileExistsError):
        review("synthetic-review")
