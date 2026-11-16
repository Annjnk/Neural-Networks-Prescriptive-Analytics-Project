"""Tests for the single test-run module (temporary git repos and synthetic data, no project files).  pytest -q"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uaq import test_run as tr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _git(root, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()


CODE = '"""Module docstring."""\n\n\ndef f(x):\n    """Old docstring."""\n    return x + 1  # a comment\n'


def _commit(root, msg="change"):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


@pytest.fixture
def repo(tmp_path):
    """A tiny git repo with the frozen files committed and tagged pre-test-v2."""
    files = {**{f: f"frozen {f}\n" for f in tr.FROZEN_FILES}, "src/uaq/models.py": CODE, "src/uaq/test_run.py": "X = 1\n",
             "notebooks/09_test_run.ipynb": "{}\n"}
    for f, text in files.items():
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _git(tmp_path, "init", "-q")
    _commit(tmp_path, "frozen")
    _git(tmp_path, "tag", tr.TAG)
    return tmp_path


def test_preconditions_pass_on_a_clean_tagged_tree(repo):
    info = tr.check_preconditions(repo)
    head = _git(repo, "rev-parse", "HEAD")
    assert info["tag_commit"] == info["head_commit"] == head and info["tag"] == tr.TAG
    assert set(info["hashes"]) == set(tr.FROZEN_FILES) and info["docstring_only_changes"] == []


def test_later_commits_to_test_run_notebooks_or_docstrings_are_allowed(repo):
    (repo / "src/uaq/test_run.py").write_text("X = 2\n")
    (repo / "notebooks/09_test_run.ipynb").write_text('{"cells": []}\n')
    (repo / "src/uaq/models.py").write_text(CODE.replace("Old docstring.", "New docstring.").replace("a comment", "edited"))
    _commit(repo)
    info = tr.check_preconditions(repo)
    assert info["head_commit"] != info["tag_commit"] and info["docstring_only_changes"] == ["src/uaq/models.py"]


@pytest.mark.parametrize("path,text", [
    ("configs/study_config.yaml", "changed\n"),
    ("outputs/results/forecast_selected.json", "changed\n"),
    ("outputs/results/network_validation.csv", "a new earlier-style results file\n"),
    ("src/uaq/models.py", CODE.replace("x + 1", "x + 2")),                  # a code change, not a docstring
    ("src/uaq/new_module.py", "Y = 1\n"),                                 # an added module
])
def test_preconditions_refuse_changed_frozen_files(repo, path, text):
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    _commit(repo)
    with pytest.raises(RuntimeError, match="changed since|added or removed"):
        tr.check_preconditions(repo)


def test_preconditions_refuse_a_dirty_tree(repo):
    (repo / "configs" / "study_config.yaml").write_text("changed\n")
    with pytest.raises(RuntimeError, match="not clean"):
        tr.check_preconditions(repo)
    _git(repo, "checkout", "--", ".")
    (repo / "stray.txt").write_text("untracked\n")
    with pytest.raises(RuntimeError, match="not clean"):
        tr.check_preconditions(repo)


def test_preconditions_refuse_a_missing_tag(repo):
    with pytest.raises(RuntimeError, match="not found"):
        tr.check_preconditions(repo, tag="no-such-tag")


def test_preconditions_refuse_an_existing_log_unless_a_rerun_reason_is_logged(repo):
    (repo / tr.LOG).write_text(json.dumps({"tag": tr.TAG}))
    (repo / tr.RESULTS / "test_summary.json").write_text("{}")             # test outputs do not count as frozen files
    (repo / tr.RESULTS / "network_test.csv").write_text("k\n")
    _commit(repo, "test run")
    with pytest.raises(RuntimeError, match="already happened"):
        tr.check_preconditions(repo)
    info = tr.check_preconditions(repo, rerun_reason="evaluation bug: wrong column")
    entry = tr.write_log(repo, info, rerun_reason="evaluation bug: wrong column")
    assert entry["reruns"][0]["reason"] == "evaluation bug: wrong column"
    with pytest.raises(RuntimeError):
        tr.write_log(repo, info)


def test_write_log_records_tag_commit_time_and_hashes(repo):
    info = tr.check_preconditions(repo)
    log = tr.write_log(repo, info)
    saved = json.loads((repo / tr.LOG).read_text())
    assert saved == log and {"tag", "tag_commit", "head_commit", "hashes", "run_time"} <= set(saved)
    assert saved["hashes"]["configs/study_config.yaml"] == tr.sha256(repo / "configs" / "study_config.yaml")


def test_outputs_never_touch_configs_or_earlier_results(tmp_path):
    tracked = _git(REPO, "ls-tree", "--name-only", f"{tr.TAG}", "outputs/results/").splitlines()
    earlier = {Path(p).name for p in tracked}
    assert earlier and not set(tr.OUTPUT_FILES) & earlier                 # new names only
    assert not any("configs" in n for n in tr.OUTPUT_FILES)
    (tmp_path / tr.RESULTS).mkdir(parents=True)
    for bad in ("forecast_selected.json", "warning_policy.json", "../../configs/study_config.yaml", "test_run_log.json"):
        with pytest.raises(ValueError):
            tr.save_outputs(tmp_path, {bad: {}})
    written = tr.save_outputs(tmp_path, {"test_summary.json": {"a": 1}, "test_forecast.csv": pd.DataFrame({"x": [1]})})
    assert {p.name for p in written} == {"test_summary.json", "test_forecast.csv"}
    assert sorted(p.name for p in (tmp_path / tr.RESULTS).iterdir()) == ["test_forecast.csv", "test_summary.json"]
    with pytest.raises(FileExistsError):                                  # written once
        tr.save_outputs(tmp_path, {"test_summary.json": {"a": 2}})


def test_paired_block_bootstrap_is_deterministic_and_brackets_the_difference():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-11-01", periods=120, freq="D")
    y = rng.uniform(50, 250, 120)
    t = pd.DataFrame({"y": y, "pm_lag0": y + rng.normal(0, 40, 120), "pred_m": y + rng.normal(0, 20, 120)}, index=idx)
    a, b = tr.bootstrap_mae_difference(t, "m", n=300), tr.bootstrap_mae_difference(t, "m", n=300)
    assert a == b and a["lo"] < a["mae_difference"] < a["hi"] < 0


def test_verdict_rules():
    scores = pd.DataFrame({"skill_mae": [0.1]}, index=["mlp (weather)"])
    assert tr.verdict_h1(scores, {"mae_difference": -5, "lo": -9, "hi": -1})["verdict"] == "SUPPORTED"
    assert tr.verdict_h1(scores, {"mae_difference": -5, "lo": -9, "hi": 1})["verdict"] == "NOT SUPPORTED"
    abl = pd.DataFrame({"difference": [-2.0, 1.0]}, index=["mlp", "lstm"])
    assert tr.verdict_h2(abl)["verdict"] == "SUPPORTED"
    rows = []
    for k, ga in ((3, 0.4), (4, 0.4), (5, 0.6)):
        rows += [{"k": k, "method": "genetic algorithm", "F_test": ga, "F_validation": 0.4},
                 {"k": k, "method": "exhaustive (optimum)", "F_test": 0.4, "F_validation": 0.4},
                 {"k": k, "method": "greedy forward", "F_test": 0.45, "F_validation": 0.45},
                 {"k": k, "method": "random (median)", "F_test": 0.7, "F_validation": 0.7},
                 {"k": k, "method": "highest mean", "F_test": 0.5, "F_validation": 0.5},
                 {"k": k, "method": "spatial spread", "F_test": 0.55, "F_validation": 0.55}]
    net = pd.DataFrame(rows)
    h3 = tr.verdict_h3(net)
    assert h3["k_won"] == [3, 4] and h3["verdict"] == "SUPPORTED"
    assert tr.verdict_h4(net)["verdict"] == "SUPPORTED"


def test_verdicts_are_json_serialisable():
    net = pd.DataFrame([{"k": np.int64(3), "method": m, "F_test": f, "F_validation": f}
                        for m, f in (("genetic algorithm", 0.4), ("exhaustive (optimum)", 0.4), ("greedy forward", 0.45),
                                     ("random (median)", 0.7), ("highest mean", 0.5), ("spatial spread", 0.55))])
    json.dumps({"h3": tr.verdict_h3(net), "h4": tr.verdict_h4(net)})
