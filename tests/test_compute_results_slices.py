"""Tests for the region/language slice selection in ``compute_results``.

``run_evaluation`` and ``load_models`` are stubbed so these run fully offline
(no dataset download, no LLM calls) and only exercise which slices are chosen.
"""

import latamqa.model_eval as me
from latamqa.eval_mcq import REGIONAL_DATASETS, TARGET_LANGUAGES

ALL_SLICES = {(r, la) for r in REGIONAL_DATASETS for la in TARGET_LANGUAGES}


def _run(monkeypatch, tmp_path, **kwargs):
    """Call compute_results with stubs, returning the set of (region, lang) evaluated."""
    calls: list[tuple[str, str]] = []

    def fake_run_evaluation(*_args, region, lang, **_kwargs):
        calls.append((region, lang))
        return {"accuracy": 1.0}

    monkeypatch.setattr(me, "load_models", lambda _path: {"m": {"LiteLLM model name": "x"}})
    monkeypatch.setattr(me, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(me, "tqdm", lambda iterable, **_k: iterable)

    me.compute_results(model_name="m", results_dir=tmp_path, **kwargs)
    return set(calls)


def test_default_covers_all_slices(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path) == ALL_SLICES


def test_single_region_and_language_is_one_slice(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, region="pt-br", lang="english") == {("pt-br", "english")}


def test_region_only_iterates_all_languages(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, region="es-la") == {("es-la", la) for la in TARGET_LANGUAGES}


def test_language_only_iterates_all_regions(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, lang="regional") == {(r, "regional") for r in REGIONAL_DATASETS}
