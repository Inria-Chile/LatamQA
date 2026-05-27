import pandas as pd
import pytest

from latamqa.eval_mcq import shuffle_options


@pytest.fixture()
def sample_dataset(tmp_path):
    """Create a minimal mock dataset structure for testing."""
    items = [
        {
            "article_id": "q1",
            "question": "¿Capital de Chile?",
            "question_en": "Capital of Chile?",
            "answer": "Santiago",
            "answer_en": "Santiago",
            "distractor1": "Valparaíso",
            "distractor1_en": "Valparaiso",
            "distractor2": "Concepción",
            "distractor2_en": "Concepcion",
            "distractor3": "La Serena",
            "distractor3_en": "La Serena",
        },
        {
            "article_id": "q2",
            "question": "¿Río más largo de Sudamérica?",
            "question_en": "Longest river in South America?",
            "answer": "Amazonas",
            "answer_en": "Amazon",
            "distractor1": "Paraná",
            "distractor1_en": "Parana",
            "distractor2": "Orinoco",
            "distractor2_en": "Orinoco",
            "distractor3": "Magdalena",
            "distractor3_en": "Magdalena",
        },
    ]
    return items


def _build_truth(items, lang="regional", seed=42):
    """Reconstruct correct letters for a set of items using the same shuffle logic."""
    if lang == "english":
        a_col, d1_col, d2_col, d3_col = "answer_en", "distractor1_en", "distractor2_en", "distractor3_en"
    else:
        a_col, d1_col, d2_col, d3_col = "answer", "distractor1", "distractor2", "distractor3"

    truth = {}
    for item in items:
        row_seed = seed + hash(str(item["article_id"])) % 10000
        _, correct_letter = shuffle_options(item[a_col], item[d1_col], item[d2_col], item[d3_col], row_seed)
        truth[item["article_id"]] = correct_letter
    return truth


def test_build_blind_mcq_dataframe_excludes_answers(sample_dataset):
    """Blind MCQ rows must not contain correct answers or correct_letter."""
    from latamqa.leaderboard import build_blind_mcq_dataframe

    df = build_blind_mcq_dataframe(sample_dataset, region="es-la", seed=42)

    expected_cols = {"article_id", "region", "lang", "question", "option_A", "option_B", "option_C", "option_D"}
    assert set(df.columns) == expected_cols
    assert "answer" not in df.columns
    assert "correct_answer" not in df.columns
    assert "correct_letter" not in df.columns


def test_build_blind_mcq_dataframe_melted_schema(sample_dataset):
    """Each article_id should produce one row per language, tagged with region+lang."""
    from latamqa.leaderboard import build_blind_mcq_dataframe

    df = build_blind_mcq_dataframe(sample_dataset, region="es-la", seed=42)

    assert len(df) == len(sample_dataset) * 2
    assert set(df["region"].unique()) == {"es-la"}
    assert set(df["lang"].unique()) == {"regional", "english"}

    regional = df[df["lang"] == "regional"].set_index("article_id")
    english = df[df["lang"] == "english"].set_index("article_id")
    assert regional.loc["q1", "question"] == "¿Capital de Chile?"
    assert english.loc["q1", "question"] == "Capital of Chile?"


def test_build_blind_mcq_dataframe_shuffle_reproducibility(sample_dataset):
    """Same seed should produce identical option orderings; different seed should differ."""
    from latamqa.leaderboard import build_blind_mcq_dataframe

    df1 = build_blind_mcq_dataframe(sample_dataset, region="es-la", seed=42)
    df2 = build_blind_mcq_dataframe(sample_dataset, region="es-la", seed=42)
    df3 = build_blind_mcq_dataframe(sample_dataset, region="es-la", seed=99)

    pd.testing.assert_frame_equal(df1, df2)
    option_cols = ["option_A", "option_B", "option_C", "option_D"]
    assert not df1[option_cols].equals(df3[option_cols])


@pytest.fixture()
def _stub_hub(monkeypatch, sample_dataset):
    """Patch load_dataset, push_to_hub, and the canonical-name resolver for export tests."""
    from datasets import Dataset as HFDataset

    mock_mcq = {"train": HFDataset.from_list(sample_dataset)}
    mock_articles = {"train": HFDataset.from_list([{"article_id": "q1", "text": "a"}, {"article_id": "q2", "text": "b"}])}

    def fake_load_dataset(name):
        return mock_articles if "articles" in name else mock_mcq

    pushed = []

    def fake_push(self, *args, **kwargs):
        # Accept both positional and keyword repo_id for robustness.
        repo_id = kwargs.get("repo_id") if "repo_id" in kwargs else (args[0] if args else None)
        private = kwargs.get("private", False)
        pushed.append((repo_id, self.to_pandas(), private))

    monkeypatch.setattr("latamqa.leaderboard.load_dataset", fake_load_dataset)
    monkeypatch.setattr("datasets.Dataset.push_to_hub", fake_push, raising=False)
    # Pretend the target repos do not exist on the Hub (no redirect alias to worry about).
    monkeypatch.setattr("latamqa.leaderboard._resolve_canonical_repo_id", lambda repo_id: None)

    return pushed


def test_export_blind_dataset_pushes_to_hub(_stub_hub):
    """export_blind_dataset should push a melted, blind MCQ dataset to the configured repo."""
    from latamqa.eval_mcq import REGIONAL_DATASETS
    from latamqa.leaderboard import export_blind_dataset

    export_blind_dataset(seed=42, target_repo="fake/latamqa", articles_repo="fake/latamqa_articles", private=True)

    pushed = _stub_hub
    assert len(pushed) == 2
    mcq_repo, mcq_df, mcq_private = pushed[0]
    articles_repo, articles_df, articles_private = pushed[1]

    assert mcq_repo == "fake/latamqa"
    assert mcq_private is True
    assert set(mcq_df["region"].unique()) == set(REGIONAL_DATASETS)
    assert set(mcq_df["lang"].unique()) == {"regional", "english"}
    expected_rows = len(sample_dataset_items()) * 2 * len(REGIONAL_DATASETS)
    assert len(mcq_df) == expected_rows
    assert "answer" not in mcq_df.columns
    assert "correct_letter" not in mcq_df.columns

    assert articles_repo == "fake/latamqa_articles"
    assert articles_private is True
    assert "region" in articles_df.columns
    assert set(articles_df["region"].unique()) == set(REGIONAL_DATASETS)


def test_export_blind_dataset_skip_articles(_stub_hub):
    """--skip_articles should push only the MCQ dataset."""
    from latamqa.leaderboard import export_blind_dataset

    export_blind_dataset(seed=42, target_repo="fake/latamqa", skip_articles=True)

    assert [repo for repo, _, _ in _stub_hub] == ["fake/latamqa"]


def test_export_blind_dataset_rejects_forbidden_target(monkeypatch):
    """Passing a known source dataset as --target_repo must abort before any push."""
    from latamqa.leaderboard import export_blind_dataset

    called = []
    monkeypatch.setattr("latamqa.leaderboard.load_dataset", lambda name: called.append(name))

    with pytest.raises(RuntimeError, match="source dataset"):
        export_blind_dataset(target_repo="inria-chile/latamqa_mcq_es-la")

    assert called == [], "load_dataset must not be called once the guard has fired"


def test_export_blind_dataset_rejects_redirect_alias(monkeypatch):
    """Passing a name that Hugging Face resolves to a source dataset must abort before any push."""
    from latamqa.leaderboard import export_blind_dataset

    # 'inria-chile/latamqa' is itself in the forbidden set (it's the historical alias),
    # so strip it from the literal check and rely on the resolver to catch it.
    monkeypatch.setattr(
        "latamqa.leaderboard.FORBIDDEN_PUSH_TARGETS",
        {"inria-chile/latamqa_mcq_es-la"},
    )
    monkeypatch.setattr(
        "latamqa.leaderboard._resolve_canonical_repo_id",
        lambda repo_id: "inria-chile/latamqa_mcq_es-la" if repo_id == "inria-chile/latamqa" else None,
    )

    called = []
    monkeypatch.setattr("latamqa.leaderboard.load_dataset", lambda name: called.append(name))

    with pytest.raises(RuntimeError, match="resolves that name to"):
        export_blind_dataset(target_repo="inria-chile/latamqa")

    assert called == [], "load_dataset must not be called once the guard has fired"


def test_export_blind_dataset_dry_run_does_not_push(_stub_hub, sample_dataset):
    """dry_run=True should build the datasets but never call push_to_hub."""
    from latamqa.leaderboard import export_blind_dataset

    export_blind_dataset(
        seed=42,
        target_repo="fake/latamqa",
        articles_repo="fake/latamqa_articles",
        dry_run=True,
    )

    assert _stub_hub == [], "dry_run must not push anything"


def sample_dataset_items():
    """Mirror of the sample_dataset fixture for use inside tests that need the count without the fixture arg."""
    return [
        {"article_id": "q1"},
        {"article_id": "q2"},
    ]


def test_score_perfect_submission(sample_dataset, tmp_path, monkeypatch):
    """A submission with all correct answers should get 100% accuracy."""
    from datasets import Dataset as HFDataset

    mock_ds = {"train": HFDataset.from_list(sample_dataset)}
    monkeypatch.setattr("latamqa.leaderboard.load_dataset", lambda name: mock_ds)

    from latamqa.leaderboard import score_submission

    # Build a perfect submission
    truth = _build_truth(sample_dataset, lang="regional", seed=42)
    sub_df = pd.DataFrame([
        {"article_id": aid, "predicted_letter": letter} for aid, letter in truth.items()
    ])
    sub_path = tmp_path / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    summary = score_submission(
        submission_path=sub_path,
        region="es-la",
        lang="regional",
        model_name="test-model",
        seed=42,
        results_dir=tmp_path / "results",
    )

    assert summary["accuracy"] == 1.0
    assert summary["correct"] == 2
    assert summary["total"] == 2


def test_score_wrong_submission(sample_dataset, tmp_path, monkeypatch):
    """A submission with all wrong answers should get 0% accuracy."""
    from datasets import Dataset as HFDataset

    mock_ds = {"train": HFDataset.from_list(sample_dataset)}
    monkeypatch.setattr("latamqa.leaderboard.load_dataset", lambda name: mock_ds)

    from latamqa.leaderboard import score_submission

    # Build wrong submission - use opposite letters
    truth = _build_truth(sample_dataset, lang="regional", seed=42)
    wrong_map = {"A": "B", "B": "C", "C": "D", "D": "A"}
    sub_df = pd.DataFrame([
        {"article_id": aid, "predicted_letter": wrong_map[letter]} for aid, letter in truth.items()
    ])
    sub_path = tmp_path / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    summary = score_submission(
        submission_path=sub_path,
        region="es-la",
        lang="regional",
        model_name="test-model",
        seed=42,
        results_dir=tmp_path / "results",
    )

    assert summary["accuracy"] == 0.0
    assert summary["correct"] == 0


def test_score_missing_article_ids(sample_dataset, tmp_path, monkeypatch):
    """Missing article_ids should be scored as incorrect."""
    from datasets import Dataset as HFDataset

    mock_ds = {"train": HFDataset.from_list(sample_dataset)}
    monkeypatch.setattr("latamqa.leaderboard.load_dataset", lambda name: mock_ds)

    from latamqa.leaderboard import score_submission

    truth = _build_truth(sample_dataset, lang="regional", seed=42)
    # Only submit one answer (correct)
    first_id = next(iter(truth.keys()))
    sub_df = pd.DataFrame([{"article_id": first_id, "predicted_letter": truth[first_id]}])
    sub_path = tmp_path / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    summary = score_submission(
        submission_path=sub_path,
        region="es-la",
        lang="regional",
        model_name="test-model",
        seed=42,
        results_dir=tmp_path / "results",
    )

    assert summary["total"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] == 0.5


def test_score_output_format(sample_dataset, tmp_path, monkeypatch):
    """Scored results CSV should have the same columns as run_evaluation output."""
    from datasets import Dataset as HFDataset

    mock_ds = {"train": HFDataset.from_list(sample_dataset)}
    monkeypatch.setattr("latamqa.leaderboard.load_dataset", lambda name: mock_ds)

    from latamqa.leaderboard import score_submission

    truth = _build_truth(sample_dataset, lang="regional", seed=42)
    sub_df = pd.DataFrame([
        {"article_id": aid, "predicted_letter": letter} for aid, letter in truth.items()
    ])
    sub_path = tmp_path / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    score_submission(
        submission_path=sub_path,
        region="es-la",
        lang="regional",
        model_name="test-model",
        seed=42,
        results_dir=tmp_path / "results",
    )

    result_csv = tmp_path / "results" / "mcq_eval_results_es-la_regional_test-model.csv"
    assert result_csv.exists()

    df = pd.read_csv(result_csv)
    expected_cols = {
        "article_id", "question", "correct_answer",
        "option_A", "option_B", "option_C", "option_D",
        "correct_letter", "model_response", "predicted_letter", "is_correct",
    }
    assert set(df.columns) == expected_cols


def test_score_invalid_letters(sample_dataset, tmp_path, monkeypatch):
    """Invalid predicted_letter values should be treated as incorrect."""
    from datasets import Dataset as HFDataset

    mock_ds = {"train": HFDataset.from_list(sample_dataset)}
    monkeypatch.setattr("latamqa.leaderboard.load_dataset", lambda name: mock_ds)

    from latamqa.leaderboard import score_submission

    sub_df = pd.DataFrame([
        {"article_id": "q1", "predicted_letter": "E"},
        {"article_id": "q2", "predicted_letter": "AB"},
    ])
    sub_path = tmp_path / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    summary = score_submission(
        submission_path=sub_path,
        region="es-la",
        lang="regional",
        model_name="test-model",
        seed=42,
        results_dir=tmp_path / "results",
    )

    assert summary["accuracy"] == 0.0
    assert summary["correct"] == 0


def test_submission_missing_columns(sample_dataset, tmp_path, monkeypatch):
    """A submission missing required columns should raise ValueError."""
    from datasets import Dataset as HFDataset

    mock_ds = {"train": HFDataset.from_list(sample_dataset)}
    monkeypatch.setattr("latamqa.leaderboard.load_dataset", lambda name: mock_ds)

    from latamqa.leaderboard import score_submission

    sub_df = pd.DataFrame([{"article_id": "q1", "answer": "A"}])
    sub_path = tmp_path / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        score_submission(
            submission_path=sub_path,
            region="es-la",
            lang="regional",
            model_name="test-model",
            seed=42,
            results_dir=tmp_path / "results",
        )
