import pandas as pd

# Import the necessary parts from compute_leaderboard
# Since it's not very modular, we'll test the logic directly
# or mock the dependencies.


def test_entry_to_dataframe():
    # Simulate the entry creation logic from main()
    results = {"Region (Lang)": 0.85}
    model_info = {
        "LiteLLM model name": "ollama/llama3.1:8b",
        "Model name": "meta-llama/Llama-3.1-8B-Instruct",
        "Model size": "8B",
    }
    entry = {**results, **model_info, "timestamp": pd.Timestamp.now()}

    # This is the line that was failing:
    df = pd.DataFrame(data=[entry])

    assert len(df) == 1
    assert df["Region (Lang)"][0] == 0.85
    assert df["Model name"][0] == "meta-llama/Llama-3.1-8B-Instruct"
    assert "timestamp" in df.columns


def test_concat_logic():
    df = pd.DataFrame()
    entry = {"a": 1, "b": 2, "timestamp": pd.Timestamp.now()}

    # The fix we applied
    df = pd.concat([df, pd.DataFrame(data=[entry])], ignore_index=True)

    assert len(df) == 1
    assert df["a"][0] == 1

    # Adding another entry
    entry2 = {"a": 3, "b": 4, "timestamp": pd.Timestamp.now()}
    df = pd.concat([df, pd.DataFrame(data=[entry2])], ignore_index=True)

    assert len(df) == 2
    assert df["a"][1] == 3


def test_save_and_load_csv_index(tmp_path):
    results_file = tmp_path / "leaderboard.csv"

    # First run
    df = pd.DataFrame()
    entry = {"a": 1, "b": 2}
    df = pd.concat([df, pd.DataFrame(data=[entry])], ignore_index=True)
    df.to_csv(results_file, index=False)

    # Second run
    df = pd.read_csv(results_file)
    entry2 = {"a": 3, "b": 4}
    df = pd.concat([df, pd.DataFrame(data=[entry2])], ignore_index=True)
    df.to_csv(results_file, index=False)

    # Load and check columns
    df_final = pd.read_csv(results_file)
    print(df_final.columns)
    # If index=True (default), we might have "Unnamed: 0"
    assert "Unnamed: 0" not in df_final.columns
