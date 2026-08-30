# References
# ----------
# The following works and accompanying source code were referenced in this implementation:
#
# [1] S. Bordt, H. Nori, V. Rodrigues, B. Nushi, and R. Caruana,
#     "Elephants Never Forget: Memorization and Learning of Tabular Data
#     in Large Language Models," COLM, 2024.
#     Code: https://github.com/interpretml/LLM-Tabular-Memorization-Checker
#
# [2] A. Capstick, R. G. Krishnan, and P. Barnaghi,
#     "AutoElicit: Using Large Language Models for Expert Prior Elicitation
#     in Predictive Modelling," ICML, 2025.
#     Code: https://github.com/alexcapstick/llm-elicited-priors
#

import io

import numpy as np
import pandas as pd

header_system_prompt = """You are an autocomplete bot for tabular datasets.
You will be prompted with parts of a tabular dataset.
Your task is to complete the dataset."""

row_system_prompt = """You are a helpful autocomplete bot for tabular datasets.
Your task is to provide rows as they are contained in tabular datasets.
The user provides a number of contiguous rows from a tabular dataset.
You then provide the next row from the dataset."""

row_system_prompt_2 = """You are a helpful autocomplete bot for tabular datasets.
Your task is to provide rows as they are contained in tabular datasets.
The user provides a number of contiguous rows from a tabular dataset.
You then provide the next {n_test_rows} rows from the dataset."""


def levenshtein_score(s1, s2, key=hash):
    def costmatrix(s1, s2, key=hash):
        rows = []

        previous_row = range(len(s2) + 1)
        rows.append(list(previous_row))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (key(c1) != key(c2))
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

            rows.append(previous_row)

        return rows

    rows = costmatrix(s1, s2, key)

    return rows[-1][-1]


def df_to_string(df):
    output = io.StringIO()
    df.to_csv(output, index=False)
    df_string = output.getvalue()
    output.close()

    return df_string.replace("\r\n", "\n")


def header_completion_test(
    df,
    few_shot_dataset_dir,
    wo_header=False,
    split_rows=[2, 4, 6, 8],
    completion_rows=8,
    system_prompt=header_system_prompt.replace("\n", " "),
    few_shot_dataset_names=[
        # "iris.csv",
        # "adult-train.csv",
        "openml-diabetes.csv",
        "uci-wine.csv",
        "california-housing.csv",
    ],
    rng=np.random.default_rng(42),
):
    df_string = df_to_string(df)
    df_rows_string = df_string.split("\n")

    if df_rows_string[-1] == "":
        df_rows_string = df_rows_string[:-1]
        df_string = "\n".join(df_rows_string)

    if wo_header:
        df_rows_string = df_rows_string[1:]
        df_string = "\n".join(df_rows_string)

    # completion_length = int(np.mean([len(s) for s in df_rows_string]) * completion_rows)

    fs_sets = {}
    for few_shot_dataset_name in few_shot_dataset_names:
        few_shot_dataset = pd.read_csv(str(few_shot_dataset_dir / few_shot_dataset_name))
        few_shot_dataset_string = df_to_string(few_shot_dataset)
        few_shot_dataset_rows_string = few_shot_dataset_string.split("\n")

        if few_shot_dataset_rows_string[-1] == "":
            few_shot_dataset_rows_string = few_shot_dataset_rows_string[:-1]
            few_shot_dataset_string = "\n".join(few_shot_dataset_rows_string)

        if wo_header:
            few_shot_dataset_rows_string = few_shot_dataset_rows_string[1:]
            few_shot_dataset_string = "\n".join(few_shot_dataset_rows_string)

        few_shot_completion_length = int(np.mean([len(s) for s in few_shot_dataset_rows_string]) * completion_rows)
        fs_sets[few_shot_dataset_name] = {
            "string": few_shot_dataset_string,
            "rows_string": few_shot_dataset_rows_string,
            "completion_length": few_shot_completion_length,
        }

    settings = {}
    for i_row in split_rows:
        user_prompts = []

        # Create tests
        offset = np.sum([len(row) for row in df_rows_string[: i_row - 1]])
        offset += rng.integers(len(df_rows_string[i_row]) // 3, 2 * len(df_rows_string[i_row]) // 3)

        test_prefix = df_string[:offset]
        # test_suffix = df_string[offset : offset + completion_length]
        test_suffix = df_string[offset:]

        # Create few-shot examples
        for few_shot_dataset_name in few_shot_dataset_names:
            few_shot_dataset_string = fs_sets[few_shot_dataset_name]["string"]
            few_shot_dataset_rows_string = fs_sets[few_shot_dataset_name]["rows_string"]
            few_shot_completion_length = fs_sets[few_shot_dataset_name]["completion_length"]

            few_shot_offset = np.sum([len(row) for row in few_shot_dataset_rows_string[: i_row - 1]])
            few_shot_offset += rng.integers(
                len(few_shot_dataset_rows_string[i_row]) // 3, 2 * len(few_shot_dataset_rows_string[i_row]) // 3
            )

            few_shot_prefix = few_shot_dataset_string[:few_shot_offset]
            few_shot_suffix = few_shot_dataset_string[few_shot_offset : few_shot_offset + few_shot_completion_length]

            user_prompts.append({"role": "user", "content": few_shot_prefix})
            user_prompts.append({"role": "assistant", "content": few_shot_suffix})

        user_prompts.append({"role": "user", "content": test_prefix})

        settings[f"ID_{i_row}"] = {
            "test_prefix": test_prefix,
            "test_suffix": test_suffix,
            "system_prompt": system_prompt,
            "user_prompts": user_prompts,
        }

    return settings


def row_completion_test(
    df,
    n_rows=6,
    n_tests=25,
    n_few_shot_examples=3,
    system_prompt=row_system_prompt.replace("\n", " "),
    rng=np.random.default_rng(42),
):
    df_string = df_to_string(df)
    df_rows_string = df_string.split("\n")

    if df_rows_string[-1] == "":
        df_rows_string = df_rows_string[:-1]
        df_string = "\n".join(df_rows_string)

    df_rows_string = df_rows_string[1:]
    df_string = "\n".join(df_rows_string)

    prefixes = []
    suffixes = []
    for idx in range(len(df_rows_string) - n_rows):
        prefixes.append("\n".join(df_rows_string[idx : idx + n_rows]))
        suffixes.append(df_rows_string[idx + n_rows])

    test_indices = []
    settings = {}
    while len(test_indices) < n_tests:
        test_idx = rng.choice(len(prefixes))
        if test_idx in test_indices:
            continue

        user_prompts = []

        # Create tests
        test_prefix = prefixes[test_idx]
        test_suffix = suffixes[test_idx]

        # Create few-shot examples
        if test_idx < n_rows:
            few_shot_candidate_indices = [i for i in range(test_idx + n_rows + 1, len(prefixes))]
        elif len(prefixes) - 1 < test_idx + n_rows + 1:
            few_shot_candidate_indices = [i for i in range(0, test_idx)]
        else:
            few_shot_candidate_indices = [i for i in range(0, test_idx)] + [
                i for i in range(test_idx + n_rows + 1, len(prefixes))
            ]

        few_shot_indices = []
        count = 0
        while len(few_shot_indices) < n_few_shot_examples:
            few_shot_idx = rng.choice(few_shot_candidate_indices)
            few_shot_prefix = prefixes[few_shot_idx]
            few_shot_suffix = suffixes[few_shot_idx]
            if (
                (few_shot_idx not in few_shot_indices)
                and len(few_shot_prefix) > 0
                and len(few_shot_suffix) > 0
                and (test_suffix not in few_shot_prefix)
                and (test_suffix not in few_shot_suffix)
            ):
                few_shot_indices.append(few_shot_idx)

            count += 1
            if count >= 1000:
                break

        if len(few_shot_indices) < n_few_shot_examples:
            continue

        for few_shot_idx in few_shot_indices:
            few_shot_prefix = prefixes[few_shot_idx]
            few_shot_suffix = suffixes[few_shot_idx]
            user_prompts.append({"role": "user", "content": few_shot_prefix})
            user_prompts.append({"role": "assistant", "content": few_shot_suffix})

        user_prompts.append({"role": "user", "content": test_prefix})

        test_indices.append(test_idx)
        settings[f"ID_{len(test_indices)}"] = {
            "test_prefix": test_prefix,
            "test_suffix": test_suffix,
            "system_prompt": system_prompt,
            "user_prompts": user_prompts,
        }

    return settings


def row_completion_test_2(
    df,
    few_shot_dataset_dir,
    n_example_rows=6,
    n_test_rows=20,
    n_tests=25,
    few_shot_dataset_names=[
        # "iris.csv",
        # "adult-train.csv",
        "openml-diabetes.csv",
        "uci-wine.csv",
        "california-housing.csv",
    ],
    system_prompt=row_system_prompt_2.replace("\n", " "),
    rng=np.random.default_rng(42),
):
    system_prompt = system_prompt.format(n_test_rows=n_test_rows)

    df_string = df_to_string(df)
    df_rows_string = df_string.split("\n")

    if df_rows_string[-1] == "":
        df_rows_string = df_rows_string[:-1]
        df_string = "\n".join(df_rows_string)

    df_rows_string = df_rows_string[1:]
    df_string = "\n".join(df_rows_string)

    prefixes = []
    suffixes = []
    for idx in range(len(df_rows_string) - n_example_rows - n_test_rows):
        prefixes.append("\n".join(df_rows_string[idx : idx + n_example_rows]))
        suffixes.append("\n".join(df_rows_string[idx + n_example_rows : idx + n_example_rows + n_test_rows]))

    fs_sets = {}
    for few_shot_dataset_name in few_shot_dataset_names:
        few_shot_dataset = pd.read_csv(str(few_shot_dataset_dir / few_shot_dataset_name))
        few_shot_dataset_string = df_to_string(few_shot_dataset)
        few_shot_dataset_rows_string = few_shot_dataset_string.split("\n")

        if few_shot_dataset_rows_string[-1] == "":
            few_shot_dataset_rows_string = few_shot_dataset_rows_string[:-1]
            few_shot_dataset_string = "\n".join(few_shot_dataset_rows_string)

        few_shot_dataset_rows_string = few_shot_dataset_rows_string[1:]
        few_shot_dataset_string = "\n".join(few_shot_dataset_rows_string)

        few_shot_prefixes = []
        few_shot_suffixes = []
        for idx in range(len(few_shot_dataset_rows_string) - n_example_rows - n_test_rows):
            few_shot_prefixes.append("\n".join(few_shot_dataset_rows_string[idx : idx + n_example_rows]))
            few_shot_suffixes.append(
                "\n".join(few_shot_dataset_rows_string[idx + n_example_rows : idx + n_example_rows + n_test_rows])
            )

        fs_sets[few_shot_dataset_name] = {
            "prefixes": few_shot_prefixes,
            "suffixes": few_shot_suffixes,
        }

    test_indices = []
    settings = {}
    while len(test_indices) < n_tests:
        test_idx = rng.choice(len(prefixes))
        if test_idx in test_indices:
            continue

        user_prompts = []

        # Create tests
        test_prefix = prefixes[test_idx]
        test_suffix = suffixes[test_idx]

        # Create few-shot examples
        for few_shot_dataset_name in few_shot_dataset_names:
            few_shot_idx = rng.choice(len(fs_sets[few_shot_dataset_name]["prefixes"]))
            few_shot_prefix = fs_sets[few_shot_dataset_name]["prefixes"][few_shot_idx]
            few_shot_suffix = fs_sets[few_shot_dataset_name]["suffixes"][few_shot_idx]
            user_prompts.append({"role": "user", "content": few_shot_prefix})
            user_prompts.append({"role": "assistant", "content": few_shot_suffix})

        user_prompts.append({"role": "user", "content": test_prefix})

        test_indices.append(test_idx)
        settings[f"ID_{len(test_indices)}"] = {
            "test_prefix": test_prefix,
            "test_suffix": test_suffix,
            "system_prompt": system_prompt,
            "user_prompts": user_prompts,
        }

    return settings
