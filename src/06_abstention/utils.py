import re
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Data containers
# =============================================================================


@dataclass
class MetricTableOutput:
    """Tables and metadata for one evaluation metric."""

    metric_name: str
    direction: str

    # rows = methods, columns = dataset x LLM
    score_values: pd.DataFrame

    # rows = methods, columns = dataset x LLM
    # method-level NA values are converted to bottom ranks.
    rank_values: pd.DataFrame

    # rows = methods
    # columns include Mean rank, Rank SD, NA count.
    summary: pd.DataFrame

    # Sub-table 1: score_values + Summary columns.
    sub_raw_table: pd.DataFrame

    # Sub-table 2: rank_values + Summary columns.
    sub_rank_table: pd.DataFrame

    # True where the original raw score was NA.
    original_na_mask: pd.DataFrame

    # Dataset x LLM conditions dropped because all methods were NA.
    dropped_all_na_conditions: list[tuple]


@dataclass
class AbstentionTableOutput:
    """All generated tables."""

    by_metric: dict[str, MetricTableOutput]

    # Main numeric tables.
    main_mean_rank_table: pd.DataFrame
    main_rank_sd_table: pd.DataFrame
    main_na_count_table: pd.DataFrame

    # Main text table with SD written inside each cell.
    # Example: "2.31 ± 1.10†"
    main_mean_rank_with_sd_table: pd.DataFrame

    # Metadata.
    metric_metadata: pd.DataFrame


# =============================================================================
# Small utilities
# =============================================================================


def direction_to_ascending(direction: str) -> bool:
    """
    Convert metric direction to pandas rank ascending flag.

    Parameters
    ----------
    direction:
        "larger_is_better" or "smaller_is_better".

    Returns
    -------
    bool
        pandas rank `ascending` flag.

    Notes
    -----
    larger_is_better:
        Larger raw scores are better, so rank in descending order.
        Best value gets rank 1.

    smaller_is_better:
        Smaller raw scores are better, so rank in ascending order.
        Best value gets rank 1.
    """
    if direction == "larger_is_better":
        return False
    elif direction == "smaller_is_better":
        return True
    else:
        raise ValueError("direction must be either 'larger_is_better' or 'smaller_is_better'")


def _safe_filename(name: str) -> str:
    """Make a filesystem-safe filename stem from a metric name."""
    name = str(name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("_")
    return name or "metric"


def flatten_columns_for_csv(
    df: pd.DataFrame,
    sep: str = " | ",
) -> pd.DataFrame:
    """
    Flatten MultiIndex columns into one-line CSV headers.

    Examples
    --------
    ("DatasetA", "LLM1") -> "DatasetA | LLM1"
    ("Summary", "Mean rank") -> "Summary | Mean rank"
    """
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [sep.join(str(level) for level in col if str(level) != "") for col in out.columns.to_list()]
    return out


def append_summary_columns(
    core_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    summary_cols: list[str],
) -> pd.DataFrame:
    """
    Append summary columns as MultiIndex columns.

    Added columns are:
        ("Summary", "Mean rank"), ("Summary", "Rank SD"), ...
    """
    missing = [col for col in summary_cols if col not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df does not contain columns: {missing}")

    summary_part = summary_df[summary_cols].copy()
    summary_part.columns = pd.MultiIndex.from_tuples(
        [("Summary", col) for col in summary_cols],
        names=["Dataset", "LLM"],
    )

    out = pd.concat([core_df, summary_part], axis=1)
    out.columns = pd.MultiIndex.from_tuples(
        out.columns.to_list(),
        names=["Dataset", "LLM"],
    )
    return out


def _format_float_or_na(x: float, fmt: str) -> str:
    """Format one value, preserving NA as the string 'NA'."""
    if pd.isna(x):
        return "NA"
    return f"{float(x):{fmt}}"


# =============================================================================
# Main string table helper
# =============================================================================


def make_main_mean_rank_with_sd_table(
    mean_rank_df: pd.DataFrame,
    rank_sd_df: pd.DataFrame,
    na_count_df: pd.DataFrame | None = None,
    mean_fmt: str = ".2f",
    sd_fmt: str = ".2f",
    style: str = "pm",
    na_display: str = "marker",
    na_marker: str = "†",
) -> pd.DataFrame:
    """
    Create a main-paper table with mean rank and rank SD in each cell.

    Parameters
    ----------
    mean_rank_df:
        Numeric table. Rows = methods, columns = metrics.

    rank_sd_df:
        Numeric table aligned with mean_rank_df.

    na_count_df:
        Optional numeric table aligned with mean_rank_df. If provided, cells can
        be marked when NA count > 0.

    style:
        "pm"    -> "2.31 ± 1.10"
        "paren" -> "2.31 (1.10)"

    na_display:
        "none"   -> no NA marker
        "marker" -> add † if NA count > 0

    Returns
    -------
    pd.DataFrame
        String-valued table with the same shape as mean_rank_df.
    """
    if mean_rank_df.shape != rank_sd_df.shape:
        raise ValueError("mean_rank_df and rank_sd_df must have the same shape")

    if not mean_rank_df.index.equals(rank_sd_df.index):
        raise ValueError("mean_rank_df and rank_sd_df must have the same index")

    if not mean_rank_df.columns.equals(rank_sd_df.columns):
        raise ValueError("mean_rank_df and rank_sd_df must have the same columns")

    if na_count_df is not None:
        if mean_rank_df.shape != na_count_df.shape:
            raise ValueError("mean_rank_df and na_count_df must have the same shape")
        if not mean_rank_df.index.equals(na_count_df.index):
            raise ValueError("mean_rank_df and na_count_df must have the same index")
        if not mean_rank_df.columns.equals(na_count_df.columns):
            raise ValueError("mean_rank_df and na_count_df must have the same columns")

    if style not in {"pm", "paren", "label"}:
        raise ValueError("style must be 'pm', 'paren', or 'label'")

    if na_display not in {"none", "marker", "count"}:
        raise ValueError("na_display must be 'none', 'marker', or 'count'")

    out = pd.DataFrame(index=mean_rank_df.index, columns=mean_rank_df.columns)

    for method in mean_rank_df.index:
        for metric in mean_rank_df.columns:
            mean_value = mean_rank_df.loc[method, metric]
            sd_value = rank_sd_df.loc[method, metric]

            if pd.isna(mean_value):
                text = "NA"
            else:
                mean_text = _format_float_or_na(mean_value, mean_fmt)
                sd_text = _format_float_or_na(sd_value, sd_fmt)

                if style == "pm":
                    text = f"{mean_text} ± {sd_text}"
                else:
                    text = f"{mean_text} ({sd_text})"

            if na_count_df is not None:
                na_count = int(na_count_df.loc[method, metric])

                if na_count > 0:
                    if na_display == "marker":
                        text = f"{text}{na_marker}"
                    # na_display == "none" does nothing.

            out.loc[method, metric] = text

    out.index.name = mean_rank_df.index.name
    return out


# =============================================================================
# Core builders
# =============================================================================


def build_metric_tables(
    df: pd.DataFrame,
    metric_name: str,
    direction: str,
    method_order: list[str],
    dataset_order: list[str],
    model_order: list[str],
    method_col: str = "method",
    dataset_col: str = "dataset_name",
    model_col: str = "model_w_params",
    aggfunc: str = "mean",
    drop_all_na_conditions: bool = True,
    include_rank_sd: bool = True,
) -> MetricTableOutput:
    """
    Build appendix raw/rank tables for one metric.

    Parameters
    ----------
    df:
        Long-form result DataFrame.

    metric_name:
        Column name of the metric to summarize.

    direction:
        "larger_is_better" or "smaller_is_better".

    method_order, dataset_order, model_order:
        Desired display/order of methods, datasets, and LLMs.

    drop_all_na_conditions:
        If True, drop dataset x LLM conditions where all methods are NA for the
        target metric. These are condition-level undefined cases, not
        method-level failures.

    include_rank_sd:
        Control which summary columns are appended to appendix tables.

    Returns
    -------
    MetricTableOutput
    """
    required_cols = [method_col, dataset_col, model_col, metric_name]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise KeyError(f"Missing columns in df: {missing_cols}")

    ascending_flag = direction_to_ascending(direction)

    expected_cols = pd.MultiIndex.from_tuples(
        [(d, m) for d, m in product(dataset_order, model_order)],
        names=["Dataset", "LLM"],
    )

    score_values = df.pivot_table(
        index=method_col,
        columns=[dataset_col, model_col],
        values=metric_name,
        aggfunc=aggfunc,
        dropna=False,
    )

    score_values = score_values.reindex(
        index=method_order,
        columns=expected_cols,
    )
    score_values.index.name = "Method"

    dropped_all_na_conditions = []

    if drop_all_na_conditions:
        all_na_cols = score_values.columns[score_values.isna().all(axis=0)]
        dropped_all_na_conditions = list(all_na_cols.to_list())

        if len(dropped_all_na_conditions) > 0:
            warnings.warn(
                f"[{metric_name}] Dropping all-NA dataset x LLM conditions: "
                + ", ".join(map(str, dropped_all_na_conditions))
            )

        score_values = score_values.loc[:, ~score_values.isna().all(axis=0)]

    original_na_mask = score_values.isna()

    rank_values = score_values.rank(
        axis=0,
        ascending=ascending_flag,
        method="average",
        na_option="bottom",
    )

    summary = pd.DataFrame(index=score_values.index)
    summary["Mean rank"] = rank_values.mean(axis=1, skipna=True)

    if rank_values.shape[1] >= 2:
        summary["Rank SD"] = rank_values.std(axis=1, ddof=1)
    else:
        summary["Rank SD"] = np.nan

    summary["NA count"] = original_na_mask.sum(axis=1).astype(int)

    summary_cols = ["Mean rank"]

    if include_rank_sd:
        summary_cols.append("Rank SD")

    summary_cols.append("NA count")

    sub_raw_table = append_summary_columns(
        core_df=score_values,
        summary_df=summary,
        summary_cols=summary_cols,
    )

    sub_rank_table = append_summary_columns(
        core_df=rank_values,
        summary_df=summary,
        summary_cols=summary_cols,
    )

    return MetricTableOutput(
        metric_name=metric_name,
        direction=direction,
        score_values=score_values,
        rank_values=rank_values,
        summary=summary,
        sub_raw_table=sub_raw_table,
        sub_rank_table=sub_rank_table,
        original_na_mask=original_na_mask,
        dropped_all_na_conditions=dropped_all_na_conditions,
    )


def build_abstention_tables(
    df: pd.DataFrame,
    score_specs: dict[str, str],
    method_order: list[str],
    dataset_order: list[str],
    model_order: list[str],
    method_col: str = "method",
    dataset_col: str = "dataset_name",
    model_col: str = "model_w_params",
    aggfunc: str = "mean",
    drop_all_na_conditions: bool = True,
    include_rank_sd_in_subtable: bool = True,
    main_dispersion_style: str = "pm",
    main_na_display: str = "marker",
    main_na_marker: str = "†",
) -> AbstentionTableOutput:
    """
    Build all appendix and main-paper tables.

    Parameters
    ----------
    score_specs:
        Metric directions, for example:
            {
                "auroc": "larger_is_better",
                "aurc": "smaller_is_better",
            }

    Returns
    -------
    AbstentionTableOutput
        Use:
            output.by_metric[metric].sub_raw_table
            output.by_metric[metric].sub_rank_table
            output.main_mean_rank_table
            output.main_rank_sd_table
            output.main_mean_rank_with_sd_table
            output.main_na_count_table
    """
    if isinstance(score_specs, dict):
        score_items = list(score_specs.items())
    else:
        raise TypeError(f"score_specs must be a dict, got {type(score_specs)}")

    by_metric: dict[str, MetricTableOutput] = {}

    main_mean_rank_table = pd.DataFrame(index=pd.Index(method_order, name="Method"))
    main_rank_sd_table = pd.DataFrame(index=pd.Index(method_order, name="Method"))
    main_na_count_table = pd.DataFrame(index=pd.Index(method_order, name="Method"))

    metadata_rows = []

    for metric_name, direction in score_items:
        metric_output = build_metric_tables(
            df=df,
            metric_name=metric_name,
            direction=direction,
            method_order=method_order,
            dataset_order=dataset_order,
            model_order=model_order,
            method_col=method_col,
            dataset_col=dataset_col,
            model_col=model_col,
            aggfunc=aggfunc,
            drop_all_na_conditions=drop_all_na_conditions,
            include_rank_sd=include_rank_sd_in_subtable,
        )

        by_metric[metric_name] = metric_output

        main_mean_rank_table[metric_name] = metric_output.summary["Mean rank"]
        main_rank_sd_table[metric_name] = metric_output.summary["Rank SD"]
        main_na_count_table[metric_name] = metric_output.summary["NA count"]

        metadata_rows.append(
            {
                "metric": metric_name,
                "direction": direction,
                "n_conditions_used": metric_output.score_values.shape[1],
                "n_dropped_all_na_conditions": len(metric_output.dropped_all_na_conditions),
                "dropped_all_na_conditions": "; ".join(map(str, metric_output.dropped_all_na_conditions)),
            }
        )

    metric_metadata = pd.DataFrame(metadata_rows)

    main_mean_rank_with_sd_table = make_main_mean_rank_with_sd_table(
        mean_rank_df=main_mean_rank_table,
        rank_sd_df=main_rank_sd_table,
        na_count_df=main_na_count_table,
        style=main_dispersion_style,
        na_display=main_na_display,
        na_marker=main_na_marker,
    )

    return AbstentionTableOutput(
        by_metric=by_metric,
        main_mean_rank_table=main_mean_rank_table,
        main_rank_sd_table=main_rank_sd_table,
        main_na_count_table=main_na_count_table,
        main_mean_rank_with_sd_table=main_mean_rank_with_sd_table,
        metric_metadata=metric_metadata,
    )


# =============================================================================
# CSV output
# =============================================================================


def save_abstention_tables_to_csv(
    output: AbstentionTableOutput,
    output_dir: str | Path,
    flatten_columns: bool = False,
    save_original_na_mask: bool = True,
    encoding: str = "utf-8-sig",
) -> None:
    """
    Save all generated tables as CSV.

    Directory structure
    -------------------
    output_dir/
        sub/
            <metric>_raw_table.csv
            <metric>_rank_table.csv
            <metric>_original_na_mask.csv
        main/
            main_mean_rank_table.csv
            main_rank_sd_table.csv
            main_na_count_table.csv
            main_mean_rank_with_sd_table.csv
            metric_metadata.csv

    Parameters
    ----------
    flatten_columns:
        If False, sub CSV files keep MultiIndex columns, which creates a two-row header in CSV.
        If True, MultiIndex columns are flattened into one header row.
    """
    output_dir = Path(output_dir)
    sub_dir = output_dir / "sub"
    main_dir = output_dir / "main"

    sub_dir.mkdir(parents=True, exist_ok=True)
    main_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, metric_output in output.by_metric.items():
        safe_name = _safe_filename(metric_name)

        raw_table = metric_output.sub_raw_table
        rank_table = metric_output.sub_rank_table
        na_mask = metric_output.original_na_mask.astype(int)

        if flatten_columns:
            raw_table = flatten_columns_for_csv(raw_table)
            rank_table = flatten_columns_for_csv(rank_table)
            na_mask = flatten_columns_for_csv(na_mask)

        raw_table.to_csv(
            sub_dir / f"{safe_name}_raw_table.csv",
            encoding=encoding,
        )

        rank_table.to_csv(
            sub_dir / f"{safe_name}_rank_table.csv",
            encoding=encoding,
        )

        if save_original_na_mask:
            na_mask.to_csv(
                sub_dir / f"{safe_name}_original_na_mask.csv",
                encoding=encoding,
            )

    output.main_mean_rank_table.to_csv(
        main_dir / "main_mean_rank_table.csv",
        encoding=encoding,
    )

    output.main_rank_sd_table.to_csv(
        main_dir / "main_rank_sd_table.csv",
        encoding=encoding,
    )

    output.main_na_count_table.to_csv(
        main_dir / "main_na_count_table.csv",
        encoding=encoding,
    )

    output.main_mean_rank_with_sd_table.to_csv(
        main_dir / "main_mean_rank_with_sd_table.csv",
        encoding=encoding,
    )

    output.metric_metadata.to_csv(
        main_dir / "metric_metadata.csv",
        index=False,
        encoding=encoding,
    )


# =============================================================================
# Convenience function
# =============================================================================


def build_and_save_abstention_tables(
    df: pd.DataFrame,
    score_specs: dict[str, str],
    method_order: list[str],
    dataset_order: list[str],
    model_order: list[str],
    output_dir: str | Path,
    method_col: str = "method",
    dataset_col: str = "dataset_name",
    model_col: str = "model_w_params",
    aggfunc: str = "mean",
    drop_all_na_conditions: bool = True,
    include_rank_sd_in_subtable: bool = True,
    main_dispersion_style: str = "pm",
    main_na_display: str = "marker",
    main_na_marker: str = "†",
    flatten_columns: bool = False,
    save_original_na_mask: bool = True,
    encoding: str = "utf-8-sig",
) -> AbstentionTableOutput:
    """
    Build all tables and immediately save them to CSV.

    This is a convenience wrapper around:
        build_abstention_tables(...)
        save_abstention_tables_to_csv(...)
    """
    output = build_abstention_tables(
        df=df,
        score_specs=score_specs,
        method_order=method_order,
        dataset_order=dataset_order,
        model_order=model_order,
        method_col=method_col,
        dataset_col=dataset_col,
        model_col=model_col,
        aggfunc=aggfunc,
        drop_all_na_conditions=drop_all_na_conditions,
        include_rank_sd_in_subtable=include_rank_sd_in_subtable,
        main_dispersion_style=main_dispersion_style,
        main_na_display=main_na_display,
        main_na_marker=main_na_marker,
    )

    save_abstention_tables_to_csv(
        output=output,
        output_dir=output_dir,
        flatten_columns=flatten_columns,
        save_original_na_mask=save_original_na_mask,
        encoding=encoding,
    )

    return output
