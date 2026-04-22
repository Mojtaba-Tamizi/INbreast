from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_history_csv(run_dir: Path) -> Path:
    history_csv = run_dir / "logs" / "history.csv"
    if not history_csv.exists():
        raise FileNotFoundError(f"Could not find history.csv at: {history_csv}")
    return history_csv


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def save_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    ylabel: str,
    save_path: Path,
    smooth_window: int = 1,
) -> None:
    if len(y_cols) == 0:
        return

    plt.figure(figsize=(8, 5))

    for col in y_cols:
        y = smooth_series(df[col], smooth_window)
        plt.plot(df[x_col], y, label=col)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def find_metric_bases(columns: list[str]) -> list[str]:
    bases = set()

    for col in columns:
        if col.startswith("train_"):
            base = col[len("train_") :]
            if base not in {"total_loss", "lr"}:
                bases.add(base)
        elif col.startswith("val_"):
            base = col[len("val_") :]
            if base not in {"total_loss", "lr"}:
                bases.add(base)

    return sorted(bases)


def main(
    history_csv_path: str | None,
    run_dir_path: str | None,
    output_dir_path: str | None,
    smooth_window: int,
) -> None:
    if history_csv_path is None and run_dir_path is None:
        raise ValueError("Provide either --history-csv or --run-dir")

    if history_csv_path is not None:
        history_csv = Path(history_csv_path)
        if not history_csv.exists():
            raise FileNotFoundError(f"history.csv not found: {history_csv}")
        default_output_dir = history_csv.parent.parent / "plots"
    else:
        run_dir = Path(run_dir_path)
        history_csv = infer_history_csv(run_dir)
        default_output_dir = run_dir / "plots"

    output_dir = Path(output_dir_path) if output_dir_path is not None else default_output_dir
    ensure_dir(output_dir)

    df = pd.read_csv(history_csv)
    if len(df) == 0:
        raise ValueError(f"history.csv is empty: {history_csv}")
    if "epoch" not in df.columns:
        raise ValueError("history.csv must contain an 'epoch' column")

    columns = df.columns.tolist()

    # 1) total loss
    loss_cols = [col for col in ["train_total_loss", "val_total_loss"] if col in columns]
    save_plot(
        df=df,
        x_col="epoch",
        y_cols=loss_cols,
        title="Loss",
        ylabel="Loss",
        save_path=output_dir / "loss.png",
        smooth_window=smooth_window,
    )

    # 2) learning rate
    if "lr" in columns:
        save_plot(
            df=df,
            x_col="epoch",
            y_cols=["lr"],
            title="Learning Rate",
            ylabel="LR",
            save_path=output_dir / "learning_rate.png",
            smooth_window=1,
        )

    # 3) metrics
    metric_bases = find_metric_bases(columns)

    for base in metric_bases:
        metric_cols = []
        train_col = f"train_{base}"
        val_col = f"val_{base}"

        if train_col in columns:
            metric_cols.append(train_col)
        if val_col in columns:
            metric_cols.append(val_col)

        if len(metric_cols) == 0:
            continue

        save_plot(
            df=df,
            x_col="epoch",
            y_cols=metric_cols,
            title=base,
            ylabel=base,
            save_path=output_dir / f"{base}.png",
            smooth_window=smooth_window,
        )

    # 4) combined overview of common segmentation metrics
    overview_candidates = [
        "train_mask_dice", "val_mask_dice",
        "train_mask_iou", "val_mask_iou",
        "train_mask_sensitivity", "val_mask_sensitivity",
        "train_mask_specificity", "val_mask_specificity",
        "train_mask_hd95", "val_mask_hd95",
    ]
    overview_cols = [col for col in overview_candidates if col in columns]

    if len(overview_cols) > 0:
        save_plot(
            df=df,
            x_col="epoch",
            y_cols=overview_cols,
            title="Segmentation Metrics Overview",
            ylabel="Metric Value",
            save_path=output_dir / "metrics_overview.png",
            smooth_window=smooth_window,
        )

    print(f"[DONE] Plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs from history.csv")
    parser.add_argument(
        "--history-csv",
        type=str,
        default=None,
        help="Path to logs/history.csv",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to training run directory containing logs/history.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Rolling mean window for smoothing",
    )
    args = parser.parse_args()

    main(
        history_csv_path=args.history_csv,
        run_dir_path=args.run_dir,
        output_dir_path=args.output_dir,
        smooth_window=args.smooth_window,
    )