import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def to_float_array(x):
    """统一转成 numpy float，避免 matplotlib fill_between 类型报错。"""
    if isinstance(x, pd.Series):
        x = pd.to_numeric(x, errors="coerce")
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def main():
    csv_path = "output/ablation_pini_agg/noise_img_gaussian_txt_gaussian/delta_corr_logs/delta_corr_step_log.csv"
    save_path = "output/ablation_pini_agg/noise_img_gaussian_txt_gaussian/delta_corr_logs/rolling_mean_metrics_with_baseline_paper_style_final.png"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    # 必要列检查
    required_cols = ["step", "gt_in_mean", "non_gt_mean"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要列: {col}")

    # 只保留需要列，并转数值
    df = df[required_cols].copy()
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 去掉关键列缺失
    df = df.dropna(subset=required_cols)

    # 排序，避免 step 乱序
    df = df.sort_values("step").reset_index(drop=True)

    # 重新计算指标
    eps = 1e-6
    df["gap"] = df["gt_in_mean"] - df["non_gt_mean"]
    df["align_ratio"] = df["gt_in_mean"] / (df["non_gt_mean"].abs() + eps)

    metrics = ["gt_in_mean", "non_gt_mean", "gap", "align_ratio"]

    # 全局字体设置
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "STIXGeneral", "CMU Serif"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    # 配色
    raw_color = "#9ec3e6"       # 浅蓝：原始曲线
    smooth_color = "#ff7f0e"    # 橙色：滚动平均
    baseline_color = "#4c90d9"  # 蓝色：基准线
    better_region_color = "#f3c178"

    # 按 step 聚合平均
    step_mean = df.groupby("step", as_index=False)[metrics].mean()

    baseline_map = {
        "gt_in_mean": 0.0,
        "non_gt_mean": 0.0,
        "gap": 0.0,
        "align_ratio": 1.0,
    }

    better_direction = {
        "gt_in_mean": "higher",
        "non_gt_mean": "lower",
        "gap": "higher",
        "align_ratio": "higher",
    }

    title_map = {
        "gt_in_mean": r"$gt\_in\_mean$ ($>0$ is better)",
        "non_gt_mean": r"$non\_gt\_mean$ ($<0$ is better)",
        "gap": r"$gap=gt\_in\_mean-non\_gt\_mean$ ($>0$ is better)",
        "align_ratio": r"$align\_ratio=\frac{gt\_in\_mean}{|non\_gt\_mean|+\epsilon}$ ($>1$ is better)",
    }

    ylabel_map = {
        "gt_in_mean": "gt_in_mean",
        "non_gt_mean": "non_gt_mean",
        "gap": "gap",
        "align_ratio": "align_ratio",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        x_series = step_mean["step"].copy()
        y_series = step_mean[metric].copy()

        # 转 numeric，防止 dtype=object 导致 fill_between 报错
        x_series = pd.to_numeric(x_series, errors="coerce")
        y_series = pd.to_numeric(y_series, errors="coerce")

        # 清除 inf / nan
        valid = np.isfinite(x_series.to_numpy()) & np.isfinite(y_series.to_numpy())
        x_series = x_series[valid].reset_index(drop=True)
        y_series = y_series[valid].reset_index(drop=True)

        # 第四个图单独处理：更强裁剪 + 更大窗口 + 单独 y 轴
        if metric == "align_ratio":
            low = y_series.quantile(0.05)
            high = y_series.quantile(0.95)
            y_series = y_series.clip(lower=low, upper=high)
            cur_window = 800
        else:
            cur_window = 500

        y_smooth_series = y_series.rolling(window=cur_window, min_periods=1).mean()

        # 再转成纯 float ndarray，确保 fill_between 不报错
        x = to_float_array(x_series)
        y = to_float_array(y_series)
        y_smooth = to_float_array(y_smooth_series)

        baseline = float(baseline_map[metric])
        baseline_arr = np.full_like(x, baseline, dtype=float)

        # 最终有效 mask
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_smooth) & np.isfinite(baseline_arr)
        x = x[mask]
        y = y[mask]
        y_smooth = y_smooth[mask]
        baseline_arr = baseline_arr[mask]

        # 原始曲线（淡）
        ax.plot(
            x,
            y,
            color=raw_color,
            linewidth=0.8,
            linestyle="-",
            alpha=0.45,
            label="Raw mean"
        )

        # 滚动平均曲线
        ax.plot(
            x,
            y_smooth,
            color=smooth_color,
            linewidth=2.0,
            linestyle="-",
            alpha=0.98,
            label=f"Rolling mean (w={cur_window})"
        )

        # 基准线
        ax.axhline(
            y=baseline,
            color=baseline_color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.95,
            label=f"Baseline = {baseline:g}"
        )

        # Better region
        if better_direction[metric] == "higher":
            cond = y_smooth >= baseline
        else:
            cond = y_smooth <= baseline

        ax.fill_between(
            x,
            y_smooth,
            baseline_arr,
            where=cond,
            color=better_region_color,
            alpha=0.22,
            interpolate=True,
            label="Better region"
        )

        ax.set_title(title_map[metric], pad=8)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel_map[metric])

        # 第四张图单独增强可读性
        if metric == "align_ratio":
            ax.set_ylim(-1, 3)

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28, color="#b0b0b0")
        ax.legend(loc="best", frameon=False)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#444444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()