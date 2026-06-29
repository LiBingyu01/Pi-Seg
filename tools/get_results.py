import os
import csv

# =========================
# 1. 两组数据集顺序
# =========================
GROUP_A = [
    "WHU-BD", "WHU-SAT", "Inria", "xBD",
    "CHN6-CUG", "DeepGlobe", "Massachusetts", "SpaceNet", "WBS-SI"
]

GROUP_B = [
    "DLRSD", "FLAIR", "iSAID", "LoveDA", "OpenEarthMap",
    "Potsdam", "UAVid", "UDD5", "Vaihingen", "VDD"
]


# =========================
# 2. 数据集名字标准化
# =========================
def normalize_dataset_name(name):
    key = name.strip().lower().replace("-", "_")

    alias_map = {
        "whu_bd": "WHU-BD",
        "whu_sat": "WHU-SAT",
        "inria": "Inria",
        "xbd": "xBD",
        "chn6_cug": "CHN6-CUG",
        "deepglobe": "DeepGlobe",
        "massachusetts": "Massachusetts",
        "spacenet": "SpaceNet",
        "wbs_si": "WBS-SI",
        "wbs_si_val": "WBS-SI",

        "dlrsd": "DLRSD",
        "flair": "FLAIR",
        "isaid": "iSAID",
        "loveda": "LoveDA",
        "openearthmap": "OpenEarthMap",
        "potsdam": "Potsdam",
        "uavid": "UAVid",
        "udd5": "UDD5",
        "vaihingen": "Vaihingen",
        "vdd": "VDD",
    }

    return alias_map.get(key, name)


# =========================
# 3. 从 log.txt 提取 mIoU / mACC
# =========================
def extract_metrics_from_log(log_path):
    """
    从 log.txt 最后一行提取 mIoU 和 mACC
    默认最后一行格式类似:
    0.1234,0.5678,0.9101
    其中:
      parts[0] = mIoU
      parts[2] = mACC
    """
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        if not lines:
            return None, None

        last_line = lines[-1].strip()

        if ":" in last_line:
            last_line = last_line.split(":")[-1].strip()

        parts = [p.strip() for p in last_line.split(",")]

        if len(parts) < 3:
            return None, None

        try:
            miou = float(parts[0])
            macc = float(parts[2])
            return miou, macc
        except Exception:
            return None, None


# =========================
# 4. 收集结果
# =========================
def collect_results(root_dir, experiment_name):
    """
    返回格式:
    [experiment_name, task_name, dataset_name, miou, macc]
    """
    results = []

    if not os.path.isdir(root_dir):
        print(f"[Warning] 路径不存在: {root_dir}")
        return results

    for task_name in os.listdir(root_dir):
        task_path = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        for sub in os.listdir(task_path):
            if not sub.startswith("eval-"):
                continue

            raw_dataset_name = sub.replace("eval-", "")
            dataset_name = normalize_dataset_name(raw_dataset_name)

            eval_path = os.path.join(task_path, sub)
            log_path = os.path.join(eval_path, "log.txt")

            if not os.path.exists(log_path):
                print(f"[Warning] 缺少 log.txt: {log_path}")
                continue

            miou, macc = extract_metrics_from_log(log_path)

            if miou is not None and macc is not None:
                results.append([experiment_name, task_name, dataset_name, miou, macc])
            else:
                print(f"[Warning] 解析失败: {log_path}")

    return results


# =========================
# 5. 透视成宽表
# =========================
def build_result_map(results):
    """
    results:
      [experiment_name, task_name, dataset_name, miou, macc]

    返回:
      result_map[experiment_name][dataset_name] = (miou, macc, task_name)
    """
    result_map = {}

    for exp_name, task_name, dataset_name, miou, macc in results:
        if exp_name not in result_map:
            result_map[exp_name] = {}
        result_map[exp_name][dataset_name] = (miou, macc, task_name)

    return result_map


# =========================
# 6. 打印横向宽表
# =========================
def print_wide_table(result_map, datasets, title=""):
    """
    输出格式类似:
    ======================================================================
    Task        WHU-BD ... Mean
                mIoU mACC ...
    ----------------------------------------------------------------------
    exp_name    ...
    """
    row_name_w = 28
    metric_w = 12
    pair_w = metric_w * 2
    total_w = row_name_w + pair_w * (len(datasets) + 1)

    if title:
        print(f"\n{title}")

    print("=" * total_w)

    # 第一行表头：Task + 各数据集 + Mean
    header_1 = f"{'Task':<{row_name_w}}"
    for ds in datasets:
        header_1 += f"{ds:^{pair_w}}"
    header_1 += f"{'Mean':^{pair_w}}"
    print(header_1)

    # 第二行表头：每个数据集下 mIoU / mACC
    header_2 = " " * row_name_w
    for _ in datasets:
        header_2 += f"{'mIoU':>{metric_w}}{'mACC':>{metric_w}}"
    header_2 += f"{'mIoU':>{metric_w}}{'mACC':>{metric_w}}"
    print(header_2)

    print("-" * total_w)

    # 每一行实验
    for exp_name in sorted(result_map.keys()):
        row = f"{exp_name:<{row_name_w}}"

        miou_list = []
        macc_list = []

        for ds in datasets:
            if ds in result_map[exp_name]:
                miou, macc, _ = result_map[exp_name][ds]
                row += f"{miou:>{metric_w}.4f}{macc:>{metric_w}.4f}"
                miou_list.append(miou)
                macc_list.append(macc)
            else:
                row += f"{'-':>{metric_w}}{'-':>{metric_w}}"

        mean_miou = sum(miou_list) / len(miou_list) if miou_list else 0.0
        mean_macc = sum(macc_list) / len(macc_list) if macc_list else 0.0
        row += f"{mean_miou:>{metric_w}.4f}{mean_macc:>{metric_w}.4f}"

        print(row)


# =========================
# 7. 保存宽表 CSV
# =========================
def save_wide_csv(result_map, datasets, out_csv_path):
    headers = ["Experiment"]
    for ds in datasets:
        headers.extend([f"{ds}_mIoU", f"{ds}_mACC"])
    headers.extend(["Mean_mIoU", "Mean_mACC"])

    rows = []
    for exp_name in sorted(result_map.keys()):
        row = [exp_name]
        miou_list = []
        macc_list = []

        for ds in datasets:
            if ds in result_map[exp_name]:
                miou, macc, _ = result_map[exp_name][ds]
                row.extend([f"{miou:.4f}", f"{macc:.4f}"])
                miou_list.append(miou)
                macc_list.append(macc)
            else:
                row.extend(["", ""])

        mean_miou = sum(miou_list) / len(miou_list) if miou_list else 0.0
        mean_macc = sum(macc_list) / len(macc_list) if macc_list else 0.0
        row.extend([f"{mean_miou:.4f}", f"{mean_macc:.4f}"])
        rows.append(row)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    exp_dirs = {
        
        # "noise_img_gaussian_txt_gaussian": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_gaussian_txt_gaussian",
        # "noise_img_gaussian_txt_laplace": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_gaussian_txt_laplace",
        # "noise_img_gaussian_txt_uniform": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_gaussian_txt_uniform",
        # "noise_img_gaussian_txt_student_t": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_gaussian_txt_student_t",
        # "noise_img_laplace_txt_gaussian": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_laplace_txt_gaussian",
        # "noise_img_laplace_txt_laplace": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_laplace_txt_laplace",
        # "noise_img_laplace_txt_uniform": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_laplace_txt_uniform",
        # "noise_img_laplace_txt_student_t": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_laplace_txt_student_t",
        # "noise_img_uniform_txt_gaussian": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_uniform_txt_gaussian",
        # "noise_img_uniform_txt_laplace": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_uniform_txt_laplace",
        # "noise_img_uniform_txt_uniform": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_uniform_txt_uniform",
        # "noise_img_uniform_txt_student_t": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_uniform_txt_student_t",
        # "noise_img_student_t_txt_gaussian": "output/ablation_student_t/img_student_t_df_3p0_txt_gaussian",
        # "noise_img_student_t_txt_laplace": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_student_t_txt_laplace",
        # "noise_img_student_t_txt_uniform": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_student_t_txt_uniform",
        # "noise_img_student_t_txt_student_t": "output/ablation_pini_agg_不同的噪声组合_l/noise_img_student_t_txt_student_t",
               
        # "noise_img_gaussian_txt_gaussian": "output/ablation_pini_agg_l/noise_img_gaussian_txt_gaussian",
        # "noise_img_gaussian_txt_laplace": "output/ablation_pini_agg_l/noise_img_gaussian_txt_laplace",
        # "noise_img_gaussian_txt_uniform": "output/ablation_pini_agg_l/noise_img_gaussian_txt_uniform",
        # "noise_img_laplace_txt_gaussian": "output/ablation_pini_agg_l/noise_img_laplace_txt_gaussian",
        # "noise_img_laplace_txt_laplace": "output/ablation_pini_agg_l/noise_img_laplace_txt_laplace",
        # "noise_img_laplace_txt_uniform": "output/ablation_pini_agg_l/noise_img_laplace_txt_uniform",
        # "noise_img_uniform_txt_gaussian": "output/ablation_pini_agg_l/noise_img_uniform_txt_gaussian",
        # "noise_img_uniform_txt_laplace": "output/ablation_pini_agg_l/noise_img_uniform_txt_laplace",
        # "noise_img_uniform_txt_uniform": "output/ablation_pini_agg_l/noise_img_uniform_txt_uniform",
        
        # "img_gaussian_txt_student_t_df_3p0": "output/ablation_student_t/img_gaussian_txt_student_t_df_3p0",
        # "img_gaussian_txt_student_t_df_5p0": "output/ablation_student_t/img_gaussian_txt_student_t_df_5p0",
        # "img_gaussian_txt_student_t_df_10p0": "output/ablation_student_t/img_gaussian_txt_student_t_df_10p0",
        # "img_student_t_df_3p0_txt_gaussian": "output/ablation_student_t/img_student_t_df_3p0_txt_gaussian",
        # "img_student_t_df_5p0_txt_gaussian": "output/ablation_student_t/img_student_t_df_5p0_txt_gaussian",
        # "img_student_t_df_10p0_txt_gaussian": "output/ablation_student_t/img_student_t_df_10p0_txt_gaussian",
        
        # "reduction_1": "output/ablation_pini_agg_reduce和std/reduction_1",
        # "reduction_2": "output/ablation_pini_agg_reduce和std/reduction_2",
        # "reduction_4": "output/ablation_pini_agg_reduce和std/reduction_4",
        # "reduction_8": "output/ablation_pini_agg_reduce和std/reduction_8",
        # "reduction_8": "output/ablation_pini_agg_reduce和std/reduction_8",
        # "text_noise_std_0p0": "output/ablation_pini_agg_reduce和std/text_noise_std_0p0",
        # "text_noise_std_0p005": "output/ablation_pini_agg_reduce和std/text_noise_std_0p005",
        # "text_noise_std_0p01": "output/ablation_pini_agg_reduce和std/text_noise_std_0p01",
        # "text_noise_std_0p02": "output/ablation_pini_agg_reduce和std/text_noise_std_0p02",
        # "text_noise_std_0p05": "output/ablation_pini_agg_reduce和std/text_noise_std_0p05",
        
        
        "baseline_no_pini": "output/ablation_pini_baseline_image_text_b/baseline_no_pini",
        "text_vpn_only": "output/ablation_pini_baseline_image_text_b/text_vpn_only",
        "image_vpn_only": "output/ablation_pini_baseline_image_text_b/image_vpn_only",
        "full_pini": "output/ablation_pini_baseline_image_text_b/full_pini",
        
        # "output_vitb_384_OVRSIS95K_time_0": "output_vitb_384_OVRSIS95K_time_0",
        # "output_vitb_384_OVRSIS95K_time_1": "output_vitb_384_OVRSIS95K_time_1",
        # "output_vitb_384_OVRSIS95K_time_2": "output_vitb_384_OVRSIS95K_time_2",
        # "output_vitb_384_OVRSIS95K_time_3": "output_vitb_384_OVRSIS95K_time_3",
        # "output_vitb_384_OVRSIS95K_time_4": "output_vitb_384_OVRSIS95K_time_4",
        # "output_vitb_384_OVRSIS95K_time_5": "output_vitb_384_OVRSIS95K_time_5",
        # "output_vitb_384_OVRSIS95K_time_6": "output_vitb_384_OVRSIS95K_time_6",
        # "output_vitb_384_OVRSIS95K_time_7": "output_vitb_384_OVRSIS95K_time_7",
        # "output_vitb_384_OVRSIS95K_time_8": "output_vitb_384_OVRSIS95K_time_8",
        # "output_vitb_384_OVRSIS95K_time_9": "output_vitb_384_OVRSIS95K_time_9",
        # "BASELINE": "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI/output_vitb_384_OVRSIS95K_baseline"
    }
    

    all_results = []

    for exp_name, root_dir in exp_dirs.items():
        print(f"\n==================== {exp_name} ====================")
        results = collect_results(root_dir, exp_name)

        if not results:
            print("没有找到可用结果")
            continue

        all_results.extend(results)

    if not all_results:
        print("没有可输出的结果")
        exit(0)

    result_map = build_result_map(all_results)

    # 第一张表：Building / Road / Flood
    print_wide_table(
        result_map,
        GROUP_A,
        title="\n[Table A: WHU-BD / WHU-SAT / Inria / xBD / CHN6-CUG / DeepGlobe / Massachusetts / SpaceNet / WBS-SI]"
    )

    # 第二张表：OVRSIS
    print_wide_table(
        result_map,
        GROUP_B,
        title="\n[Table B: DLRSD / FLAIR / iSAID / LoveDA / OpenEarthMap / Potsdam / UAVid / UDD5 / Vaihingen / VDD]"
    )

    # 保存 CSV
    first_root_dir = list(exp_dirs.values())[0]
    out_csv_a = os.path.join(first_root_dir, "summary_group_A_wide.csv")
    out_csv_b = os.path.join(first_root_dir, "summary_group_B_wide.csv")

    save_wide_csv(result_map, GROUP_A, out_csv_a)
    save_wide_csv(result_map, GROUP_B, out_csv_b)

    print(f"\n已保存宽表 CSV: {out_csv_a}")
    print(f"已保存宽表 CSV: {out_csv_b}")