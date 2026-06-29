import os
import csv
import re

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
    key = name.strip()

    if key.startswith("eval-"):
        key = key[len("eval-"):]

    key = key.lower().replace("-", "_")

    # 这里只去掉和数据集本身无关的后缀
    suffixes = ["_sem_seg", "_all"]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
                changed = True

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


def parse_dataset_and_mode(folder_name):
    """
    例如:
      eval-iSAID_noslide -> ("iSAID", "noslide")
      eval-iSAID_slide   -> ("iSAID", "slide")
      eval-iSAID         -> ("iSAID", "unknown")
    """
    name = folder_name.strip()

    if name.startswith("eval-"):
        name = name[len("eval-"):]

    raw = name.lower().replace("-", "_")
    mode = "unknown"

    if raw.endswith("_noslide"):
        raw = raw[:-len("_noslide")]
        mode = "noslide"
    elif raw.endswith("_slide"):
        raw = raw[:-len("_slide")]
        mode = "slide"

    dataset_name = normalize_dataset_name(raw)
    return dataset_name, mode


# =========================
# 3. 从 log.txt 提取 mIoU / mACC
# =========================
def parse_metric_line(line):
    line = line.strip()
    if not line:
        return None, None

    lower_line = line.lower()
    if "copypaste:" in lower_line:
        idx = lower_line.rfind("copypaste:")
        line = line[idx + len("copypaste:"):].strip()
    elif ":" in line:
        line = line.rsplit(":", 1)[-1].strip()

    parts = [p.strip() for p in line.split(",") if p.strip()]

    if len(parts) >= 3:
        try:
            miou = float(parts[0])
            macc = float(parts[2])
            return miou, macc
        except Exception:
            pass

    nums = re.findall(r"[-+]?\d*\.?\d+", line)
    if len(nums) >= 3:
        try:
            miou = float(nums[0])
            macc = float(nums[2])
            return miou, macc
        except Exception:
            pass

    return None, None


def extract_metrics_from_log(log_path):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    if not lines:
        return None, None

    for line in reversed(lines):
        if "copypaste" in line.lower():
            miou, macc = parse_metric_line(line)
            if miou is not None and macc is not None:
                return miou, macc

    return parse_metric_line(lines[-1])


# =========================
# 4. 扫描 eval 目录
# =========================
def scan_eval_dirs(base_dir):
    eval_dirs = []
    if not os.path.isdir(base_dir):
        return eval_dirs

    for sub in sorted(os.listdir(base_dir)):
        sub_path = os.path.join(base_dir, sub)
        if os.path.isdir(sub_path) and sub.startswith("eval-"):
            eval_dirs.append((sub, sub_path))

    return eval_dirs


# =========================
# 5. 收集结果
# =========================
def collect_results(root_dir, experiment_name):
    """
    返回格式:
    [experiment_name, task_name, dataset_name, eval_mode, miou, macc]
    """
    results = []

    if not os.path.isdir(root_dir):
        print(f"[Warning] 路径不存在: {root_dir}")
        return results

    seen_log_paths = set()
    root_task_name = os.path.basename(os.path.normpath(root_dir))

    # 情况 1：root_dir 下面直接就是 eval-xxx
    for sub, eval_path in scan_eval_dirs(root_dir):
        log_path = os.path.join(eval_path, "log.txt")
        if not os.path.exists(log_path):
            print(f"[Warning] 缺少 log.txt: {log_path}")
            continue

        seen_log_paths.add(os.path.abspath(log_path))

        dataset_name, eval_mode = parse_dataset_and_mode(sub)
        miou, macc = extract_metrics_from_log(log_path)

        if miou is not None and macc is not None:
            results.append([experiment_name, root_task_name, dataset_name, eval_mode, miou, macc])
        else:
            print(f"[Warning] 解析失败: {log_path}")

    # 情况 2：root_dir/task_name/eval-xxx
    for task_name in sorted(os.listdir(root_dir)):
        task_path = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        for sub, eval_path in scan_eval_dirs(task_path):
            log_path = os.path.join(eval_path, "log.txt")
            abs_log_path = os.path.abspath(log_path)

            if abs_log_path in seen_log_paths:
                continue

            if not os.path.exists(log_path):
                print(f"[Warning] 缺少 log.txt: {log_path}")
                continue

            dataset_name, eval_mode = parse_dataset_and_mode(sub)
            miou, macc = extract_metrics_from_log(log_path)

            if miou is not None and macc is not None:
                results.append([experiment_name, task_name, dataset_name, eval_mode, miou, macc])
            else:
                print(f"[Warning] 解析失败: {log_path}")

    return results


# =========================
# 6. 透视成宽表（按 mode 分开）
# =========================
def build_result_map(results):
    """
    results:
      [experiment_name, task_name, dataset_name, eval_mode, miou, macc]

    返回:
      result_map[eval_mode][experiment_name][dataset_name] = (miou, macc, task_name)
    """
    result_map = {}

    for exp_name, task_name, dataset_name, eval_mode, miou, macc in results:
        if eval_mode not in result_map:
            result_map[eval_mode] = {}
        if exp_name not in result_map[eval_mode]:
            result_map[eval_mode][exp_name] = {}

        result_map[eval_mode][exp_name][dataset_name] = (miou, macc, task_name)

    return result_map


# =========================
# 7. 打印横向宽表
# =========================
def print_wide_table(result_map, datasets, title=""):
    row_name_w = 28
    metric_w = 12
    pair_w = metric_w * 2
    total_w = row_name_w + pair_w * (len(datasets) + 1)

    if title:
        print(f"\n{title}")

    print("=" * total_w)

    header_1 = f"{'Task':<{row_name_w}}"
    for ds in datasets:
        header_1 += f"{ds:^{pair_w}}"
    header_1 += f"{'Mean':^{pair_w}}"
    print(header_1)

    header_2 = " " * row_name_w
    for _ in datasets:
        header_2 += f"{'mIoU':>{metric_w}}{'mACC':>{metric_w}}"
    header_2 += f"{'mIoU':>{metric_w}}{'mACC':>{metric_w}}"
    print(header_2)

    print("-" * total_w)

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
# 8. 保存宽表 CSV
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
    exp_roots = {
        "output_vitb_384_DLRSD": "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI-V1/output_vitb_384_DLRSD",
        "output_vitb_384_iSAID": "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI-V1/output_vitb_384_iSAID",
        "output_vitl_336_DLRSD": "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI-V1/output_vitl_336_DLRSD",
        "output_vitl_336_iSAID": "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI-V1/output_vitl_336_iSAID",
    }

    exp_dirs = {}
    for exp_name, base_dir in exp_roots.items():
        for seed in range(6):   # 0,1,2,3,4,5
            seed_dir = os.path.join(base_dir, str(seed))
            exp_dirs[f"{exp_name}_{seed}"] = seed_dir

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

    # noslide
    noslide_map = result_map.get("noslide", {})
    if noslide_map:
        print_wide_table(
            noslide_map,
            GROUP_A,
            title="\n[Table A - NoSlide]"
        )
        print_wide_table(
            noslide_map,
            GROUP_B,
            title="\n[Table B - NoSlide]"
        )

    # slide
    slide_map = result_map.get("slide", {})
    if slide_map:
        print_wide_table(
            slide_map,
            GROUP_A,
            title="\n[Table A - Slide]"
        )
        print_wide_table(
            slide_map,
            GROUP_B,
            title="\n[Table B - Slide]"
        )

    # 保存 CSV：建议单独放 summary 目录，不要写进 seed 0 目录里
    summary_dir = "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI-V1"
    os.makedirs(summary_dir, exist_ok=True)

    if noslide_map:
        save_wide_csv(
            noslide_map, GROUP_A,
            os.path.join(summary_dir, "summary_group_A_noslide_wide.csv")
        )
        save_wide_csv(
            noslide_map, GROUP_B,
            os.path.join(summary_dir, "summary_group_B_noslide_wide.csv")
        )

    if slide_map:
        save_wide_csv(
            slide_map, GROUP_A,
            os.path.join(summary_dir, "summary_group_A_slide_wide.csv")
        )
        save_wide_csv(
            slide_map, GROUP_B,
            os.path.join(summary_dir, "summary_group_B_slide_wide.csv")
        )

    print("\nDone.")