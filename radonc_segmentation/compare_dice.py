# ============================================================
# compare_dice.py — сравнение сегментаций по Dice на каждый орган
# ------------------------------------------------------------
# Режим 1. Два файла:
#   python compare_dice.py segA.nii.gz segB.nii.gz
#   (напр. legacy vs aligned: seg_nnunet\P110.nii.gz vs P110_aligned.nii.gz)
#
# Режим 2. Валидация с ground truth (запускает ОБА варианта инференса
# на N случаях из validation-набора и считает Dice против GT):
#   python compare_dice.py --validation 2
#   ВНИМАНИЕ: долго (каждый случай = 2 полных инференса, ~30-60 мин на
#   пару на i5-9400T). Удобно запускать на ночь.
#
# Вывод: таблица Dice по каждому органу + среднее; при --csv ещё и CSV-файл.
# ============================================================
import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
import SimpleITK as sitk

from radonc_segmentation import config

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return None  # орган отсутствует в обеих масках
    return 2.0 * np.logical_and(a, b).sum() / denom


def compare_files(file_a, file_b):
    img_a = sitk.GetArrayFromImage(sitk.ReadImage(str(file_a)))
    img_b = sitk.GetArrayFromImage(sitk.ReadImage(str(file_b)))
    if img_a.shape != img_b.shape:
        print(f"!! Размеры не совпадают: {img_a.shape} vs {img_b.shape} — "
              f"сравнение в исходной сетке файла B невозможно. "
              f"Используйте пары файлов в одинаковой геометрии.")
        return None
    labels = sorted(set(np.unique(img_a)) | set(np.unique(img_b)))
    labels = [int(v) for v in labels if int(v) > 0]
    name_by_value = {v: k for k, v in config.LABELS.items()}
    rows = []
    for lbl in labels:
        d = dice(img_a == lbl, img_b == lbl)
        rows.append((lbl, name_by_value.get(lbl, f"label_{lbl}"), d))
    return rows


def print_table(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'label':>5}  {'орган':<20} {'Dice':>7}")
    mean_vals = []
    for lbl, name, d in rows:
        ds = "n/a" if d is None else f"{d:.3f}"
        if d is not None:
            mean_vals.append(d)
        print(f"{lbl:>5}  {name:<20} {ds:>7}")
    if mean_vals:
        print(f"{'':>5}  {'MEAN':<20} {np.mean(mean_vals):.3f}  (по {len(mean_vals)} присутствующим)")


def find_validation_cases(limit):
    cases = []
    for case, img, gt in config.validation_cases():
        if gt:
            cases.append((img, gt, case))
        if len(cases) >= limit:
            break
    if not cases:
        raise SystemExit(
            "Не найдены валидационные случаи с GT "
            f"(validation\\{config.MODEL_NAME}\\*, требуется nnunet_input + gt)")
    return cases


def run_validation(n_cases, csv_path=None):
    cases = find_validation_cases(n_cases)
    if not cases:
        raise SystemExit(
            f"Не найдены случаи валидации в {config.VALIDATION_IMAGES} "
            f"(с парными GT в {config.VALIDATION_LABELS})"
        )
    print(f"Случаев к валидации: {len(cases)}")

    infer_legacy = [sys.executable, "-m", "radonc_segmentation.infer_ov"]
    infer_aligned = [sys.executable, "-m", "radonc_segmentation.aligned.infer_ov_aligned"]

    per_variant = {"legacy": [], "aligned": []}
    name_by_value = {v: k for k, v in config.LABELS.items()}

    for img_path, gt_path, case in cases:
        print(f"\n### Случай {case}")
        out_legacy = os.path.join(config.LOG_DIR, f"_val_{case}_legacy.nii.gz")
        out_aligned = os.path.join(config.LOG_DIR, f"_val_{case}_aligned.nii.gz")
        os.makedirs(config.LOG_DIR, exist_ok=True)

        for script, out, variant in [
            (infer_legacy, out_legacy, "legacy"),
            (infer_aligned, out_aligned, "aligned"),
        ]:
            tmp = os.path.join(config.LOG_DIR, f"_tmp_{variant}")
            cmd = script + [case, "--input", img_path, "--output", out,
                            "--force", "--tmp-dir", tmp]
            print(f"[{variant}] {' '.join(cmd)}")
            t0 = time.perf_counter()
            r = subprocess.run(cmd, cwd=PROJECT_DIR)
            if r.returncode != 0:
                print(f"[{variant}] ОШИБКА (код {r.returncode}) — пропуск")
                continue
            print(f"[{variant}] готово за {(time.perf_counter() - t0) / 60:.1f} мин")

        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        for variant, out in [("legacy", out_legacy), ("aligned", out_aligned)]:
            if not os.path.isfile(out):
                continue
            pred = sitk.GetArrayFromImage(sitk.ReadImage(out))
            if pred.shape != gt_arr.shape:
                print(f"[{variant}] размер предсказания {pred.shape} != GT {gt_arr.shape}, пропуск")
                continue
            labels = [int(v) for v in np.unique(gt_arr) if int(v) > 0]
            row = {}
            for lbl in labels:
                d = dice(pred == lbl, gt_arr == lbl)
                if d is not None:
                    row[lbl] = d
            per_variant[variant].append((case, row))
            os.remove(out)

    # Сводная таблица
    all_labels = sorted({lbl for v in per_variant.values() for _, row in v for lbl in row})
    print("\n===== СВОДКА (Dice против GT) =====")
    header = f"{'label':>5}  {'орган':<20} {'legacy':>8} {'aligned':>8} {'разница':>8}"
    print(header)
    csv_rows = []
    for lbl in all_labels:
        leg = [row[lbl] for _, row in per_variant["legacy"] if lbl in row]
        alg = [row[lbl] for _, row in per_variant["aligned"] if lbl in row]
        leg_m = np.mean(leg) if leg else None
        alg_m = np.mean(alg) if alg else None
        diff = (alg_m - leg_m) if (leg_m is not None and alg_m is not None) else None
        print(f"{lbl:>5}  {name_by_value.get(lbl, lbl):<20} "
              f"{leg_m if leg_m is not None else float('nan'):>8.3f} "
              f"{alg_m if alg_m is not None else float('nan'):>8.3f} "
              f"{diff if diff is not None else float('nan'):>+8.3f}")
        csv_rows.append([lbl, name_by_value.get(lbl, lbl), leg_m, alg_m, diff])

    for variant, runs in per_variant.items():
        vals = [d for _, row in runs for d in row.values()]
        if vals:
            print(f"MEAN {variant}: {np.mean(vals):.3f} ({len(runs)} случаев, {len(vals)} органов)")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["label", "organ", "dice_legacy", "dice_aligned", "diff"])
            w.writerows(csv_rows)
        print(f"CSV: {csv_path}")


def main():
    p = argparse.ArgumentParser(description="Dice-сравнение сегментаций H&N")
    p.add_argument("file_a", nargs="?", help="Первая сегментация (NIfTI)")
    p.add_argument("file_b", nargs="?", help="Вторая сегментация (NIfTI)")
    p.add_argument("--validation", type=int, default=0, metavar="N",
                   help="Прогнать оба варианта инференса на N валидационных случаях")
    p.add_argument("--csv", default=None, help="Сохранить сводку в CSV")
    args = p.parse_args()

    if args.validation:
        run_validation(args.validation, args.csv)
        return

    if not args.file_a or not args.file_b:
        raise SystemExit("Укажите два NIfTI-файла или --validation N")

    rows = compare_files(args.file_a, args.file_b)
    if rows is not None:
        print_table(f"{os.path.basename(args.file_a)} vs {os.path.basename(args.file_b)}", rows)
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["label", "organ", "dice"])
                w.writerows(rows)
            print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
