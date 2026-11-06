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

    infer_cmd = [sys.executable, "-m", "radonc_segmentation.infer_ov"]
    name_by_value = {v: k for k, v in config.LABELS.items()}

    per_case = []  # [(case, {label: dice})]
    for img_path, gt_path, case in cases:
        print(f"\n### Случай {case}")
        out = os.path.join(config.LOG_DIR, f"_val_{case}.nii.gz")
        os.makedirs(config.LOG_DIR, exist_ok=True)

        tmp = os.path.join(config.LOG_DIR, "_tmp_val")
        cmd = infer_cmd + [case, "--input", img_path, "--output", out,
                           "--force", "--tmp-dir", tmp]
        print(f"[infer] {' '.join(cmd)}")
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=PROJECT_DIR)
        if r.returncode != 0:
            print(f"[infer] ОШИБКА (код {r.returncode}) — пропуск")
            continue
        print(f"[infer] готово за {(time.perf_counter() - t0) / 60:.1f} мин")

        if not os.path.isfile(out):
            continue
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        pred = sitk.GetArrayFromImage(sitk.ReadImage(out))
        if pred.shape != gt_arr.shape:
            print(f"[infer] размер предсказания {pred.shape} != GT {gt_arr.shape}, пропуск")
            continue
        labels = [int(v) for v in np.unique(gt_arr) if int(v) > 0]
        row = {}
        for lbl in labels:
            d = dice(pred == lbl, gt_arr == lbl)
            if d is not None:
                row[lbl] = d
        per_case.append((case, row))
        os.remove(out)

    # Сводная таблица
    all_labels = sorted({lbl for _, row in per_case for lbl in row})
    print("\n===== СВОДКА (Dice против GT) =====")
    print(f"{'label':>5}  {'орган':<20} {'dice':>8} {'n':>3}")
    csv_rows = []
    for lbl in all_labels:
        vals = [row[lbl] for _, row in per_case if lbl in row]
        m = np.mean(vals) if vals else None
        print(f"{lbl:>5}  {name_by_value.get(lbl, lbl):<20} "
              f"{m if m is not None else float('nan'):>8.3f} {len(vals):>3}")
        csv_rows.append([lbl, name_by_value.get(lbl, lbl), m])

    vals = [d for _, row in per_case for d in row.values()]
    if vals:
        print(f"MEAN: {np.mean(vals):.4f} ({len(per_case)} случаев, {len(vals)} органов)")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["label", "organ", "dice"])
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
