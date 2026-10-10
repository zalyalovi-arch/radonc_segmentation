# ============================================================
# benchmark_dice.py — баланс качество/скорость: Dice против GT
# ------------------------------------------------------------
# Прогоняет случаи из валидационного набора через несколько моделей
# (например original vs квантованная), считает Dice по каждому органу
# против GT и время инференса.
#
#   python benchmark_dice.py --models original,int8 --n 2
#   python benchmark_dice.py --models original,w4_exp1 --n 3 --stride 0.625
#   python benchmark_dice.py --models original,int8 --n 1 --folds 0   # быстрый тест
#
# Данные: C:\nnunet\validation\imagesTr\*_0000.nii.gz (КТ) и
#         labelsTr\<случай>.nii.gz (GT, метки 1..31 как в config.LABELS).
# Результат: CSV + таблица в консоль; сегменты сохраняются в out-папке.
# ВНИМАНИЕ: полный прогон = N случаев × N моделей × ~10-15 мин.
# ============================================================
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

from radonc_segmentation import config
from radonc_segmentation import compare_dice as cd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_IMAGES = config.VALIDATION_IMAGES
VALID_LABELS = config.VALIDATION_LABELS


def find_cases(limit):
    # новая per-case структура validation\<модель>\<кейс> (+ legacy imagesTr)
    cases = [(case, img, gt) for case, img, gt in config.validation_cases() if gt]
    if cases:
        return cases[:limit] if limit else cases
    cases = []
    for p in sorted(glob.glob(os.path.join(VALID_IMAGES, "*_0000.nii.gz"))):
        name = os.path.basename(p)[:-7]          # снять .nii.gz
        if name.endswith("_0000"):
            name = name[:-5]
        gt = os.path.join(VALID_LABELS, f"{name}.nii.gz")
        if os.path.isfile(gt):
            cases.append((name, p, gt))
        if len(cases) >= limit:
            break
    return cases


def run_inference(case, img_path, out_path, model_cfg, args, tmp_dir):
    """Прогон инференса одной модели; возвращает (total_s, sec_per_patch, n_patches)."""
    cmd = [sys.executable, "-m", "radonc_segmentation.infer_ov",
           case, "--input", img_path, "--output", out_path,
           "--force", "--tmp-dir", tmp_dir, "--model", model_cfg["name"]]
    if args.folds:
        cmd += ["--folds", args.folds]
    if args.stride:
        cmd += ["--stride", str(args.stride)]
    if args.device:
        cmd += ["--device", args.device]

    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"Инференс {model_cfg['name']} на {case} упал (код {r.returncode})")

    m = re.search(r"Всего: ([\d.]+) с \(([\d.]+) мин\), в среднем ([\d.]+) с/патч", r.stdout)
    m2 = re.search(r"Патчей на фолд: (\d+)", r.stdout)
    total_s = float(m.group(1)) if m else wall
    spp = float(m.group(3)) if m else float("nan")
    n_patches = int(m2.group(1)) if m2 else -1
    return total_s, spp, n_patches


def dice_vs_gt(pred_path, gt_path):
    rows = cd.compare_files(pred_path, gt_path)
    if rows is None:
        raise RuntimeError(f"Геометрия предсказания и GT не совпала: {pred_path}")
    dices = {lbl: d for lbl, _n, d in rows if d is not None}
    return dices


def main():
    p = argparse.ArgumentParser(description="Dice+время: оригинал vs квантованные модели")
    p.add_argument("--models", default=f"original,{config.MODEL}",
                   help=f"Модели через запятую (по умолчанию original,{config.MODEL})")
    p.add_argument("--n", type=int, default=2, help="Сколько первых случаев валидации брать")
    p.add_argument("--folds", default=None, help="Фолды (пусто=все): напр. 0 для быстрого теста")
    p.add_argument("--stride", default=None, help="Переопределить stride на время бенчмарка")
    p.add_argument("--device", default=None, choices=["cpu", "gpu", "gpu.0"])
    p.add_argument("--out-dir", default=None, help="Куда складывать результаты (по умолч. LOG_DIR\\benchmark)")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    model_cfgs = [config.resolve_model(m) for m in models]

    cases = find_cases(args.n)
    if not cases:
        raise SystemExit(f"Не найдены случаи в {VALID_IMAGES} (с GT в {VALID_LABELS})")
    print(f"Модели: {[m['name'] for m in model_cfgs]}")
    print(f"Случаев: {len(cases)}; фолды: {args.folds or 'все у каждой модели'}; "
          f"stride: {args.stride or 'из settings.ini'}")

    out_dir = args.out_dir or os.path.join(config.LOG_DIR, "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # (case, model) -> (dices dict, total_s, spp, n_patches)
    results = {}
    for case, img_path, gt_path in cases:
        print(f"\n===== Случай {case} =====")
        tmp_dir = os.path.join(out_dir, f"_tmp_{case}")
        for mc in model_cfgs:
            out_path = os.path.join(out_dir, f"{case}__{mc['name']}.nii.gz")
            print(f"--- модель {mc['name']} ---")
            total_s, spp, n_patches = run_inference(case, img_path, out_path, mc, args, tmp_dir)
            dices = dice_vs_gt(out_path, gt_path)
            results[(case, mc["name"])] = (dices, total_s, spp, n_patches)
            mean_d = np.mean(list(dices.values()))
            print(f"    Dice(mean)={mean_d:.3f} по {len(dices)} органам | "
                  f"время {total_s:.0f} с ({total_s / 60:.1f} мин) | {spp:.2f} с/патч | "
                  f"патчей/фолд {n_patches}")
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    # ---- сводка ----
    all_labels = sorted({l for (_c, _m), (d, *_r) in results.items() for l in d})
    name_by_value = {v: k for k, v in config.LABELS.items()}

    print("\n================ ИТОГИ ================")
    per_model_stats = {}
    for mc in model_cfgs:
        mn = mc["name"]
        runs = [(d, t, s) for (c, m), (d, t, s, _n) in results.items() if m == mn]
        case_means = [np.mean(list(d.values())) for d, _t, _s in runs]
        per_model_stats[mn] = {
            "dice_mean": float(np.mean(case_means)),
            "time_mean_s": float(np.mean([t for _d, t, _s in runs])),
            "spp_mean": float(np.mean([s for _d, _t, s in runs])),
            "n": len(runs),
        }
    for mn, st in per_model_stats.items():
        print(f"{mn:<20} Dice(mean)={st['dice_mean']:.4f} | "
              f"время/случай={st['time_mean_s'] / 60:.1f} мин | "
              f"{st['spp_mean']:.2f} с/патч | случаев={st['n']}")

    if len(model_cfgs) > 1:
        print(f"\n{'орган':<20}" + "".join(f"{mn:>14}" for mn in per_model_stats))
        organ_rows = []
        for lbl in all_labels:
            vals = []
            for mc in model_cfgs:
                ds = [d[lbl] for (c, m), (d, *_r) in results.items()
                      if m == mc["name"] and lbl in d]
                vals.append(float(np.mean(ds)) if ds else float("nan"))
            organ_rows.append((lbl, vals))
            diff = vals[1] - vals[0] if len(vals) > 1 else float("nan")
            print(f"{name_by_value.get(lbl, lbl):<20}"
                  + "".join(f"{v:>14.4f}" for v in vals)
                  + (f"   (разница {diff:+.4f})" if len(vals) > 1 else ""))

    # ---- CSV ----
    csv_path = os.path.join(out_dir, f"benchmark_{stamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["case", "model", "organ", "dice"])
        for (case, mn), (dices, *_r) in sorted(results.items()):
            for lbl in sorted(dices):
                w.writerow([case, mn, name_by_value.get(lbl, lbl), f"{dices[lbl]:.4f}"])
        w.writerow([])
        w.writerow(["model", "dice_mean", "time_min", "sec_per_patch", "n_cases"])
        for mn, st in per_model_stats.items():
            w.writerow([mn, f"{st['dice_mean']:.4f}", f"{st['time_mean_s'] / 60:.1f}",
                        f"{st['spp_mean']:.2f}", st["n"]])
    print(f"\nCSV: {csv_path}")
    print(f"Сегменты и GT-сравнение: {out_dir}")


if __name__ == "__main__":
    main()
