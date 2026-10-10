# ============================================================
# compare_external.py — наша модель vs внешняя zero-shot модель vs GT
# ------------------------------------------------------------
# Сравнивает по органам (пересечение меток из ts_to_labels.json):
#   Dice(наша сегментация, GT)  vs  Dice(TotalSegmentator, GT)
#
# Ожидаемые файлы на кейс <case>:
#   validation\labelsTr\<case>.nii.gz                  — GT
#   validation\seg_<case>.nii.gz                       — наша сегментация
#   validation\external\totalsegmentator\<case>.nii.gz — TS (из run_ts.py)
#
# Запуск (в nnunet_env):
#   python external_baselines\compare_external.py
#   python external_baselines\compare_external.py --cases val001 val002 --csv ...
# ============================================================
import argparse
import csv
import json
import os

import numpy as np
import SimpleITK as sitk

from radonc_segmentation import config


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return None
    return 2.0 * np.logical_and(a, b).sum() / denom


def load(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Наша модель vs TotalSegmentator vs GT (Dice по органам)")
    ap.add_argument("--cases", nargs="*", default=None, help="кейсы (по умолчанию все с внешним результатом)")
    ap.add_argument("--external-name", default="totalsegmentator",
                    help="имя внешней модели (папка validation\\external\\<name> и подписи в таблице)")
    ap.add_argument("--ours-dir", default=None, help="папка с нашими сегментациями (по умолчанию validation)")
    ap.add_argument("--csv", default=None, help="CSV (по умолчанию logs\\compare_external.csv)")
    args = ap.parse_args()

    val_root = os.path.dirname(config.VALIDATION_IMAGES)
    ext_root = os.path.join(val_root, "external", args.external_name)
    ours_dir = args.ours_dir or val_root
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts_to_labels.json")
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    groups = mapping.get("groups", {})

    if args.cases:
        cases = args.cases
    else:
        cases = sorted(f[:-7] for f in os.listdir(ext_root) if f.endswith(".nii.gz"))
    if not cases:
        raise SystemExit(f"Нет внешних результатов в {ext_root} (сначала run_ts.py)")

    all_rows = []  # (case, label, organ, dice_ours, dice_ts)
    for case in cases:
        gt_path = os.path.join(config.VALIDATION_LABELS, f"{case}.nii.gz")
        ours_path = os.path.join(ours_dir, f"seg_{case}.nii.gz")
        ext_path = os.path.join(ext_root, f"{case}.nii.gz")
        for p in (gt_path, ours_path, ext_path):
            if not os.path.isfile(p):
                raise SystemExit(f"Не найден файл: {p}")
        gt, ours, ext = load(gt_path), load(ours_path), load(ext_path)
        if not (gt.shape == ours.shape == ext.shape):
            raise SystemExit(f"[{case}] размеры не совпадают: GT {gt.shape}, ours {ours.shape}, ext {ext.shape}")

        print(f"\n===== {case} ({args.external_name}) =====")
        print(f"{'lbl':>4} {'орган':<20} {'наша':>8} {args.external_name[:12]:>12} {'Δ(ts-ours)':>10}")
        for lbl_str, spec in mapping["mapping"].items():
            lbl = int(lbl_str)
            d_ours = dice(ours == lbl, gt == lbl)
            d_ts = dice(ext == lbl, gt == lbl)
            if d_ours is None and d_ts is None:
                continue
            d_o = 0.0 if d_ours is None else d_ours
            d_t = 0.0 if d_ts is None else d_ts
            print(f"{lbl:>4} {spec['organ']:<20} "
                  f"{'n/a' if d_ours is None else f'{d_ours:.3f}':>8} "
                  f"{'n/a' if d_ts is None else f'{d_ts:.3f}':>12} {d_t - d_o:>+10.3f}")
            all_rows.append((case, lbl, spec["organ"], d_ours, d_ts))

        # сводные группы (напр. PharynxConst: объединение наших частей vs один класс TS)
        for gname, lbls in groups.items():
            m_ours = np.isin(ours, lbls)
            m_gt = np.isin(gt, lbls)
            m_ext = np.isin(ext, lbls)
            d_ours, d_ts = dice(m_ours, m_gt), dice(m_ext, m_gt)
            print(f"{'':>4} {gname + ' (объед.)':<20} "
                  f"{'n/a' if d_ours is None else f'{d_ours:.3f}':>8} "
                  f"{'n/a' if d_ts is None else f'{d_ts:.3f}':>12} "
                  f"{(d_ts - d_ours) if (d_ts is not None and d_ours is not None) else float('nan'):>+10.3f}")

    # --- средние по пересечению ---
    ours_mean = np.mean([r[3] for r in all_rows if r[3] is not None])
    ts_mean = np.mean([r[4] for r in all_rows if r[4] is not None])
    print(f"\nMEAN по пересечению меток ({len(all_rows)} строк, {len(cases)} кейс(ов)): "
          f"наша = {ours_mean:.4f}, {args.external_name} = {ts_mean:.4f}, "
          f"Δ = {ts_mean - ours_mean:+.4f}")

    csv_path = args.csv or os.path.join(config.LOG_DIR, "compare_external.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["case", "label", "organ", "dice_ours", f"dice_{args.external_name}", "delta"])
        for case, lbl, organ, d_ours, d_ts in all_rows:
            d_o = "" if d_ours is None else f"{d_ours:.4f}"
            d_t = "" if d_ts is None else f"{d_ts:.4f}"
            delta = "" if (d_ours is None or d_ts is None) else f"{d_ts - d_ours:+.4f}"
            w.writerow([case, lbl, organ, d_o, d_t, delta])
        w.writerow(["", "", "MEAN", f"{ours_mean:.4f}", f"{ts_mean:.4f}", f"{ts_mean - ours_mean:+.4f}"])
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
