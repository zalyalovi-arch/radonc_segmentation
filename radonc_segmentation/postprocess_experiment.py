# ============================================================
# postprocess_experiment.py — исследование постпроцессинга масок
# ------------------------------------------------------------
# Сравнивает варианты ПОСТПРОЦЕССИНГА сегментации по Dice против GT,
# по органам. Малые структуры (Cochlea/Lens и т.п.) намеренно НЕ
# удаляются — никаких «очисток мелких компонент», только closing /
# fill holes / сглаживание (референс closing: nnunetv2
# training/data_augmentation/custom_transforms/cascade_transforms.py,
# skimage.morphology.binary_closing).
#
# Запуск:
#   python postprocess_experiment.py --seg <seg.nii.gz> --gt <gt.nii.gz>
#       [--out-csv <logs\postprocess_experiment.csv>] [--save-dir <папка>]
#       [--radius 1 2 3] [--sigma 0.7 1.0]
#
# Варианты:
#   base            — без постпроцессинга
#   close_r<N>      — binary_closing по каждому органу (структурный элемент — шар)
#   fill3d          — binary_fill_holes по каждому органу (3D)
#   fill2d          — заполнение дыр послойно (по Z)
#   close_r<N>_fill — closing + fill3d
#   gauss_s<S>      — гауссово сглаживание бинарной маски, порог 0.5
#
# Пересечения органов после обработки решаются приоритетом: маска органа
# с БОЛЬШИМ номером метки вырезается из меньших (Lens/Eye, Cochlea/Brain
# и т.п.) — так же, как в GT.
# ============================================================
import argparse
import csv
import os
import time

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
from skimage.morphology import ball

import SimpleITK as sitk

from radonc_segmentation import config
from radonc_segmentation import env_check

env_check.ensure_ready()


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return None
    return 2.0 * np.logical_and(a, b).sum() / denom


def apply_per_class(seg, fn):
    """Применяет fn(mask_bool)->mask_bool к каждому классу независимо."""
    out = np.zeros_like(seg, dtype=np.uint8)
    for lbl in np.unique(seg):
        if lbl == 0:
            continue
        out[fn(seg == lbl)] = lbl
    return out


def resolve_overlaps(seg):
    """Больший номер метки 'выигрывает': воксели старших меток вырезаются из
    младших (Lens из Eye, Cochlea из Brain/TMJ и т.п. — как в GT)."""
    out = seg.copy()
    labels = [int(lab) for lab in np.unique(seg) if lab > 0]
    for i, hi in enumerate(labels):
        mask_hi = out == hi
        if not mask_hi.any():
            continue
        for lo in labels[:i]:
            m = (out == lo) & mask_hi
            if m.any():
                out[m] = hi
    return out


def count_overlaps(seg):
    """Число вокселей, заявленных >1 классом (до resolve_overlaps)."""
    labels = [int(lab) for lab in np.unique(seg) if lab > 0]
    occ = np.zeros(seg.shape, dtype=np.uint8)
    for lbl in labels:
        occ += (seg == lbl).astype(np.uint8)
    return int((occ > 1).sum())


def closing_variant(seg, radius):
    st = ball(radius)

    def f(m):
        return binary_closing(m, structure=st)

    return apply_per_class(seg, f)


def gauss_variant(seg, sigma):
    def f(m):
        sm = gaussian_filter(m.astype(np.float32), sigma=sigma)
        return sm >= 0.5

    return apply_per_class(seg, f)


VARIANTS = {}  # name -> fn(seg)->seg


def build_variants(radii, sigmas):
    VARIANTS["base"] = lambda s: s
    for r in radii:
        VARIANTS[f"close_r{r}"] = (lambda rr: (lambda s: closing_variant(s, rr)))(r)
        VARIANTS[f"close_r{r}_fill"] = (
            lambda rr: (lambda s: apply_per_class(closing_variant(s, rr), binary_fill_holes))
        )(r)
    VARIANTS["fill3d"] = lambda s: apply_per_class(s, binary_fill_holes)

    def fill2d(m):
        out = m.copy()
        for z in range(m.shape[0]):
            out[z] = binary_fill_holes(m[z])
        return out

    VARIANTS["fill2d"] = lambda s: apply_per_class(s, fill2d)
    for sgm in sigmas:
        VARIANTS[f"gauss_s{sgm}"] = (lambda ss: (lambda s: gauss_variant(s, ss)))(sgm)


def main():
    p = argparse.ArgumentParser(description="Эксперимент: постпроцессинг vs GT (Dice по органам)")
    p.add_argument("--seg", required=True, help="сегментация (NIfTI), напр. validation\\seg_val001.nii.gz")
    p.add_argument("--gt", required=True, help="GT-маска (NIfTI), напр. validation\\labelsTr\\val001.nii.gz")
    p.add_argument("--radius", type=int, nargs="+", default=[1, 2, 3], help="радиусы closing")
    p.add_argument("--sigma", type=float, nargs="+", default=[0.7, 1.0], help="сигмы сглаживания")
    p.add_argument("--out-csv", default=None, help="куда сохранить CSV (по умолч. logs\\postprocess_experiment.csv)")
    p.add_argument("--save-dir", default=None, help="сохранять обработанные маски NIfTI в эту папку")
    args = p.parse_args()

    seg_img = sitk.ReadImage(args.seg)
    seg = sitk.GetArrayFromImage(seg_img).astype(np.uint8)  # (z, y, x)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(args.gt)).astype(np.uint8)
    if seg.shape != gt.shape:
        raise SystemExit(f"Размеры не совпадают: seg {seg.shape} vs GT {gt.shape}")

    labels = sorted(set(int(v) for v in np.unique(gt) if v > 0) |
                    set(int(v) for v in np.unique(seg) if v > 0))
    names = {v: config.LABELS.get(v, f"label_{v}") for v in labels}
    print(f"Сег: {args.seg} {seg.shape}; GT: {args.gt}")
    print(f"Органов в сравнении: {len(labels)}; вариантов: заранее неизвестно (считаем)\n")

    build_variants(args.radius, args.sigma)

    rows = []          # (variant, label, name, dice)
    mean_rows = []     # (variant, mean_dice_over_present)
    t_all = time.perf_counter()
    for vname, fn in VARIANTS.items():
        t0 = time.perf_counter()
        raw = fn(seg)
        ov = count_overlaps(raw) if vname != "base" else 0
        processed = resolve_overlaps(raw)
        dt = time.perf_counter() - t0
        dices = []
        for lbl in labels:
            d = dice(processed == lbl, gt == lbl)
            rows.append((vname, lbl, names[lbl], d))
            if d is not None:
                dices.append(d)
        mean_d = float(np.mean(dices)) if dices else float("nan")
        mean_rows.append((vname, mean_d, len(dices), dt, ov))
        print(f"{vname:<18} MEAN Dice = {mean_d:.4f}  ({len(dices)} органов, {dt:.1f} с, пересечений {ov})")
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            out_img = sitk.GetImageFromArray(processed)
            out_img.CopyInformation(seg_img)
            sitk.WriteImage(out_img, os.path.join(args.save_dir, f"seg_{vname}.nii.gz"))
    print(f"\nВсего вариантов просчитано за {time.perf_counter() - t_all:.1f} с")

    # --- детальная таблица: строки = органы, колонки = варианты ---
    vnames = list(VARIANTS.keys())
    print(f"\n{'lbl':>4} {'орган':<20}" + "".join(f"{v:>12}" for v in vnames))
    for lbl in labels:
        cells = []
        for vname in vnames:
            d = next(r[3] for r in rows if r[0] == vname and r[1] == lbl)
            cells.append("n/a" if d is None else f"{d:.3f}")
        print(f"{lbl:>4} {names[lbl]:<20}" + "".join(f"{c:>12}" for c in cells))
    print(f"{'':>4} {'MEAN':<20}" + "".join(f"{m[1]:>12.4f}" for m in mean_rows))

    # --- CSV ---
    csv_path = args.out_csv or os.path.join(config.LOG_DIR, "postprocess_experiment.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["label", "organ"] + vnames)
        for lbl in labels:
            cells = []
            for vname in vnames:
                d = next(r[3] for r in rows if r[0] == vname and r[1] == lbl)
                cells.append("" if d is None else f"{d:.4f}")
            w.writerow([lbl, names[lbl]] + cells)
        w.writerow(["", "MEAN"] + [f"{m[1]:.4f}" for m in mean_rows])
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
