# ============================================================
# aligned/infer_ov_aligned.py — вариант инференса с преобработкой,
# выровненной по обучению nnUNet (для сравнения с основным вариантом)
# ------------------------------------------------------------
# Отличия от infer_ov.py:
#   1. Ресемплинг тома в training spacing (2.0 x 1.171875 x 1.171875 мм) —
#      у ВСЕХ пациентов обучающей выборки spacing ровно такой, а основной
#      вариант скармливает модели том в исходном spacing пациента.
#   2. CTNormalization с ГЛОБАЛЬНЫМИ статистиками датасета
#      (clip -990..1470 -> (x-50.02)/215.94 из dataset_fingerprint.json),
#      а не перцентили/z-score по каждому объёму.
#   3. Пэддинг нулями (как в nnUNet), а не -1000.
#
# Запуск:
#   python aligned/infer_ov_aligned.py P110
# Результат: seg_nnunet\PXXX_aligned.nii.gz (основной результат не трогает).
# Сравнение: python compare_dice.py <patients>\PXXX\seg_nnunet\PXXX.nii.gz
#                                   <patients>\PXXX\seg_nnunet\PXXX_aligned.nii.gz
# ============================================================
import argparse
import csv
import json
import os
import time

import numpy as np
import SimpleITK as sitk

from radonc_segmentation import config
from radonc_segmentation import env_check
from radonc_segmentation import ov_infer_core as core_mod

# предупреждаем о проблемах окружения ДО трейсбеков (см. env_check.py)
env_check.ensure_ready()


def load_norm_stats():
    """Глобальные статистики из fingerprint датасета; при недоступности — константы из config."""
    if not config.DATASET_FINGERPRINT:
        return config.CT_CLIP_LO, config.CT_CLIP_HI, config.CT_MEAN, config.CT_STD
    try:
        with open(config.DATASET_FINGERPRINT, "r", encoding="utf-8") as f:
            fp = json.load(f)
        props = fp["foreground_intensity_properties_per_channel"]["0"]
        return (
            float(props["percentile_00_5"]),
            float(props["percentile_99_5"]),
            float(props["mean"]),
            float(props["std"]),
        )
    except (OSError, KeyError, ValueError):
        print(f"[aligned] DATASET_FINGERPRINT задан, но не читается: "
              f"{config.DATASET_FINGERPRINT} — использую константы из config")
        return config.CT_CLIP_LO, config.CT_CLIP_HI, config.CT_MEAN, config.CT_STD


def resample_to_target(img, tgt_spacing_xyz):
    """Ресемплинг тома в целевой spacing (линейная интерполяция, вне поля -1000 HU)."""
    orig_size = np.array(img.GetSize(), dtype=float)
    orig_spacing = np.array(img.GetSpacing(), dtype=float)
    new_size_xyz = [
        max(1, int(round(orig_size[i] * orig_spacing[i] / tgt_spacing_xyz[i])))
        for i in range(3)
    ]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(tgt_spacing_xyz)
    rs.SetSize([int(v) for v in new_size_xyz])
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1000.0)
    return rs.Execute(img), new_size_xyz


def parse_args():
    p = argparse.ArgumentParser(description="OpenVINO инференс (aligned-преобработка)")
    p.add_argument("patient", help="ID пациента (P110 или 110)")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None, help="По умолчанию seg_nnunet\\PXXX_aligned.nii.gz")
    p.add_argument("--stride", type=float, default=config.STEP_SIZE_RATIO)
    p.add_argument("--no-crop", action="store_true", help="Отключить кроп воздуха")
    p.add_argument("--no-async", action="store_true")
    p.add_argument("--folds", default=",".join(str(i) for i in config.DEFAULT_FOLDS))
    p.add_argument("--n-async", type=int, default=config.N_ASYNC_REQUESTS)
    p.add_argument("--hint", default=config.OV_PERFORMANCE_HINT, choices=["LATENCY", "THROUGHPUT", "none"])
    p.add_argument("--device", default=None, choices=["cpu", "gpu", "gpu.0"],
                   help="Переопределить устройство из settings.ini на один запуск")
    p.add_argument("--model", default=None,
                   help="Имя модели из MODELS_ROOT (по умолчанию из settings.ini)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--tmp-dir", default=None)
    return p.parse_args()


def normalize_patient_id(s):
    s = str(s).strip()
    if s.isdigit():
        return f"P{int(s):03d}"
    if len(s) == 4 and s.startswith("P") and s[1:].isdigit():
        return s.upper()
    raise SystemExit(f"Некорректный ID пациента: {s!r}")


def append_log(row):
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        path = os.path.join(config.LOG_DIR, "infer_log.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            if new:
                w.writerow(["timestamp", "patient", "variant", "stride", "crop", "async",
                            "folds", "patches_per_fold", "sec_per_patch", "total_min", "output"])
            w.writerow(row)
    except OSError as e:
        print(f"   (не удалось записать лог: {e})")


def main():
    args = parse_args()
    # При явных --input и --output допускаем произвольную метку (напр. случай валидации)
    if args.input and args.output:
        patient_id = str(args.patient)
    else:
        patient_id = normalize_patient_id(args.patient)
    base_dir = config.patient_dir(patient_id) if patient_id.startswith("P") else os.path.dirname(args.input)

    input_path = args.input or os.path.join(base_dir, "nnunet_input", f"{patient_id}_0000.nii.gz")
    output_path = args.output or os.path.join(base_dir, "seg_nnunet", f"{patient_id}_aligned.nii.gz")
    if not os.path.isfile(input_path):
        raise SystemExit(f"Входной NIfTI не найден: {input_path}")
    if os.path.isfile(output_path) and not args.force and not args.output:
        raise SystemExit(f"Результат уже существует: {output_path} (--force для перезаписи)")

    model_cfg = config.resolve_model(args.model)
    folds = sorted(set(int(x) for x in str(args.folds).split(",") if x != "")
                   & set(model_cfg["folds"]))
    if not folds:
        raise SystemExit(f"У модели {model_cfg['name']!r} нет выбранных фолдов "
                         f"({args.folds}); доступны: {model_cfg['folds']}")
    use_crop = not args.no_crop and config.CROP_AIR
    use_async = not args.no_async and args.n_async > 1
    hint = None if args.hint == "none" else args.hint
    tgt_spacing_xyz = tuple(reversed(config.TARGET_SPACING_ZYX))

    print(f"Пациент: {patient_id} (aligned-вариант)")
    print(f"Вход:  {input_path}")
    print(f"Выход: {output_path}")
    print(f"Модель: {model_cfg['name']} ({'ансамбль' if model_cfg['ensemble'] else 'одиночная'})")
    print(f"Режим: устройство={str(args.device or config.DEVICE).upper()}, фолды={folds}, stride={args.stride}, "
          f"кроп={use_crop}, async={'да' if use_async else 'нет'}, hint={hint}, "
          f"накопитель={config.ACCUMULATOR}, target spacing (x,y,z)={tgt_spacing_xyz}")

    t_total = time.perf_counter()

    # 1. Чтение и ресемплинг в training spacing
    img = sitk.ReadImage(input_path)
    print(f"Исходный том (x,y,z): {img.GetSize()}, spacing: "
          f"{tuple(round(s, 3) for s in img.GetSpacing())} мм")
    img_rs, new_size = resample_to_target(img, tgt_spacing_xyz)
    print(f"После ресемплинга (x,y,z): {tuple(new_size)}")

    arr_zyx = sitk.GetArrayFromImage(img_rs).astype(np.float32)  # (Z, Y, X), HU
    full_shape_zyx = arr_zyx.shape

    # 2. Кроп тела по HU (глобальные статистики от кропа не зависят)
    bbox = None
    if use_crop:
        bbox = core_mod.body_bbox_zyx(
            arr_zyx, threshold_hu=config.AIR_THRESHOLD_HU, margin=config.CROP_MARGIN_VOXELS
        )
        if bbox is not None:
            z0, z1, y0, y1, x0, x1 = bbox
            print(f"Кроп воздуха: Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}]")

    # 3. CTNormalization (глобальные статистики датасета — как при обучении)
    clip_lo, clip_hi, ct_mean, ct_std = load_norm_stats()
    print(f"CTNormalization: clip [{clip_lo}, {clip_hi}], mean={ct_mean:.2f}, std={ct_std:.2f}")
    data = np.clip(arr_zyx, clip_lo, clip_hi)
    data = ((data - ct_mean) / ct_std).astype(np.float32)
    del arr_zyx

    if bbox is not None:
        data = core_mod.crop_bbox(data, bbox)

    # 4. Инференс (движок общий с основным вариантом)
    import openvino as ov
    core = ov.Core()
    device = str(args.device or config.DEVICE).upper()
    core_mod.ensure_device(core, device)
    compile_config = core_mod.make_compile_config(
        config.NUM_THREADS, model_cfg["cache_dir"], hint, device
    )
    tmp_dir = args.tmp_dir or os.path.join(base_dir, "~tmp_infer_aligned")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Скользящее окно:")
    seg, stats = core_mod.run_sliding_window(
        core,
        data,
        patch_size=config.DEFAULT_PATCH_SIZE,
        stride_ratio=args.stride,
        folds=folds,
        model_base_path=model_cfg["base_path"],
        model_filename=model_cfg["filename"],
        device=device,
        compile_config=compile_config,
        tmp_dir=tmp_dir,
        pad_value=0.0,  # nnUNet пэддит нормализованный том нулями
        n_async=args.n_async if use_async else 1,
        accumulator=config.ACCUMULATOR,
    )

    # 5. Возврат кропа и ресемплинг сегментации в исходную геометрию пациента
    if bbox is not None:
        seg = core_mod.uncrop_bbox(seg, original_shape=full_shape_zyx, bbox=bbox)
    seg_sitk = sitk.GetImageFromArray(seg.astype(np.uint8))
    seg_sitk.CopyInformation(img_rs)

    rs_back = sitk.ResampleImageFilter()
    rs_back.SetReferenceImage(img)
    rs_back.SetInterpolator(sitk.sitkNearestNeighbor)
    rs_back.SetDefaultPixelValue(0)
    seg_orig = rs_back.Execute(seg_sitk)

    sitk.WriteImage(seg_orig, output_path)
    total_s = time.perf_counter() - t_total

    print(f"Готово: {output_path}")
    print(f"Всего: {total_s:.1f} с ({total_s / 60:.1f} мин), "
          f"в среднем {stats['sec_per_patch']:.2f} с/патч")

    append_log([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        patient_id,
        f"aligned/{model_cfg['name']}",
        args.stride,
        "on" if use_crop else "off",
        args.n_async if use_async else 1,
        ",".join(map(str, folds)),
        stats["n_patches_per_fold"],
        f"{stats['sec_per_patch']:.2f}",
        f"{total_s / 60:.1f}",
        output_path,
    ])


if __name__ == "__main__":
    main()
