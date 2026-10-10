# ============================================================
# infer_ov.py — инференс H&N модели (OpenVINO INT8, 5 фолдов)
# ------------------------------------------------------------
# Преемник slwinensembz.py: та же преобработка (перцентили по объёму,
# [0,1], z-score), но:
#   - пациент задаётся аргументом, ничего не правится в коде
#   - кроп воздуха по контуру тела (меньше патчей)
#   - async-конвейер OpenVINO (перекрытие преп/пост-обработки)
#   - настраиваемый шаг скользящего окна
#   - тайминги + CSV-лог
#
# Запуск:
#   python infer_ov.py P110                      # настройки по умолчанию
#   python infer_ov.py P110 --stride 0.625       # реже сетка патчей (быстрее)
#   python infer_ov.py P110 --no-crop --no-async # поведение как у slwinensembz.py
#   python infer_ov.py P110 --folds 0            # быстрый смоук-тест (~1/5 времени)
# ============================================================
import argparse
import csv
import os
import time

import numpy as np
import nibabel as nib

from radonc_segmentation import config
from radonc_segmentation import env_check
from radonc_segmentation import ov_infer_core as core_mod

# предупреждаем о проблемах окружения ДО трейсбеков (см. env_check.py)
env_check.ensure_ready()


def ct_normalize(data, percentile_lower=0.5, percentile_upper=99.5):
    """Legacy-преобработка slwinensembz.py: перцентили по объёму -> [0,1] -> z-score."""
    lower_bound = np.percentile(data, percentile_lower)
    upper_bound = np.percentile(data, percentile_upper)
    data_clipped = np.clip(data, lower_bound, upper_bound)

    if upper_bound > lower_bound:
        data_normalized = (data_clipped - lower_bound) / (upper_bound - lower_bound)
    else:
        data_normalized = data_clipped - lower_bound

    mean = np.mean(data_normalized)
    std = np.std(data_normalized)
    if std > 0:
        return (data_normalized - mean) / std
    return data_normalized - mean


def parse_args():
    p = argparse.ArgumentParser(description="OpenVINO инференс H&N сегментации")
    p.add_argument("patient", help="ID пациента (P110 или 110) — либо путь через --input")
    p.add_argument("--input", default=None, help="Явный путь к NIfTI (по умолчанию <patients>\\PXXX\\nnunet_input\\PXXX_0000.nii.gz)")
    p.add_argument("--output", default=None, help="Явный путь к результату (по умолчанию seg_nnunet\\PXXX.nii.gz)")
    p.add_argument("--stride", type=float, default=config.STEP_SIZE_RATIO,
                   help=f"Шаг скользящего окна (по умолч. {config.STEP_SIZE_RATIO})")
    p.add_argument("--no-crop", action="store_true", help="Отключить кроп воздуха")
    p.add_argument("--no-async", action="store_true", help="Синхронный инференс (как slwinensembz.py)")
    p.add_argument("--folds", default=",".join(str(i) for i in config.DEFAULT_FOLDS),
                   help="Фолды через запятую, напр. 0,1,2,3,4 (по умолчанию из settings.ini)")
    p.add_argument("--n-async", type=int, default=config.N_ASYNC_REQUESTS,
                   help=f"Число параллельных infer-запросов (по умолч. {config.N_ASYNC_REQUESTS})")
    p.add_argument("--hint", default=config.OV_PERFORMANCE_HINT, choices=["LATENCY", "THROUGHPUT", "none"],
                   help="OpenVINO PERFORMANCE_HINT")
    p.add_argument("--device", default=None, choices=["cpu", "gpu", "gpu.0"],
                   help="Переопределить устройство из settings.ini на один запуск (cpu/gpu)")
    p.add_argument("--model", default=None,
                   help="Имя модели из MODELS_ROOT (по умолчанию из settings.ini)")
    p.add_argument("--batch", type=int, default=1,
                   help="Патчей на один инференс (эксперимент с THROUGHPUT; "
                        "1 = по одному, как всегда)")
    p.add_argument("--num-threads", type=int, default=None,
                   help="Потоков CPU-плагина OpenVINO на этот запуск "
                        "(по умолчанию из settings.ini)")
    p.add_argument("--force", action="store_true", help="Перезаписать существующий результат")
    p.add_argument("--tmp-dir", default=None, help="Папка для временных memmap-файлов")
    return p.parse_args()


def normalize_patient_id(s):
    s = str(s).strip()
    if s.isdigit():
        return f"P{int(s):03d}"
    if len(s) == 4 and s.startswith("P") and s[1:].isdigit():
        return s.upper()
    raise SystemExit(f"Некорректный ID пациента: {s!r} (ожидается P001 / 001 / 1)")


def append_log(row):
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        path = os.path.join(config.LOG_DIR, "infer_log.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            if new:
                w.writerow(["timestamp", "patient", "variant", "stride", "crop", "async",
                            "folds", "patches_per_fold", "sec_per_patch", "total_min", "output",
                            "batch", "num_threads", "hint"])
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
    base_dir = config.patient_dir(patient_id) if patient_id.startswith("P") else os.path.dirname(args.input or ".")

    input_path = args.input or os.path.join(base_dir, "nnunet_input", f"{patient_id}_0000.nii.gz")
    output_path = args.output or os.path.join(base_dir, "seg_nnunet", f"{patient_id}.nii.gz")
    if not os.path.isfile(input_path):
        raise SystemExit(f"Входной NIfTI не найден: {input_path}\n"
                         f"Сначала выполните: python dicom_to_nifti.py {patient_id}")
    if os.path.isfile(output_path) and not args.force and not args.output:
        raise SystemExit(f"Результат уже существует: {output_path}\n"
                         f"Используйте --force для перезаписи или --output <файл>.")

    model_cfg = config.resolve_model(args.model)
    num_threads = args.num_threads if args.num_threads is not None else config.NUM_THREADS
    folds = sorted(set(int(x) for x in str(args.folds).split(",") if x != "")
                   & set(model_cfg["folds"]))
    if not folds:
        raise SystemExit(f"У модели {model_cfg['name']!r} нет выбранных фолдов "
                         f"({args.folds}); доступны: {model_cfg['folds']}")
    use_crop = not args.no_crop and config.CROP_AIR
    use_async = not args.no_async and args.n_async > 1
    hint = None if args.hint == "none" else args.hint

    print(f"Пациент: {patient_id}")
    print(f"Вход:  {input_path}")
    print(f"Выход: {output_path}")
    print(f"Модель: {model_cfg['name']} ({'ансамбль' if model_cfg['ensemble'] else 'одиночная'}, "
          f"{model_cfg['filename']})")
    print(f"Режим: устройство={str(args.device or config.DEVICE).upper()}, фолды={folds}, stride={args.stride}, "
          f"кроп={use_crop}, async={'да' if use_async else 'нет'} "
          f"(n={args.n_async if use_async else 1}), hint={hint}, накопитель={config.ACCUMULATOR}, "
          f"батч={args.batch}")

    t_total = time.perf_counter()

    # 1. Загрузка (nibabel отдаёт (X, Y, Z))
    nii = nib.load(input_path)
    raw_xyz = nii.get_fdata(dtype=np.float32)
    affine = nii.affine
    print(f"Загружен объём (X,Y,Z): {raw_xyz.shape}")

    # 2. Кадр модели (Z, Y, X)
    raw_zyx = np.transpose(raw_xyz, (2, 1, 0))
    full_shape_zyx = raw_zyx.shape
    del raw_xyz

    # 3. Bounding box тела по HU (по сырому объёму, ДО нормализации)
    bbox = None
    if use_crop:
        bbox = core_mod.body_bbox_zyx(
            raw_zyx, threshold_hu=config.AIR_THRESHOLD_HU, margin=config.CROP_MARGIN_VOXELS
        )
        if bbox is not None:
            z0, z1, y0, y1, x0, x1 = bbox
            print(f"Кроп воздуха: Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}] "
                  f"-> {z1 - z0}x{y1 - y0}x{x1 - x0} (было {raw_zyx.shape[0]}x{raw_zyx.shape[1]}x{raw_zyx.shape[2]})")
        else:
            print("Кроп воздуха: тело не найдено, работаем по всему объёму")

    # 4. Нормализация по ПОЛНОМУ объёму (элементарная операция — кроп на статистику
    #    не влияет; результат идентичен slwinensembz.py)
    print("Нормализация (перцентили 0.5/99.5 -> [0,1] -> z-score)...")
    data = ct_normalize(raw_zyx).astype(np.float32)
    del raw_zyx

    # 5. Кроп уже нормализованного объёма
    if bbox is not None:
        data = core_mod.crop_bbox(data, bbox)

    # 6. Инференс
    import openvino as ov
    core = ov.Core()
    device = str(args.device or config.DEVICE).upper()
    core_mod.ensure_device(core, device)
    compile_config = core_mod.make_compile_config(
        num_threads, model_cfg["cache_dir"], hint, device
    )
    tmp_dir = args.tmp_dir or os.path.join(base_dir, "~tmp_infer")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

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
        pad_value=-1000.0,  # как в slwinensembz.py (legacy-поведение)
        n_async=args.n_async if use_async else 1,
        accumulator=config.ACCUMULATOR,
        batch_size=args.batch,
    )

    # 7. Возврат кропа и кадра (X, Y, Z)
    if bbox is not None:
        seg = core_mod.uncrop_bbox(seg, original_shape=full_shape_zyx, bbox=bbox)
    seg_xyz = np.transpose(seg, (2, 1, 0))

    # 8. Сохранение
    seg_nii = nib.Nifti1Image(seg_xyz, affine)
    nib.save(seg_nii, output_path)
    total_s = time.perf_counter() - t_total

    print(f"Готово: {output_path}")
    print(f"Всего: {total_s:.1f} с ({total_s / 60:.1f} мин), "
          f"в среднем {stats['sec_per_patch']:.2f} с/патч, "
          f"патчей на фолд: {stats['n_patches_per_fold']}")

    append_log([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        patient_id,
        f"legacy/{model_cfg['name']}",
        args.stride,
        "on" if use_crop else "off",
        args.n_async if use_async else 1,
        ",".join(map(str, folds)),
        stats["n_patches_per_fold"],
        f"{stats['sec_per_patch']:.2f}",
        f"{total_s / 60:.1f}",
        output_path,
        args.batch,
        num_threads,
        str(hint),
    ])


if __name__ == "__main__":
    main()
