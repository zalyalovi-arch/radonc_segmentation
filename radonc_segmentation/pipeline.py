# ============================================================
# pipeline.py — весь пайплайн одной командой
# ------------------------------------------------------------
# import DICOM -> NIfTI -> инференс (OpenVINO) -> RTSTRUCT -> экспорт
#
# Запуск (из корня проекта, после pip install -e . или без установки):
#   python -m radonc_segmentation.pipeline "D:\экспорт\пациент\CT"   # новый пациент (ID выдаётся сам)
#   python -m radonc_segmentation.pipeline P007                      # пересчёт существующего
#   python -m radonc_segmentation.pipeline P007 --variant aligned    # инференс с выровненной преобработкой
#   python -m radonc_segmentation.pipeline P007 --stride 0.625       # реже сетка патчей (быстрее)
#   python -m radonc_segmentation.pipeline P007 --no-export          # без копирования RTSTRUCT обратно
# (после установки пакета доступен и короткий алиас: radonc-pipeline ...)
#
# Исходная папка DICOM запоминается в patients\PXXX\patient.json при импорте,
# поэтому экспорт обратно не требует перепечатывать путь.
# ============================================================
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

from radonc_segmentation import config
from radonc_segmentation import env_check

# корень проекта (папка с settings.ini, patients\, models\) — cwd для
# дочерних шагов и база относительных путей
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_row(row):
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        path = os.path.join(config.LOG_DIR, "pipeline_log.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            if new:
                w.writerow(["timestamp", "patient", "variant", "convert_s", "infer_s",
                            "rtstruct_s", "export_s", "total_min", "status"])
            w.writerow(row)
    except OSError as e:
        print(f"(не удалось записать лог: {e})")


def normalize_patient_id(s):
    s = str(s).strip()
    if s.isdigit():
        return f"P{int(s):03d}"
    if re.fullmatch(r"[Pp]\d{3,}", s):
        return s.upper()
    return None


def run_step(title, cmd):
    print(f"\n--- {title} ---")
    print("> " + " ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=PROJECT_DIR)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"Шаг '{title}' завершился с кодом {r.returncode}")
    print(f"--- {title}: готово за {dt:.1f} с ---")
    return dt


def main():
    # проверяем зависимости ВСЕХ шагов заранее — с понятным сообщением
    # вместо трейсбеков в середине пайплайна (см. env_check.py)
    env_check.ensure_ready()
    p = argparse.ArgumentParser(description="Полный пайплайн оконтуривания H&N")
    p.add_argument("target", help="ID пациента (P007 / 7) ИЛИ путь к папке с DICOM")
    p.add_argument("--variant", default=config.DEFAULT_VARIANT, choices=["legacy", "aligned"],
                   help=f"legacy — как в продакшне; aligned — выровненная преобработка (сравнение). "
                        f"По умолчанию из settings.ini: {config.DEFAULT_VARIANT}")
    p.add_argument("--stride", type=float, default=config.STEP_SIZE_RATIO)
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("--no-async", action="store_true")
    p.add_argument("--no-export", action="store_true", help="Не копировать RTSTRUCT в исходную папку")
    p.add_argument("--force-convert", action="store_true", help="Пересоздать NIfTI, даже если он есть")
    p.add_argument("--folds", default=None, help="Фолды через запятую (напр. 0 для смоук-теста)")
    p.add_argument("--device", default=None, choices=["cpu", "gpu", "gpu.0"],
                   help="Переопределить устройство из settings.ini на один запуск")
    p.add_argument("--model", default=None,
                   help="Имя модели из MODELS_ROOT (по умолчанию из settings.ini)")
    p.add_argument("--no-rtstruct", action="store_true", help="Остановиться после инференса")
    p.add_argument("--postprocess-template", default=None, metavar="FILE",
                   help="JSON-шаблон per-структурного постпроцессинга "
                        "(scripts\\postprocess_templates\\*.json) — применяется к сегментации после инференса")
    args = p.parse_args()

    config.ensure_dirs()
    timings = {}
    t_total = time.perf_counter()

    # 1. Определение пациента
    if os.path.isdir(args.target):
        print(f"Цель — папка с DICOM: {args.target}")
        from radonc_segmentation import import_dicom
        patient_id, base_dir = import_dicom.import_dicom(args.target)
    else:
        patient_id = normalize_patient_id(args.target)
        if patient_id is None:
            raise SystemExit(f"Не понял аргумент {args.target!r}: это не ID (P007/7) и не папка")
        base_dir = config.patient_dir(patient_id)
        if not os.path.isdir(base_dir):
            raise SystemExit(f"Папка пациента не найдена: {base_dir}")
    print(f"Пациент: {patient_id}  ({base_dir})")

    infer_flags = []
    if args.stride != config.STEP_SIZE_RATIO:
        infer_flags += ["--stride", str(args.stride)]
    if args.no_crop:
        infer_flags.append("--no-crop")
    if args.no_async:
        infer_flags.append("--no-async")
    if args.folds:
        infer_flags += ["--folds", args.folds]
    if args.device:
        infer_flags += ["--device", args.device]
    if args.model:
        infer_flags += ["--model", args.model]

    try:
        # 2. DICOM -> NIfTI
        nifti_path = os.path.join(base_dir, "nnunet_input", f"{patient_id}_0000.nii.gz")
        if os.path.isfile(nifti_path) and not args.force_convert:
            print(f"\nNIfTI уже существует, конвертация пропущена: {nifti_path}"
                  f" (--force-convert чтобы пересоздать)")
            timings["convert"] = 0.0
        else:
            timings["convert"] = run_step(
                "Конвертация DICOM -> NIfTI",
                [sys.executable, "-m", "radonc_segmentation.dicom_to_nifti", patient_id],
            )

        # 3. Инференс
        if args.variant == "aligned":
            infer_cmd = [sys.executable, "-m", "radonc_segmentation.aligned.infer_ov_aligned",
                         patient_id] + infer_flags
            seg_name = f"{patient_id}_aligned.nii.gz"
        else:
            infer_cmd = [sys.executable, "-m", "radonc_segmentation.infer_ov",
                         patient_id, "--force"] + infer_flags
            seg_name = f"{patient_id}.nii.gz"
        timings["infer"] = run_step(f"Инференс ({args.variant})", infer_cmd)

        # 3.5. Per-структурный постпроцессинг по шаблону (до RTSTRUCT)
        if args.postprocess_template:
            if not os.path.isfile(args.postprocess_template):
                raise RuntimeError(f"Шаблон постпроцессинга не найден: {args.postprocess_template}")
            import SimpleITK as sitk
            import numpy as np
            from radonc_segmentation import postprocess_core as ppcore
            seg_path = os.path.join(base_dir, "seg_nnunet", seg_name)
            print(f"\n--- Постпроцессинг по шаблону: {args.postprocess_template} ---")
            nii = sitk.ReadImage(seg_path)
            arr = sitk.GetArrayFromImage(nii).astype(np.uint8)
            tmpl = ppcore.load_template(args.postprocess_template)
            arr = ppcore.apply_template(arr, tmpl)
            out = sitk.GetImageFromArray(arr)
            out.CopyInformation(nii)
            sitk.WriteImage(out, seg_path)
            n_organs = len(tmpl.get("organs", {}))
            print(f"--- обработано органов: {n_organs} ---")

        # 4. RTSTRUCT
        timings["rtstruct"] = 0.0
        if not args.no_rtstruct:
            rt_cmd = [sys.executable, "-m", "radonc_segmentation.make_rtstruct_hn",
                      patient_id, seg_name]
            timings["rtstruct"] = run_step("Создание RTSTRUCT", rt_cmd)
        else:
            print("\nRTSTRUCT пропущен (--no-rtstruct)")

        # 5. Экспорт обратно в исходную папку
        timings["export"] = 0.0
        if not args.no_export and not args.no_rtstruct:
            pj_path = os.path.join(base_dir, "patient.json")
            src = None
            if os.path.isfile(pj_path):
                try:
                    with open(pj_path, "r", encoding="utf-8") as f:
                        src = json.load(f).get("source_path")
                except (OSError, ValueError):
                    src = None
            if src and os.path.isdir(src):
                timings["export"] = run_step(
                    "Экспорт RTSTRUCT в исходную папку",
                    [sys.executable, "-m", "radonc_segmentation.export_rtstruct", patient_id, src],
                )
            else:
                print("\nИсходная папка неизвестна (нет patient.json) — экспорт пропущен.")
                print(f"Вручную: python -m radonc_segmentation.export_rtstruct {patient_id} \"<исходная папка DICOM>\"")
    except RuntimeError as e:
        total_min = (time.perf_counter() - t_total) / 60
        print(f"\n!!! {e}")
        log_row([time.strftime("%Y-%m-%d %H:%M:%S"), patient_id, args.variant,
                 f"{timings.get('convert', 0):.1f}", f"{timings.get('infer', 0):.1f}",
                 f"{timings.get('rtstruct', 0):.1f}", f"{timings.get('export', 0):.1f}",
                 f"{total_min:.1f}", "FAILED"])
        sys.exit(1)

    total_min = (time.perf_counter() - t_total) / 60
    print(f"\n===== ИТОГ {patient_id} ({args.variant}) =====")
    for k, human in [("convert", "Конвертация"), ("infer", "Инференс"),
                     ("rtstruct", "RTSTRUCT"), ("export", "Экспорт")]:
        print(f"  {human:<12} {timings[k]:>8.1f} с")
    print(f"  {'ВСЕГО':<12} {total_min * 60:>8.1f} с ({total_min:.1f} мин)")
    print(f"RTSTRUCT: {os.path.join(base_dir, 'rtstruct.dcm')}")

    log_row([time.strftime("%Y-%m-%d %H:%M:%S"), patient_id, args.variant,
             f"{timings['convert']:.1f}", f"{timings['infer']:.1f}",
             f"{timings['rtstruct']:.1f}", f"{timings['export']:.1f}",
             f"{total_min:.1f}", "OK"])


if __name__ == "__main__":
    main()
