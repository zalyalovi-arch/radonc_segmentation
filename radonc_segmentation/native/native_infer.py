# ============================================================
# native/native_infer.py — инференс оригинальным nnUNet
# (nnUNetv2Predictor, PyTorch). Основной сценарий — NVIDIA GPU;
# без CUDA запуск блокируется (--allow-cpu для проверки
# эквивалентности: на CPU это 65+ мин/кейс против ~6 мин INT8).
# ------------------------------------------------------------
# Отличие от OpenVINO-движка: nnUNet сам делает преобработку
# (ресемплинг в training spacing, CTNormalization по fingerprint),
# sliding window, ансамбль фолдов и запись NIfTI — поэтому варианты
# legacy/aligned здесь не участвуют, а результат всегда
# nnUNet-эквивалентен. На выходе обычный seg NIfTI, дальше
# RTSTRUCT/постпроцессинг работают как раньше.
#
# Запуск:
#   python -m radonc_segmentation.native.native_infer P110
#   python -m radonc_segmentation.native.native_infer P110 --folds 0
# Модель берётся из settings.ini [model] MODEL (или --model).
# Требуется стек nnunetv2: requirements-gpu.txt (NVIDIA, основной режим)
# или requirements-native.txt (поверх torch-CPU, только CPU-проверки).
# ============================================================
import argparse
import csv
import os
import shutil
import sys
import time

from radonc_segmentation import config
from radonc_segmentation.native import native_stage

# nnUNet читает переменные окружения при импорте — задаём их ДО него
# (значения не используются для инференса через Python API, но без них
# paths.py печатает предупреждения при каждом запуске)
def _set_nnunet_env():
    base = os.path.join(config.MODELS_ROOT, "_nnunet_native")
    os.environ.setdefault("nnUNet_raw", os.path.join(base, "nnUNet_raw"))
    os.environ.setdefault("nnUNet_preprocessed", os.path.join(base, "nnUNet_preprocessed"))
    os.environ.setdefault("nnUNet_results", os.path.join(base, "nnUNet_results"))


# чекпоинты, импортированные с Kaggle, ссылаются на кастомный тренер,
# которого нет в stock nnunetv2. Инференсу от класса тренера нужен только
# унаследованный classmethod build_network_architecture (архитектура берётся
# из plans.json), поэтому ЛЮБОЙ неизвестный тренер подменяем базовым —
# файлы site-packages не трогаем. Несовместимые весы поймает строгий
# load_state_dict внутри initialize_from_trained_model_folder.
def _install_trainer_shim():
    import nnunetv2.inference.predict_from_raw_data as pfrm
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    if getattr(pfrm, "_radonc_shim", False):
        return
    orig = pfrm.recursive_find_python_class

    def _finder(folder, class_name, current_module):
        cls = orig(folder, class_name, current_module)
        if cls is None:
            print(f"[native] тренер {class_name} не входит в stock nnunetv2 — "
                  f"использую базовый nnUNetTrainer (для инференса достаточно)")
            return nnUNetTrainer
        return cls

    _finder._radonc_orig = orig
    pfrm.recursive_find_python_class = _finder
    pfrm._radonc_shim = True


# На Windows штатный экспорт предсказания идёт через multiprocessing-пул:
# nnUNetPredictor pickle'ит ПОЛНЫЙ массив логитов (32 класса x объём, ~4.7 ГБ
# для H&N-кейса) и передаёт его воркеру одной записью WriteFile. Больше
# 32-битного лимита DWORD (4 ГБ) такая запись невозможна — пул всегда падает
# с WinError 1450 независимо от объёма RAM. Подменяем predict_from_data_iterator
# синхронной версией: тот же код nnunetv2 (predict_logits_from_preprocessed_data
# + export_prediction_from_logits), но экспорт в основном процессе, без
# pickle/пайпа. Однокейсовые прогоны от асинхронности ничего не теряют.
def _install_lowmem_export_shim():
    import numpy as np
    import torch
    import nnunetv2.inference.predict_from_raw_data as pfrm
    from nnunetv2.inference.predict_from_raw_data import export_prediction_from_logits
    if getattr(pfrm, "_radonc_lowmem_export", False):
        return

    def predict_from_data_iterator_sync(self, data_iterator,
                                        save_probabilities=False,
                                        num_processes_segmentation_export=None):
        ret = []
        for preprocessed in data_iterator:
            data = preprocessed['data']
            if isinstance(data, str):
                tmp = data
                data = torch.from_numpy(np.load(data))
                os.remove(tmp)
            ofile = preprocessed['ofile']
            properties = preprocessed['data_properties']
            if ofile is not None:
                print(f'\nPredicting {os.path.basename(ofile)}:')
            else:
                print(f'\nPredicting image of shape {data.shape}:')
            print(f'perform_everything_on_device: {self.perform_everything_on_device}')
            prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()
            del data
            if ofile is not None:
                print('[native] экспорт: ресемплинг и запись (в основном процессе, '
                      'low-memory режим Windows)')
                export_prediction_from_logits(
                    prediction, properties, self.configuration_manager,
                    self.plans_manager, self.dataset_json, ofile, save_probabilities)
                del prediction
                print(f'done with {os.path.basename(ofile)}')
            else:
                print('[native] экспорт в память (синхронно)')
                ret.append(export_prediction_from_logits(
                    prediction, properties, self.configuration_manager,
                    self.plans_manager, self.dataset_json, ofile, save_probabilities))
        pfrm.compute_gaussian.cache_clear()
        pfrm.empty_cache(self.device)
        return ret

    pfrm.nnUNetPredictor.predict_from_data_iterator = predict_from_data_iterator_sync
    pfrm._radonc_lowmem_export = True


def parse_args():
    p = argparse.ArgumentParser(description="Нативный nnUNet-инференс (PyTorch; GPU CUDA)")
    p.add_argument("patient", help="ID пациента (P110 или 110) — либо путь через --input")
    p.add_argument("--input", default=None,
                   help="Явный путь к NIfTI (по умолчанию <patients>\\PXXX\\nnunet_input\\PXXX_0000.nii.gz)")
    p.add_argument("--output", default=None,
                   help="По умолчанию seg_nnunet\\PXXX_native.nii.gz")
    p.add_argument("--stride", type=float, default=config.STEP_SIZE_RATIO,
                   help=f"Шаг скользящего окна (по умолч. {config.STEP_SIZE_RATIO})")
    p.add_argument("--folds", default=",".join(str(i) for i in config.DEFAULT_FOLDS),
                   help="Фолды через запятую (по умолчанию из settings.ini)")
    p.add_argument("--model", default=None,
                   help="Имя модели из MODELS_ROOT (.pth-чекпоинт; по умолчанию [model] MODEL)")
    p.add_argument("--force", action="store_true", help="Перезаписать существующий результат")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Разрешить CPU вместо CUDA (КРАЙНЕ медленно, ~65+ мин/кейс — "
                        "только для проверки эквивалентности движков)")
    return p.parse_args()


def normalize_patient_id(s):
    s = str(s).strip()
    if s.isdigit():
        return f"P{int(s):03d}"
    if s.startswith("P") and s[1:].isdigit():
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
                            "folds", "patches_per_fold", "sec_per_patch", "total_min", "output"])
            w.writerow(row)
    except OSError as e:
        print(f"   (не удалось записать лог: {e})")


def main():
    args = parse_args()
    if args.input and args.output:
        patient_id = str(args.patient)
    else:
        patient_id = normalize_patient_id(args.patient)
    base_dir = config.patient_dir(patient_id) if patient_id.startswith("P") else os.path.dirname(args.input)

    input_path = args.input or os.path.join(base_dir, "nnunet_input", f"{patient_id}_0000.nii.gz")
    output_path = args.output or os.path.join(base_dir, "seg_nnunet", f"{patient_id}_native.nii.gz")
    if not os.path.isfile(input_path):
        raise SystemExit(f"Входной NIfTI не найден: {input_path}\n"
                         f"Сначала выполните: python dicom_to_nifti.py {patient_id}")
    if os.path.isfile(output_path) and not args.force and not args.output:
        raise SystemExit(f"Результат уже существует: {output_path}\n"
                         f"Используйте --force для перезаписи или --output <файл>.")

    # модель (.pth) — до проверки CUDA, чтобы ошибка «нет чекпоинта»
    # была понятнее ошибки окружения
    try:
        model_cfg = native_stage.resolve_native_model(args.model)
    except config.ModelNotFoundError as e:
        raise SystemExit(str(e))
    folds = sorted(set(int(x) for x in str(args.folds).split(",") if x != "")
                   & set(model_cfg["folds"]))
    if not folds:
        raise SystemExit(f"У модели {model_cfg['name']!r} нет выбранных фолдов "
                         f"({args.folds}); доступны: {model_cfg['folds']}")

    print(f"Пациент: {patient_id} (нативный nnUNet)")
    print(f"Вход:  {input_path}")
    print(f"Выход: {output_path}")
    print(f"Модель: {model_cfg['name']} ({len(folds)} фолд(ов): {folds}), "
          f"чекпоинт {model_cfg['checkpoint']}, папка {model_cfg['dir']}")

    _set_nnunet_env()
    # torch/nnunetv2 — тяжёлые импорты (CUDA-контекст), держим ленивыми
    import torch
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok and not args.allow_cpu:
        raise SystemExit(
            "CUDA недоступна — нативный nnUNet без GPU КРАЙНЕ медленный "
            "(~65+ мин/кейс против ~6 мин у OpenVINO INT8).\n"
            "Варианты: работайте движком openvino (по умолчанию, settings.ini "
            "[inference] ENGINE), либо запустите с --allow-cpu для проверки "
            "эквивалентности (в pipeline.py и GUI этот флаг подставляется сам).")
    device = torch.device("cuda" if cuda_ok else "cpu")
    if not cuda_ok:
        print("!!! ВНИМАНИЕ: CPU-режим — 65+ минут на кейс. Ждите или прерывайте.")
    print(f"Устройство: {device}" + (f" ({torch.cuda.get_device_name(0)})" if cuda_ok else ""))

    t_total = time.perf_counter()

    native_stage.ensure_dataset_json(model_cfg["dir"])
    _install_trainer_shim()
    if sys.platform == "win32":
        _install_lowmem_export_shim()
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=args.stride,
            use_mirroring=True,  # оси из чекпоинта: None => TTA выключен, как учили
            perform_everything_on_device=cuda_ok,
            device=device,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            model_cfg["dir"], use_folds=tuple(folds),
            checkpoint_name=model_cfg["checkpoint"])
    except RuntimeError as e:
        raise SystemExit(str(e))

    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(output_path)
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    predictor.predict_from_files(
        [[input_path]], [os.path.join(out_dir, stem)],
        # по 1 воркеру на препроцессинг/экспорт: каждый spawn-процесс тянет
        # torch (~0.5 ГБ), на Windows-машинах с 8 ГБ RAM пул из 4 воркеров
        # падает с WinError 1450 (нехватка системных ресурсов) при экспорте;
        # предсказание само по себе CPU-bound, воркеры почти не ускоряют
        num_processes_preprocessing=1, num_processes_segmentation_export=1,
        overwrite=True, save_probabilities=False)

    produced = os.path.join(out_dir, stem + ".nii.gz")
    if os.path.abspath(produced) != os.path.abspath(output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(produced, output_path)
    if not os.path.isfile(output_path):
        raise SystemExit(f"nnUNet не создал результат: {produced} — смотрите вывод выше")

    total_s = time.perf_counter() - t_total
    print(f"Готово: {output_path}")
    print(f"Всего: {total_s:.1f} с ({total_s / 60:.1f} мин)")

    append_log([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        patient_id,
        f"native/{model_cfg['name']}",
        args.stride,
        "-",  # кроп/кэширование делает nnUNet, отдельного кропа воздуха нет
        "-",
        ",".join(map(str, folds)),
        "", "",  # патчи/сек-на-патч считает nnUNet внутренне — не экспортируем
        f"{total_s / 60:.1f}",
        output_path,
    ])


if __name__ == "__main__":
    main()
