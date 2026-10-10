# ============================================================
# run_ts.py — zero-shot прогон TotalSegmentator v2 на валидации
# ------------------------------------------------------------
# Запускается в env ts_env (python + torch CUDA + pip install
# totalsegmentator). Для каждого кейса из validation\imagesTr:
#   1. TS по нужным таскам (см. ts_to_labels.json "tasks")
#   2. сборка масок TS в ОДИН labelmap в нашей нумерации меток
# Результат: validation\external\totalsegmentator\<case>.nii.gz
#
# Запуск (в отдельном окружении ts_env с torch CUDA + totalsegmentator):
#   python run_ts.py [--device gpu] [--fast]
# Дальнейшее сравнение с нашей моделью и GT: compare_external.py (в nnunet_env).
# ============================================================
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import SimpleITK as sitk

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
sys.path.insert(0, os.path.dirname(SCRIPTS_DIR))  # scripts\ — для config

import config  # noqa: E402

MAP_PATH = os.path.join(SCRIPTS_DIR, "ts_to_labels.json")


def find_cases():
    imgs = sorted(f for f in os.listdir(config.VALIDATION_IMAGES)
                  if f.endswith(".nii.gz") and f.endswith("_0000.nii.gz"))
    return [f[:-len("_0000.nii.gz")] for f in imgs]


# Только эти таски поддерживают --fast; специализированные H&N-таски требуют
# полного разрешения ("task X does not work with option --fast")
FAST_TASKS = {"total", "total_mr"}


def run_ts(ct_path, out_dir, task, device, fast, extra):
    # CLI TS — консольный скрипт env'а (python -m totalsegmentator не работает:
    # у пакета нет __main__)
    ts_exe = os.path.join(os.path.dirname(sys.executable), "Scripts", "totalsegmentator.exe")
    if not os.path.isfile(ts_exe):
        raise SystemExit(f"Не найден CLI TotalSegmentator: {ts_exe}")
    use_fast = fast and task in FAST_TASKS
    cmd = [ts_exe,
           "-i", ct_path, "-o", out_dir, "-ta", task,
           "--device", device, "--quiet"]
    if use_fast:
        cmd.append("--fast")
    cmd += extra
    print(f"  $ {' '.join(cmd)}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"TotalSegmentator ({task}) завершился с кодом {r.returncode}")
    return dt


def build_labelmap(task_outputs, mapping, ref_img):
    """task_outputs: {task: папка с масками <structure>.nii.gz} -> labelmap (z,y,x) uint8."""
    labelmap = None
    used, missing = [], []
    for lbl_str, spec in mapping.items():
        lbl = int(lbl_str)
        mask = None
        src_task = None
        for task in task_outputs:
            for src in spec["sources"]:
                p = os.path.join(task_outputs[task], f"{src}.nii.gz")
                if os.path.isfile(p):
                    arr = sitk.GetArrayFromImage(sitk.ReadImage(p)) > 0
                    mask = arr if mask is None else (mask | arr)
                    src_task = task
                    break
            if mask is not None:
                break
        if mask is None:
            missing.append(spec["organ"])
            continue
        if labelmap is None:
            labelmap = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint8)
        labelmap[mask] = lbl  # больший номер метки перекрывает меньший (как в GT)
        used.append((spec["organ"], lbl, src_task, int(mask.sum())))
    return labelmap, used, missing


def main():
    ap = argparse.ArgumentParser(description="Zero-shot TotalSegmentator v2 на валидации")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu", "gpu.0"])
    ap.add_argument("--no-fast", dest="fast", action="store_false", default=True,
                    help="полное разрешение вместо --fast (медленнее)")
    ap.add_argument("--extra", default="--body_seg",
                    help="доп. флаги totalsegmentator ЧЕРЕЗ ЗАПЯТУЮ "
                         "(по умолчанию --body_seg; напр. "
                         "'--body_seg,--roi_subset,mandible')")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="переопределить список тасков (по умолчанию из ts_to_labels.json)")
    args = ap.parse_args()
    extra = [x for x in args.extra.split(",") if x.strip()]

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    tasks = args.tasks or cfg["tasks"]
    mapping = cfg["mapping"]

    cases = find_cases()
    if not cases:
        raise SystemExit(f"Нет кейсов в {config.VALIDATION_IMAGES} "
                         f"(запустите сначала validation_to_nifti.py)")
    print(f"Кейсов: {len(cases)}: {cases}")
    print(f"Таски TS: {tasks}; device={args.device}, fast={args.fast}\n")

    out_root = os.path.join(os.path.dirname(config.VALIDATION_IMAGES), "external", "totalsegmentator")
    os.makedirs(out_root, exist_ok=True)

    for case in cases:
        ct_path = os.path.join(config.VALIDATION_IMAGES, f"{case}_0000.nii.gz")
        out_path = os.path.join(out_root, f"{case}.nii.gz")
        # пропускаем кейс, только если итоговый labelmap есть И пользователь не
        # попросил конкретные таски (иначе — досчитать недостающее и пересобрать)
        if os.path.isfile(out_path) and not args.tasks:
            print(f"[{case}] уже есть: {out_path} — пропуск (удалите файл для пересчёта)")
            continue
        print(f"=== Кейс {case} ===")
        ref_img = sitk.ReadImage(ct_path)
        task_outputs = {}
        for task in cfg["tasks"]:
            tdir = os.path.join(out_root, f"_{case}_{task}")
            done = os.path.isdir(tdir) and any(f.endswith(".nii.gz") for f in os.listdir(tdir))
            if task in (args.tasks or cfg["tasks"]) and not done:
                os.makedirs(tdir, exist_ok=True)
                try:
                    dt = run_ts(ct_path, tdir, task, args.device, args.fast, extra)
                    print(f"  {task}: {dt:.1f} с")
                except RuntimeError:
                    # частичные результаты не переиспользуем
                    for f in os.listdir(tdir):
                        os.remove(os.path.join(tdir, f))
                    raise
            if os.path.isdir(tdir) and any(f.endswith(".nii.gz") for f in os.listdir(tdir)):
                task_outputs[task] = tdir
            else:
                print(f"  {task}: нет результатов (пропущен)")

        labelmap, used, missing = build_labelmap(task_outputs, mapping, ref_img)
        print(f"  Собрано меток: {len(used)}; нет в TS: {missing or '—'}")
        for organ, lbl, task, n in used:
            print(f"    {lbl:>2} {organ:<18} <- {task:<28} {n:>8} вокс.")

        out_img = sitk.GetImageFromArray(labelmap)
        out_img.CopyInformation(ref_img)
        sitk.WriteImage(out_img, out_path)
        print(f"  -> {out_path}\n")
        # сырые папки тасков оставляем (пересборка маппинга без перезапуска TS);
        # можно удалить вручную — занимают немного (маски сильно сжаты)

    print("Готово. Дальше: compare_external.py (в nnunet_env).")


if __name__ == "__main__":
    main()
