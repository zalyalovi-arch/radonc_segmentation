# ============================================================
# validate_model.py — тест модели на валидационном кейсе (шаги 2+3 и 5
# из kaggle/FINAL_STEPS.md): инференс + Dice против GT + гейт качества.
# ------------------------------------------------------------
#   python -m radonc_segmentation.quantization.validate_model \
#       --model headandneck/distilled --reference headandneck/original \
#       --case validation_DICOM_0001 [--folds 0]
#
# Сегмент пишется в validation\<MODEL_NAME>\<case>\seg_nnunet\<case>.nii.gz
# (кейс сразу открывается на вкладке «Постпроцессинг») и копия с суффиксом
# модели <case>__<модель>.nii.gz — для переключения между моделями.
#
# Гейт (FINAL_STEPS.md): Dice(mean) >= 0.8485; малые структуры
# (Lens/Cochlea/TMJ) падают не более чем на 0.03 относительно эталона.
# ============================================================
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from radonc_segmentation import config
from radonc_segmentation import compare_dice as cd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# гейты качества из FINAL_STEPS.md (шаг 3)
GATE_MEAN_DICE = 0.8485
SMALL_STRUCTURES = ("Lens_R", "Lens_L", "TMJoint_R", "TMJoint_L",
                    "Cochlea_R", "Cochlea_L")
SMALL_MAX_DROP = 0.03

TIMING_RE = re.compile(r"Всего: ([\d.]+) с \(([\d.]+) мин\), в среднем ([\d.]+) с/патч")


def sanitize(name):
    """Имя модели -> безопасная часть имени файла."""
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(name))


def gate_verdict(mean_dice, small_dices, ref_small_dices=None):
    """Гейт FINAL_STEPS. Возвращает (passed, строки отчёта).
    mean_dice — float|None; small_dices — {орган: dice} тестируемой модели;
    ref_small_dices — то же для эталона (если задан)."""
    lines = []
    if mean_dice is None:
        return False, ["Mean Dice недоступен (нет GT или геометрия не совпала)"]
    ok = True
    lines.append(f"Dice(mean) = {mean_dice:.4f}  (гейт ≥ {GATE_MEAN_DICE})")
    if mean_dice < GATE_MEAN_DICE:
        ok = False
        lines.append("   ✗ НИЖЕ ПОРОГА")
    else:
        lines.append("   ✓ норма")

    if ref_small_dices:
        lines.append(f"Малые структуры (допустимое падение ≤ {SMALL_MAX_DROP}):")
        for s in SMALL_STRUCTURES:
            a = small_dices.get(s)
            b = ref_small_dices.get(s)
            if a is None and b is None:
                continue
            if a is None or b is None:
                lines.append(f"   {s:<12} тест={_fmt(a)} эталон={_fmt(b)}  (нет в одной из масок — пропущено)")
                continue
            drop = b - a
            good = drop <= SMALL_MAX_DROP + 1e-9
            ok = ok and good
            lines.append(f"   {s:<12} тест={a:.4f} эталон={b:.4f} дельта={-drop:+.4f}  "
                         f"{'✓' if good else '✗ ПАДЕНИЕ > ' + str(SMALL_MAX_DROP)}")
    else:
        present = [(s, small_dices[s]) for s in SMALL_STRUCTURES if s in small_dices]
        if present:
            lines.append("Малые структуры (эталон не задан — без проверки падения):")
            for s, v in present:
                lines.append(f"   {s:<12} {v:.4f}")
    return ok, lines


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def run_inference(case, img_path, out_path, model_name, args, tmp_dir):
    """Прогон инференса одной модели; возвращает (total_s, sec_per_patch).
    Вариант преобработки берётся из settings.ini ([inference] VARIANT):
    модель надо проверять в тех же условиях, в которых будет работать пайплайн."""
    if str(config.DEFAULT_VARIANT).lower() == "aligned":
        module = "radonc_segmentation.aligned.infer_ov_aligned"
    else:
        module = "radonc_segmentation.infer_ov"
    cmd = [sys.executable, "-m", module,
           case, "--input", img_path, "--output", out_path,
           "--force", "--tmp-dir", tmp_dir, "--model", model_name]
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
        raise RuntimeError(f"Инференс {model_name} на {case} упал (код {r.returncode})")
    m = TIMING_RE.search(r.stdout or "")
    total_s = float(m.group(1)) if m else wall
    spp = float(m.group(3)) if m else float("nan")
    return total_s, spp


def dice_dict(pred_path, gt_path):
    rows = cd.compare_files(pred_path, gt_path)
    if rows is None:
        return None
    return {lbl: d for lbl, _n, d in rows if d is not None}


def main():
    p = argparse.ArgumentParser(
        description="Тест модели на валидации: инференс + Dice против GT + гейт качества")
    p.add_argument("--model", required=True, help="имя модели из MODELS_ROOT (напр. headandneck/distilled)")
    p.add_argument("--reference", default=None,
                   help="модель-эталон для дельт и проверки малых структур (напр. headandneck/original)")
    p.add_argument("--case", default=None, help="имя кейса валидации (по умолчанию все с GT)")
    p.add_argument("--n", type=int, default=None, help="ограничить число кейсов (с --case игнорируется)")
    p.add_argument("--folds", default=None, help="фолды (пусто = все у модели), напр. 0")
    p.add_argument("--stride", default=None)
    p.add_argument("--device", default=None, choices=["cpu", "gpu", "gpu.0"])
    args = p.parse_args()

    # проверка модели до долгих прогонов
    test_cfg = config.resolve_model(args.model)
    ref_cfg = config.resolve_model(args.reference) if args.reference else None
    name_by_value = {v: k for k, v in config.LABELS.items()}

    cases = [(c, img, gt) for c, img, gt in config.validation_cases()]
    if args.case:
        cases = [(c, img, gt) for c, img, gt in cases if c == args.case]
        if not cases:
            raise SystemExit(f"Кейс не найден: {args.case} (validation\\{config.MODEL_NAME}\\)")
    if args.n:
        cases = cases[:args.n]
    cases = [(c, img, gt) for c, img, gt in cases if gt]
    if not cases:
        raise SystemExit("Не найдены валидационные кейсы с GT "
                         f"(validation\\{config.MODEL_NAME}\\<кейс>\\{{nnunet_input,gt}})")

    print(f"Модель:      {test_cfg['name']} (фолды: {test_cfg['folds']})")
    if ref_cfg:
        print(f"Эталон:      {ref_cfg['name']} (фолды: {ref_cfg['folds']})")
    print(f"Кейсов:      {len(cases)}; преобработка: {config.DEFAULT_VARIANT}")

    out_dir = os.path.join(config.LOG_DIR, "validate")
    os.makedirs(out_dir, exist_ok=True)

    # (case, model_name) -> (dices, total_s, spp)
    results = {}
    for case, img_path, gt_path in cases:
        print(f"\n===== Случай {case} =====")
        case_segs = os.path.join(config.validation_cases_dir(), case, "seg_nnunet")
        os.makedirs(case_segs, exist_ok=True)
        tmp_dir = os.path.join(out_dir, f"_tmp_{case}")

        for mc in filter(None, (test_cfg, ref_cfg)):
            seg_path = os.path.join(case_segs, f"{case}__{sanitize(mc['name'])}.nii.gz")
            print(f"--- модель {mc['name']} ---")
            total_s, spp = run_inference(case, img_path, seg_path, mc["name"], args, tmp_dir)
            dices = dice_dict(seg_path, gt_path)
            if dices is None:
                print("    !! геометрия предсказания и GT не совпала — метрики недоступны")
            results[(case, mc["name"])] = (dices, total_s, spp)
            if dices:
                print(f"    Dice(mean)={np.mean(list(dices.values())):.4f} | "
                      f"время {total_s:.0f} с ({total_s / 60:.1f} мин) | {spp:.2f} с/патч")
            # каноническое имя: кейс под tested-моделью виден в «Постпроцессинге»
            if mc is test_cfg:
                shutil.copyfile(seg_path, os.path.join(case_segs, f"{case}.nii.gz"))
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    # ---- сводка и гейт ----
    print("\n================ ИТОГИ ================")
    per_case_mean = {}
    for (case, mn), (dices, total_s, spp) in sorted(results.items()):
        if dices:
            per_case_mean[(case, mn)] = float(np.mean(list(dices.values())))
            print(f"{case} | {mn}: Dice(mean)={per_case_mean[(case, mn)]:.4f} | {total_s / 60:.1f} мин")

    test_key = test_cfg["name"]
    test_means = [v for (c, m), v in per_case_mean.items() if m == test_key]
    ref_means = ([v for (c, m), v in per_case_mean.items() if m == ref_cfg["name"]]
                 if ref_cfg else [])

    # гейт по среднему across кейсов; малые структуры — по среднему across кейсов
    def mean_small(mn):
        smalls = {}
        for (c, m), (dices, *_r) in results.items():
            if m != mn or not dices:
                continue
            for s in SMALL_STRUCTURES:
                if s in dices:
                    smalls.setdefault(s, []).append(dices[s])
        return {s: float(np.mean(v)) for s, v in smalls.items()}

    overall_mean = float(np.mean(test_means)) if test_means else None
    ref_overall = float(np.mean(ref_means)) if ref_means else None
    passed, lines = gate_verdict(
        overall_mean, mean_small(test_key),
        mean_small(ref_cfg["name"]) if ref_cfg else None)
    if ref_cfg and ref_overall is not None and overall_mean is not None:
        lines.insert(1, f"Эталон {ref_cfg['name']}: Dice(mean) = {ref_overall:.4f} "
                        f"(дельта {overall_mean - ref_overall:+.4f})")
    verdict = "ГАЙТ ПРОЙДЕН ✓" if passed else "ГАЙТ НЕ ПРОЙДЕН ✗"
    print("\n----- ГЕЙТ КАЧЕСТВА -----")
    for ln in lines:
        print(ln)
    print(f"ВЕРДИКТ: {verdict}")

    # ---- CSV ----
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"validate_{stamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["case", "model", "organ", "dice"])
        for (case, mn), (dices, *_r) in sorted(results.items()):
            if not dices:
                continue
            for lbl in sorted(dices):
                w.writerow([case, mn, name_by_value.get(lbl, lbl), f"{dices[lbl]:.4f}"])
        w.writerow([])
        w.writerow(["gate_mean", GATE_MEAN_DICE, "small_max_drop", SMALL_MAX_DROP])
        w.writerow(["mean_dice", "" if overall_mean is None else f"{overall_mean:.4f}",
                    "passed", "yes" if passed else "no"])
    print(f"\nCSV: {csv_path}")
    print(f"Сегменты: validation\\{config.MODEL_NAME}\\<кейс>\\seg_nnunet "
          "(вкладка «Постпроцессинг»)")


if __name__ == "__main__":
    main()
