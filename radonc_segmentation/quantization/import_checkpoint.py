# ============================================================
# import_checkpoint.py — импорт обученной (дистиллированной) модели
# PyTorch-чекпоинт (.pth с Kaggle) -> OpenVINO IR (FP32)
# ------------------------------------------------------------
# Шаги 0+1 из kaggle/FINAL_STEPS.md одной командой:
#   0) раскладывает скачанный чекпоинт в канонический вид
#      models\_src\<имя>\fold_<N>\checkpoint_*.pth + plans.json
#      (принимает отдельный .pth, .zip из Kaggle Output или папку
#      с вложенными nnUNet_results\...\fold_N\);
#   1) экспортирует в OpenVINO IR FP32 -> models\<MODEL_NAME>\<имя>\fold_<N>\model.xml
# После этого модель видна в GUI как "<MODEL_NAME>/<имя>" (вкладка «Настройки»).
#
# Запуск:
#   python -m radonc_segmentation.quantization.import_checkpoint \
#       --src checkpoint_best.pth --name distilled [--folds 0]
#
# Исходники только читаются/копируются, ничего не изменяется.
# ============================================================
import argparse
import os
import re
import shutil
import zipfile

from radonc_segmentation import config

CHECKPOINT_RE = re.compile(r"checkpoint_(best|final|latest)\.pth$", re.IGNORECASE)
# в папке фолда может лежать несколько чекпоинтов — берём ОДИН (как
# export_original): best > final > latest
CHECKPOINT_PRIORITY = ("checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth")
FOLD_RE = re.compile(r"fold_(\d+)$")
NAME_RE = re.compile(r"[A-Za-z0-9_\-]+")


def check_deps():
    """None если окружение готово, иначе текст с инструкцией по установке."""
    import importlib.util
    missing = [m for m in ("torch", "dynamic_network_architectures", "openvino")
               if importlib.util.find_spec(m) is None]
    if not missing:
        return None
    return ("Отсутствуют пакеты, нужные для экспорта из .pth: " + ", ".join(missing) + "\n"
            "Установите их одной командой:\n"
            "  pip install -e .[quantization]\n"
            "(или заново запустите python setup_env.py — он ставит всё автоматически)")


def _copy_plans(start_dir, stop_dir, dst):
    """Ищет plans.json от start_dir вверх до stop_dir (включительно), затем
    в глубину по всему дереву; копирует в dst. Возвращает путь-источник."""
    d = os.path.abspath(start_dir)
    stop = os.path.abspath(stop_dir)
    while True:
        p = os.path.join(d, "plans.json")
        if os.path.isfile(p):
            shutil.copy2(p, dst)
            return p
        if os.path.samefile(d, stop):
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # запасной вариант: где угодно в дереве (ближайший к корню)
    for root, _dirs, files in os.walk(stop):
        if "plans.json" in files:
            p = os.path.join(root, "plans.json")
            shutil.copy2(p, dst)
            return p
    return None


def _extract_zip(src, dst):
    print(f"Распаковка {src} -> {dst}")
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)
    return dst


def locate_and_stage(src, name, plans=None):
    """Копирует чекпоинт(ы)+plans.json в models\\_src\\<имя>\\.
    Возвращает (staged_dir, [(fold, ck_path), ...]) — один чекпоинт на фолд."""
    staged = os.path.join(config.MODELS_ROOT, "_src", name)
    if os.path.isdir(staged):
        print(f"Папка стейджинга уже есть — файлы перезапишутся: {staged}")
    os.makedirs(staged, exist_ok=True)

    src = os.path.abspath(src)
    was_zip = os.path.isfile(src) and src.lower().endswith(".zip")
    if was_zip:
        src = _extract_zip(src, os.path.join(staged, "_raw"))

    if os.path.isfile(src) and CHECKPOINT_RE.search(os.path.basename(src)):
        # одиночный .pth (напр. checkpoint_best.pth из Kaggle) -> fold_0
        fold_dir = os.path.join(staged, "fold_0")
        os.makedirs(fold_dir, exist_ok=True)
        dst_ck = os.path.join(fold_dir, os.path.basename(src))
        shutil.copy2(src, dst_ck)
        plans_src = plans
        if plans_src:
            shutil.copy2(plans_src, os.path.join(staged, "plans.json"))
        else:
            # plans.json обычно лежит рядом (тот же Output-архив Kaggle)
            plans_src = _copy_plans(os.path.dirname(src), os.path.dirname(src), staged)
        if plans_src is None:
            raise SystemExit(
                "plans.json не найден рядом с чекпоинтом.\n"
                "Скачайте его из того же Output (архива) Kaggle и укажите:\n"
                "  --plans <путь к plans.json>")
        print(f"Чекпоинт: {src} -> {dst_ck}")
        print(f"plans.json: {plans_src} -> {os.path.join(staged, 'plans.json')}")
        return staged, [(0, dst_ck)]

    if os.path.isdir(src):
        # дерево вида nnUNet_results\Dataset...\trainer\fold_N\checkpoint_*.pth
        found = []
        for root, _dirs, files in os.walk(src):
            m = FOLD_RE.fullmatch(os.path.basename(root))
            if not m:
                continue
            cks = [f for f in files if CHECKPOINT_RE.search(f)]
            if not cks:
                print(f"  !! в {root} нет checkpoint-файла — пропуск")
                continue
            cks.sort(key=lambda f: CHECKPOINT_PRIORITY.index(f.lower())
                     if f.lower() in CHECKPOINT_PRIORITY else len(CHECKPOINT_PRIORITY))
            found.append((int(m.group(1)), os.path.join(root, cks[0])))
        if not found:
            raise SystemExit(f"В {src} не найдено fold_N\\checkpoint_*.pth")
        staged_found = []
        for fold, ck in sorted(found):
            fold_dir = os.path.join(staged, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            dst_ck = os.path.join(fold_dir, os.path.basename(ck))
            shutil.copy2(ck, dst_ck)
            # убрать другие чекпоинты, оставшиеся от прошлого импорта
            for f in os.listdir(fold_dir):
                other = os.path.join(fold_dir, f)
                if (CHECKPOINT_RE.search(f)
                        and os.path.abspath(other) != os.path.abspath(dst_ck)):
                    os.remove(other)
            print(f"Чекпоинт: {ck} -> {dst_ck}")
            staged_found.append((fold, dst_ck))
        plans_src = plans
        if plans_src:
            shutil.copy2(plans_src, os.path.join(staged, "plans.json"))
        else:
            plans_src = _copy_plans(os.path.dirname(sorted(found)[0][1]), src, staged)
        if plans_src is None:
            raise SystemExit(
                "plans.json не найден в дереве исходников.\n"
                "Укажите его явно:  --plans <путь к plans.json>")
        print(f"plans.json: {plans_src} -> {os.path.join(staged, 'plans.json')}")
        if was_zip:
            shutil.rmtree(os.path.join(staged, "_raw"), ignore_errors=True)
            print("Временная распаковка _raw удалена (чекпоинты скопированы)")
        return staged, staged_found

    raise SystemExit(f"Не найден источник (файл .pth/.zip или папка): {src}")


def main():
    p = argparse.ArgumentParser(
        description="Импорт .pth-чекпоинта nnUNet в OpenVINO IR FP32 (шаги 0+1 FINAL_STEPS)")
    p.add_argument("--src", required=True,
                   help="файл checkpoint_*.pth, .zip из Kaggle Output или папка с fold_N\\")
    p.add_argument("--plans", default=None, help="plans.json (если не найдётся автоматически)")
    p.add_argument("--name", required=True,
                   help="имя модели, напр. distilled (результат: models\\<MODEL_NAME>\\<имя>)")
    p.add_argument("--folds", default=None, help="напр. 0,1 — по умолчанию все найденные")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--out", default=None,
                   help="папка вывода (по умолчанию models\\<MODEL_NAME>\\<имя>)")
    args = p.parse_args()

    if not NAME_RE.fullmatch(args.name):
        raise SystemExit("Имя модели: латиница/цифры/_/- (напр. distilled)")

    deps_msg = check_deps()
    if deps_msg:
        raise SystemExit(deps_msg)

    staged, found = locate_and_stage(args.src, args.name, args.plans)

    want = None
    if args.folds:
        want = {int(x) for x in args.folds.split(",")}
        found = [(f, ck) for f, ck in found if f in want]
        if not found:
            raise SystemExit(f"Среди найденных фолдов нет {sorted(want)}")

    out = args.out or os.path.join(config.MODELS_ROOT, config.MODEL_NAME, args.name)
    plans_path = os.path.join(staged, "plans.json")

    # экспорт — переиспользуем проверенный export_original
    from radonc_segmentation.quantization import export_original as eo

    print(f"\nЭкспорт {len(found)} фолд(ов) -> {out}")
    for fold, ck in found:
        print(f"\n=== Фолд {fold} ===")
        eo.export_fold(fold, ck, plans_path, os.path.join(out, f"fold_{fold}"),
                       args.configuration)

    model_key = f"{config.MODEL_NAME}/{args.name}"
    print(f"\nГотово: {out}")
    print(f"Модель доступна для инференса и теста как \"{model_key}\" "
          f"(вкладка «Настройки» -> MODEL, или «Квантизация» -> «Тест модели»).")


if __name__ == "__main__":
    main()
