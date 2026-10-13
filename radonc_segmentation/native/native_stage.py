# ============================================================
# native/native_stage.py — подготовка нативной nnUNet-модели
# ------------------------------------------------------------
# nnUNetPredictor ожидает trainer-папку:
#   <папка>\plans.json + dataset.json + fold_N\checkpoint_*.pth
# Именно такой layout после импорта .pth создаёт стейджинг
# models\_src\<имя>\ (import_checkpoint.py) — не хватает только
# dataset.json, который здесь генерируется из config.LABELS.
# Весы не копируются: нативный движок работает прямо в _src.
# ============================================================
import json
import os
import re

from radonc_segmentation import config

# приоритет чекпоинта в папке фолда — как в import_checkpoint/export_original
CHECKPOINT_PRIORITY = ("checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth")

_FOLD_RE = re.compile(r"fold_(\d+)$")


def dataset_json_template():
    """dataset.json для нашего 31-классового H&N датасета (для инференса
    значимы labels/channel_names/file_ending; numTraining — нет)."""
    labels = {"background": 0}
    labels.update(config.LABELS)
    return {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": 100,
        "file_ending": ".nii.gz",
        "licence": "hands off!",
    }


def ensure_dataset_json(model_dir, log=print):
    """dataset.json в trainer-папке: пишет шаблон, если файла нет.
    Возвращает путь к нему."""
    path = os.path.join(model_dir, "dataset.json")
    if os.path.isfile(path):
        return path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset_json_template(), f, indent=4, ensure_ascii=False)
    log(f"[native] dataset.json создан (шаблон для {len(config.LABELS)} органов): {path}")
    return path


def _pick_checkpoint_name(model_dir, folds):
    """Единое имя чекпоинта, которое есть во ВСЕХ выбранных фолдах
    (nnUNetPredictor принимает один checkpoint_name на все фолды)."""
    for name in CHECKPOINT_PRIORITY:
        if all(os.path.isfile(os.path.join(model_dir, f"fold_{f}", name))
               for f in folds):
            return name
    return None


def resolve_native_model(name=None):
    """Имя модели из MODELS_ROOT -> параметры для nnUNetPredictor:
    {"name", "dir", "folds", "checkpoint"}.
    Приоритет папок:
      1) models\\_src\\<квантизация> — стейджинг импорта .pth (чекпоинты);
      2) models\\<анатомия>\\<квантизация> — если пользователь положил
         туда полноценное дерево nnUNet_results (plans + fold_N\\*.pth).
    Папка с IR (model.xml) нативному движку не подходит — там нет весов."""
    name = str(name or config.MODEL).strip()
    if "/" in name:
        anatomy, quant = name.split("/", 1)
    else:
        anatomy, quant = config.MODEL_NAME, name
    candidates = [
        os.path.join(config.MODELS_ROOT, "_src", quant),
        os.path.join(config.MODELS_ROOT, name),
        os.path.join(config.MODELS_ROOT, anatomy, quant),
    ]
    for d in candidates:
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "plans.json")):
            continue
        fold_dirs = [fd for fd in os.listdir(d)
                     if _FOLD_RE.fullmatch(fd) and os.path.isdir(os.path.join(d, fd))]
        if not fold_dirs:
            continue
        folds = sorted(int(fd.split("_")[1]) for fd in fold_dirs)
        checkpoint = _pick_checkpoint_name(d, folds)
        if checkpoint is None:
            continue
        return {"name": name, "dir": d, "folds": folds, "checkpoint": checkpoint}
    raise config.ModelNotFoundError(
        f"Для нативного движка (nnunet_native) нужен .pth-чекпоинт модели "
        f"{name!r}, а найден только OpenVINO IR.\n"
        f"Импортируйте чекпоинт (GUI: «Квантизация» → «Импорт модели», "
        f"или python -m radonc_segmentation.quantization.import_checkpoint) —\n"
        f"после импорта нативный движок работает прямо по стейджингу "
        f"models\\_src\\<имя> (весы не копируются).\n"
        f"Проверял: {candidates}")
