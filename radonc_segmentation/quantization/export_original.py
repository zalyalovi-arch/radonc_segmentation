# ============================================================
# export_original.py — экспорт ОРИГИНАЛЬНОЙ (несжатой) модели
# PyTorch-чекпоинты nnUNet -> ONNX -> OpenVINO IR (FP32)
# ------------------------------------------------------------
# Запускается для каждого фолда; результат — папка вида
#   models\original\fold_0\model.xml (+model.bin), ...
# Эту папку затем можно квантизовать (quantize_ov.py) или использовать
# как есть для инференса (модель = "original" в settings.ini).
#
# Запуск:
#   python export_original.py                       # все фолды, дефолтные пути
#   python export_original.py --folds 0             # один фолд
#   python export_original.py --out models\my_model # другая папка вывода
# ВАЖНО: исходные чекпоинты только читаются, ничего не изменяется.
# ============================================================
import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import torch

from radonc_segmentation import config

RESULTS_DIR = (r"C:\nnunet\model_headandneck\nnUNet_results\Dataset001_headandneck"
               r"\nnUNetTrainerNoMirroring__nnUNetResEncUNetMPlans__3d_fullres")
CONFIGURATION = "3d_fullres"
CHECKPOINT_NAMES = ("checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth")


def find_checkpoints(results_dir):
    folds = []
    for name in sorted(os.listdir(results_dir)):
        m = re.fullmatch(r"fold_(\d+)", name)
        if not m or not os.path.isdir(os.path.join(results_dir, name)):
            continue
        fold_dir = os.path.join(results_dir, name)
        for ck in CHECKPOINT_NAMES:
            p = os.path.join(fold_dir, ck)
            if os.path.isfile(p):
                folds.append((int(m.group(1)), p))
                break
        else:
            print(f"  !! в {fold_dir} нет checkpoint-файла — пропуск")
    return folds


def load_nnunet_model(checkpoint_path, plans_path, configuration):
    """Точная реконструкция сети из чекпоинта + plans.json (как в nnUNet-инференсе)."""
    print(f"   Чекпоинт: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ck.get("network_weights", ck.get("state_dict", ck))

    num_classes = 32
    for key in state_dict.keys():
        if "seg_outputs" in key and "weight" in key:
            num_classes = state_dict[key].shape[0]
            break
    print(f"   Классов на выходе: {num_classes}")

    with open(plans_path, "r", encoding="utf-8") as f:
        plans = json.load(f)
    cfg = plans["configurations"][configuration]
    net_kwargs = (cfg.get("network_arch_init_kwargs")
                  or cfg.get("architecture", {}).get("arch_kwargs"))
    if net_kwargs is None:
        raise RuntimeError("Не нашёл параметры архитектуры в plans.json")

    import importlib
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

    def to_class(v):
        if isinstance(v, str):
            module_path, cls = v.rsplit(".", 1)
            return getattr(importlib.import_module(module_path), cls)
        return v

    params = {"input_channels": 1, "num_classes": num_classes}
    for k, v in net_kwargs.items():
        params[k] = to_class(v) if k in ("conv_op", "norm_op", "dropout_op", "nonlin") else v

    model = ResidualEncoderUNet(**params)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"   load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        raise RuntimeError(f"Не хватило весов: {missing[:5]}...")
    model.eval()
    return model, cfg["patch_size"]


def export_fold(fold, ck_path, plans_path, out_fold_dir, configuration, dynamic_batch=False):
    os.makedirs(out_fold_dir, exist_ok=True)
    model, patch_size = load_nnunet_model(ck_path, plans_path, configuration)
    input_shape = [1, 1] + list(patch_size)
    print(f"   ONNX export, вход {input_shape} (opset 11)...")

    tmp_onnx = os.path.join(out_fold_dir, "_tmp_export.onnx")
    dummy = torch.randn(*input_shape)
    dyn = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, tmp_onnx,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes=dyn,
        )
    del model, dummy
    gc.collect()

    # ONNX -> OpenVINO IR напрямую (утилита ovc в этом окружении падает из-за
    # несовместимости openvino_tokenizers 2025 с openvino 2024.6)
    print("   Конвертация в OpenVINO IR (FP32, без сжатия)...")
    import openvino as ov
    ov_model = ov.convert_model(tmp_onnx)
    ov.serialize(ov_model, os.path.join(out_fold_dir, "model.xml"))
    del ov_model
    gc.collect()
    os.remove(tmp_onnx)

    # верификация
    core = ov.Core()
    m = core.read_model(os.path.join(out_fold_dir, "model.xml"))
    compiled = core.compile_model(m, "CPU")
    print(f"   OK: вход {m.input(0).partial_shape}, выход {m.output(0).partial_shape}, компилируется на CPU")
    del compiled, m


def main():
    p = argparse.ArgumentParser(description="Экспорт nnUNet-чекпоинтов в OpenVINO FP32")
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--out", default=os.path.join(config.MODELS_ROOT, "original"))
    p.add_argument("--folds", default=None, help="напр. 0,1 — по умолчанию все найденные")
    p.add_argument("--configuration", default=CONFIGURATION)
    p.add_argument("--dynamic-batch", action="store_true",
                   help="Экспорт с динамическим батчем (нужно для инференса пачками патчей, --batch > 1)")
    args = p.parse_args()

    folds = find_checkpoints(args.results_dir)
    if not folds:
        raise SystemExit(f"Не найдены фолды в {args.results_dir}")
    if args.folds:
        want = {int(x) for x in args.folds.split(",")}
        folds = [f for f in folds if f[0] in want]
    print(f"Фолдов к экспорту: {len(folds)} -> {args.out}")

    plans_path = os.path.join(args.results_dir, "plans.json")
    if not os.path.isfile(plans_path):
        raise SystemExit(f"plans.json не найден: {plans_path}")

    for fold, ck_path in folds:
        print(f"\n=== Фолд {fold} ===")
        out_fold_dir = os.path.join(args.out, f"fold_{fold}")
        if os.path.isfile(os.path.join(out_fold_dir, "model.xml")):
            print("   Уже есть model.xml — пропуск (удалите папку для пересборки)")
            continue
        export_fold(fold, ck_path, plans_path, out_fold_dir, args.configuration,
                    dynamic_batch=args.dynamic_batch)

    print(f"\nГотово: {args.out}")
    print("Модель доступна для инференса по имени папки (settings.ini [model] MODEL=...)")

    # мусор в temp
    gc.collect()


if __name__ == "__main__":
    main()
