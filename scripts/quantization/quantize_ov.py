# ============================================================
# quantize_ov.py — квантизация OpenVINO-моделей для экспериментов
# с качеством оконтуривания (INT8 / INT4)
# ------------------------------------------------------------
# Режимы:
#   int8         — полное PTQ-квантизация NNCF (FakeQuantize 8 бит),
#                  нужна калибровка на реальных КТ-патчах (как было сделано
#                  для models\int8)
#   int4         — та же полная PTQ, но ВЕСА 4-битные (активации 8 бит):
#                  калибровка + bias correction, как у int8. Недокументированный
#                  путь NNCF (QuantizationParameters.num_bits) — качество
#                  просядет заметнее int8; ускорения на CPU с VNNI не будет.
#                  Биты настраиваются: --weights-bits (2..8), --act-bits.
#   int8_weights — сжатие только ВЕСОВ до int8 (быстро, без калибровки)
#   int4_weights — веса до 4 бит. ВНИМАНИЕ: в nncf 2.14 4-битную основную
#                  конфигурацию получают только MatMul-слои, плотные
#                  Convolution всегда остаются в резервной INT8 — для
#                  ResEnc UNet (без матмулов) результат == int8_weights.
#
# Запуск:
#   python quantize_ov.py --source ..\models\original --name int8_exp1 --mode int8
#   python quantize_ov.py --source ..\models\original --name int4_exp1 --mode int4
#   python quantize_ov.py --list              # какие модели есть
#
# Результат: models\<name>\[fold_X\]quantized_model.xml — модель сразу
# доступна для инференса (settings.ini [model] MODEL=<name> или --model <name>).
# Источник только читается.
# ============================================================
import argparse
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# NNCF при импорте пробует загрузить TF-бэкенд; TensorFlow в этом окружении
# сломан (скампилирован под NumPy 1.x, стоит NumPy 2.2) и валит весь импорт.
# None в sys.modules превращает `import tensorflow` в ImportError, который
# NNCF корректно обрабатывает (TF для квантизации OV-моделей не нужен).
sys.modules.setdefault("tensorflow", None)
sys.modules.setdefault("tensorboard", None)

import nncf
import openvino as ov


def list_models():
    for name in config.available_models():
        try:
            m = config.resolve_model(name)
            kind = f"ансамбль x{len(m['folds'])} фолдов" if m["ensemble"] else "одиночная"
            print(f"  {name:<20} {kind}, {m['filename']}")
        except SystemExit as e:
            print(f"  {name:<20} (не читается: {e})")


def resolve_source(source):
    """Тот же принцип, что config.resolve_model, но для произвольной папки."""
    base = os.path.abspath(source)
    if not os.path.isdir(base):
        raise SystemExit(f"Папка источника не найдена: {base}")
    fold_dirs = [d for d in os.listdir(base)
                 if re.fullmatch(r"fold_\d+", d) and os.path.isdir(os.path.join(base, d))]
    if fold_dirs:
        folds = sorted(int(d.split("_")[1]) for d in fold_dirs)
        units = [(f, os.path.join(base, f"fold_{f}")) for f in folds]
    else:
        units = [(0, base)]
    for _f, d in units:
        config._detect_xml(d)  # проверка наличия xml
    return units


def build_calibration(n_volumes, patches_per_volume, seed=42):
    """
    Патчи калибровки из реальных КТ пациентов (legacy-преобработка,
    как в infer_ov.py). Кэш: models\_calibration.npz.
    """
    cache = os.path.join(config.MODELS_ROOT, "_calibration.npz")
    if os.path.isfile(cache):
        data = np.load(cache)["patches"]
        print(f"Калибровка из кэша: {cache} ({data.shape[0]} патчей)")
        return data

    import nibabel as nib
    from scipy.ndimage import gaussian_filter  # noqa: F401 (совместимость)
    rng = random.Random(seed)
    volumes = []
    for pid in sorted(os.listdir(config.PATIENTS_ROOT)):
        p = os.path.join(config.PATIENTS_ROOT, pid, "nnunet_input", f"{pid}_0000.nii.gz")
        if os.path.isfile(p):
            volumes.append(p)
    if not volumes:
        raise SystemExit(f"Нет NIfTI пациентов в {config.PATIENTS_ROOT} — "
                         f"импортируйте хотя бы одного пациента для калибровки")
    rng.shuffle(volumes)
    volumes = volumes[:n_volumes]
    print(f"Калибровка: {len(volumes)} томов x {patches_per_volume} патчей")

    patch_shape = config.DEFAULT_PATCH_SIZE  # (Z, Y, X)
    all_patches = []
    for path in volumes:
        nii = nib.load(path)
        data = nii.get_fdata(dtype=np.float32)          # (X, Y, Z)
        data = np.transpose(data, (2, 1, 0))            # (Z, Y, X)
        # legacy-нормализация infer_ov.py
        lo, hi = np.percentile(data, (0.5, 99.5))
        data = np.clip(data, lo, hi)
        if hi > lo:
            data = (data - lo) / (hi - lo)
        mean, std = data.mean(), data.std()
        if std > 0:
            data = (data - mean) / std
        # случайные центры патчей (с отступом от краёв)
        max_z = max(1, data.shape[0] - patch_shape[0])
        max_y = max(1, data.shape[1] - patch_shape[1])
        max_x = max(1, data.shape[2] - patch_shape[2])
        for _ in range(patches_per_volume):
            z = rng.randint(0, max_z)
            y = rng.randint(0, max_y)
            x = rng.randint(0, max_x)
            patch = data[z:z + patch_shape[0], y:y + patch_shape[1], x:x + patch_shape[2]]
            if patch.shape == tuple(patch_shape):
                all_patches.append(patch.astype(np.float32))
    patches = np.stack(all_patches)[:, np.newaxis, ...]  # (N, 1, Z, Y, X)
    os.makedirs(config.MODELS_ROOT, exist_ok=True)
    np.savez_compressed(cache, patches=patches)
    print(f"Кэш калибровки: {cache} ({patches.shape})")
    return patches


def patch_hw_config_for_low_bits(bits):
    """
    Стандартный CPU HW-конфиг NNCF допускает только 8-битные веса, поэтому
    PTQ с num_bits<8 падает на проверке совместимости. Генерируем расширенный
    конфиг (разрешаем низкобитные веса для всех операций, где есть веса)
    и подменяем путь к нему на время этой квантизации.
    """
    import jstyleson
    from nncf.common.hardware.config import HWConfig, HWConfigType

    base = HWConfig.get_path_to_hw_config(HWConfigType.CPU)
    with open(base, "r", encoding="utf-8") as f:
        cfg = jstyleson.load(f)

    q = cfg["config"]["quantization"]
    aliases = []
    for suffix, mode in (("sym", "symmetric"), ("asym", "asymmetric")):
        alias = f"q{bits}w_{suffix}"
        # narrow_range обязателен в nncf >= 2.19
        q[alias] = {"bits": bits, "mode": mode,
                    "granularity": ["perchannel", "pertensor"],
                    "narrow_range": False}
        aliases.append(alias)

    patched = 0
    for op in cfg.get("operations", []):
        quant = op.get("quantization")
        if quant and "weights" in quant:
            w = quant["weights"]
            if isinstance(w, str):  # у части операций weights — голая строка
                w = [w]
            quant["weights"] = aliases + [x for x in w if x not in aliases]
            patched += 1

    os.makedirs(config.MODELS_ROOT, exist_ok=True)
    out = os.path.join(config.MODELS_ROOT, "_hw_config_lowbits.json")
    with open(out, "w", encoding="utf-8") as f:
        f.write(jstyleson.dumps(cfg, indent=4))

    HWConfig.get_path_to_hw_config = staticmethod(lambda t: out)
    print(f"   HW-конфиг расширен до {bits}-битных весов ({patched} операций): {out}")


def make_calib_dataset(calib_patches):
    """nncf.Dataset из патчей; каждый элемент — (1,1,Z,Y,X), как вход модели."""
    items = [p[np.newaxis, ...] for p in calib_patches]
    return nncf.Dataset(items)


def quantize_one(xml_path, out_dir, mode, calib_patches, weights_bits=4, act_bits=8):
    model = ov.Core().read_model(xml_path)
    if mode == "int8":
        q = nncf.quantize(model, make_calib_dataset(calib_patches),
                          preset=nncf.QuantizationPreset.MIXED,
                          subset_size=min(len(calib_patches), 32))
    elif mode == "int4":
        # Полная PTQ с 4-битными весами: недокументированный путь через
        # QuantizationParameters.num_bits — NNCF не проверяет и не запрещает.
        from nncf.quantization.advanced_parameters import (
            AdvancedQuantizationParameters,
            QuantizationParameters,
        )
        adv = nncf.AdvancedQuantizationParameters(
            weights_quantization_params=QuantizationParameters(num_bits=weights_bits),
            activations_quantization_params=QuantizationParameters(num_bits=act_bits),
            overflow_fix=nncf.OverflowFix.DISABLE,  # overflow-фикс про 8 бит
        )
        q = nncf.quantize(model, make_calib_dataset(calib_patches),
                          preset=nncf.QuantizationPreset.MIXED,
                          subset_size=min(len(calib_patches), 32),
                          advanced_parameters=adv)
    elif mode == "int8_weights":
        # в nncf 2.14 INT8-режимам all_layers передавать нельзя (только int4)
        q = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM,
                                  ratio=1.0, group_size=-1)
    elif mode == "int4_weights":
        try:
            q = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT4_ASYM,
                                      all_layers=True, ratio=1.0, group_size=64)
        except (TypeError, nncf.ParameterNotSupportedError):
            q = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT4_ASYM,
                                      group_size=64)
    else:
        raise SystemExit(f"Неизвестный режим: {mode}")
    os.makedirs(out_dir, exist_ok=True)
    out_xml = os.path.join(out_dir, "quantized_model.xml")
    ov.serialize(q, out_xml)
    del q, model

    # Верификация: загрузка + компиляция + пробный прогон (ловит проблемы
    # свёртки низкобитных FakeQuantize в OpenVINO до того, как модель уедет
    # в инференс)
    m = ov.Core().read_model(out_xml)
    inp = tuple(int(v) for v in m.input(0).shape)
    try:
        compiled = ov.Core().compile_model(m, "CPU")
        res = compiled(np.zeros(inp, dtype=np.float32))
        out_shape = tuple(int(v) for v in compiled.output(0).shape)
        del res, compiled
    except Exception as e:
        raise SystemExit(
            f"Сериализованная модель не исполняется на CPU ({e}).\n"
            f"Для низкобитных режимов это может означать, что OpenVINO "
            f"{ov.__version__} не поддерживает свёртку таких FakeQuantize — "
            f"попробуйте другой режим или обновите openvino/nncf."
        )
    size_mb = os.path.getsize(os.path.join(out_dir, "quantized_model.bin")) / 1e6
    print(f"   OK: {out_xml} ({size_mb:.0f} МБ, вход {inp}, выход {out_shape})")
    del m


def main():
    p = argparse.ArgumentParser(description="Квантизация OpenVINO-моделей (INT8/INT4)")
    p.add_argument("--source", default=None, help="Папка модели-источника (models\<имя>)")
    p.add_argument("--name", default=None, help="Имя новой модели (подпапка в models)")
    p.add_argument("--mode", default="int8",
                   choices=["int8", "int4", "int8_weights", "int4_weights"])
    p.add_argument("--weights-bits", type=int, default=4, choices=range(2, 9),
                   help="Бит весов для режима int4 (по умолчанию 4)")
    p.add_argument("--act-bits", type=int, default=8, choices=range(2, 9),
                   help="Бит активаций для режима int4 (по умолчанию 8)")
    p.add_argument("--n-volumes", type=int, default=6, help="Томов для калибровки (int8/int4)")
    p.add_argument("--patches-per-volume", type=int, default=4)
    p.add_argument("--regen-calib", action="store_true", help="Пересобрать кэш калибровки")
    p.add_argument("--list", action="store_true", help="Показать доступные модели")
    p.add_argument("--folds", default=None, help="Квантизовать только эти фолды (напр. 0)")
    args = p.parse_args()

    if args.list:
        print(f"Модели в {config.MODELS_ROOT}:")
        list_models()
        return
    if not args.source or not args.name:
        raise SystemExit("Укажите --source и --name (или --list)")

    if os.path.isdir(os.path.join(config.MODELS_ROOT, args.name)):
        raise SystemExit(f"Модель {args.name!r} уже существует — выберите другое имя")

    if args.regen_calib and os.path.isfile(os.path.join(config.MODELS_ROOT, "_calibration.npz")):
        os.remove(os.path.join(config.MODELS_ROOT, "_calibration.npz"))

    units = resolve_source(args.source)
    if args.folds:
        want = {int(x) for x in args.folds.split(",")}
        units = [u for u in units if u[0] in want]
    bits_note = f", веса {args.weights_bits} бит / активации {args.act_bits} бит" if args.mode == "int4" else ""
    print(f"Источник: {os.path.abspath(args.source)} -> модели: {args.name} "
          f"(режим {args.mode}{bits_note}, единиц: {len(units)})")

    calib = None
    if args.mode in ("int8", "int4"):
        calib = build_calibration(args.n_volumes, args.patches_per_volume)
    if args.mode == "int4":
        patch_hw_config_for_low_bits(args.weights_bits)

    out_base = os.path.join(config.MODELS_ROOT, args.name)
    for fold, unit_dir in units:
        xml = os.path.join(unit_dir, config._detect_xml(unit_dir))
        print(f"  Единица fold_{fold}: {xml}")
        out_dir = os.path.join(out_base, f"fold_{fold}") if len(units) > 1 or "fold_" in os.path.basename(unit_dir) else out_base
        quantize_one(xml, out_dir, args.mode, calib,
                     weights_bits=args.weights_bits, act_bits=args.act_bits)

    print(f"\nГотово. Модель {args.name!r} доступна для инференса:")
    print(f"  python infer_ov.py PXXX --model {args.name}")
    print(f"  или settings.ini [model] MODEL = {args.name}")


if __name__ == "__main__":
    main()
