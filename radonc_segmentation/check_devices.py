# check_devices.py — какие устройства вычислений видит OpenVINO.
# Запуск:  python check_devices.py
# Если в списке нет "GPU" — обновите драйвер Intel Graphics (с OpenCL).


import openvino as ov

core = ov.Core()
devices = core.available_devices
print("OpenVINO:", ov.__version__)
print("Доступные устройства:", devices)
for d in devices:
    try:
        name = core.get_property(d, "FULL_DEVICE_NAME")
    except Exception:
        name = "?"
    print(f"  {d}: {name}")

gpu = any(str(d).upper().startswith("GPU") for d in devices)
print()
if gpu:
    print("GPU доступен. Чтобы включить его для инференса:")
    print("  1) в settings.ini: DEVICE = gpu")
    print("  2) первый запуск на GPU перекомпилирует модель (~1-5 мин)")
    print("  3) сравните время с CPU в infer_log.csv — UHD 630 может быть и медленнее CPU")
else:
    print("GPU НЕ виден OpenVINO. Варианты:")
    print("  - обновите драйвер Intel Graphics (нужна поддержка OpenCL 3.0)")
    print("  - если монитор подключён к дискретной видеокарте, iGPU может быть отключён в BIOS")
