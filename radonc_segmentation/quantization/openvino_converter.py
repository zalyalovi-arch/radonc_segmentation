import torch
import os
import json
import importlib
import onnx
import openvino as ov
import psutil
import gc
import onnxruntime as ort
import subprocess
import sys

print("=== ЗАПУСК ПОЛНОГО КОНВЕЙЕРА КОНВЕРТАЦИИ ===")

# === НАСТРОЙКИ ===
checkpoint_path = r"C:\nnunet\model_headandneck\nnUNet_results\Dataset001_headandneck\nnUNetTrainerNoMirroring__nnUNetResEncUNetMPlans__3d_fullres\fold_0\checkpoint_latest.pth"
plans_json_path = r"C:\nnunet\model_headandneck\nnUNet_results\Dataset001_headandneck\nnUNetTrainerNoMirroring__nnUNetResEncUNetMPlans__3d_fullres\plans.json"
configuration = "3d_fullres"
output_onnx_path = r"C:\nnunet\model_headandneck\onnx\model.onnx"
openvino_output_dir = r"C:\nnunet\model_headandneck\openvino"

def load_exact_nnunet_model(checkpoint_path: str, plans_json_path: str, configuration: str):
    """Загрузка модели с архитектурой, точно соответствующей checkpoint."""
    print(f"Загрузка checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['network_weights']

    # Загрузка конфигурации из plans.json
    with open(plans_json_path, 'r') as f:
        plans = json.load(f)

    # Определяем количество классов
    num_classes = 32
    for key in state_dict.keys():
        if 'seg_outputs' in key and 'weight' in key:
            num_classes = state_dict[key].shape[0]
            break
    print(f"Количество классов: {num_classes}")

    # Извлекаем параметры архитектуры для указанной конфигурации
    config = plans['configurations'][configuration]
    network_config = config['architecture']['arch_kwargs']
    print("Использованы параметры архитектуры из configurations")

    # Прямой импорт класса сети
    try:
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
        network_class = ResidualEncoderUNet
        print("Успешно импортирован ResidualEncoderUNet")
    except ImportError as e:
        raise ImportError(f"Не удалось импортировать ResidualEncoderUNet: {e}")

    def string_to_class(class_string):
        if class_string is None:
            return None
        if isinstance(class_string, str):
            module_path, class_name = class_string.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        return class_string

    valid_params = {'input_channels': 1, 'num_classes': num_classes}

    for param, value in network_config.items():
        if param == 'conv_op':
            valid_params[param] = string_to_class(value)
        elif param == 'norm_op':
            valid_params[param] = string_to_class(value)
        elif param == 'dropout_op':
            valid_params[param] = string_to_class(value)
        elif param == 'nonlin':
            valid_params[param] = string_to_class(value)
        else:
            valid_params[param] = value

    print(f"Параметры для создания модели: {valid_params}")

    model = network_class(**valid_params)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"Пропущено ключей: {len(missing_keys)}")
    print(f"Неожиданные ключи: {len(unexpected_keys)}")

    if missing_keys:
        print("Отсутствующие ключи (первые 5):", missing_keys[:5])
    if unexpected_keys:
        print("Неожиданные ключи (первые 5):", unexpected_keys[:5])

    model.eval()
    return model

def get_patch_size_from_plans(plans_json_path: str, configuration: str) -> list:
    """Извлекает patch_size из plans.json для указанной конфигурации."""
    with open(plans_json_path, 'r') as f:
        plans = json.load(f)
    if 'configurations' in plans and configuration in plans['configurations']:
        config = plans['configurations'][configuration]
        if 'patch_size' in config:
            return config['patch_size']
    raise KeyError(f"patch_size не найден для конфигурации {configuration}")

# Шаг 1: Загрузка модели и экспорт в ONNX
print("\n=== ШАГ 1: КОНВЕРТАЦИЯ В ONNX ===")
onnx_created = False  # Инициализируем флаг до блока try

try:
    # Получаем patch_size из plans.json
    try:
        patch_size = get_patch_size_from_plans(plans_json_path, configuration)
    except KeyError as e:
        print(f'Ошибка: {e}')
        print("Используем стандартный размер [56, 192, 192] на основе анализа plans.json")
        patch_size = [56, 192, 192]

    input_shape = [1, 1] + patch_size
    print(f'Используемый patch_size: {patch_size}')
    print(f'Форма входа: {input_shape}')

    # Загрузка модели
    model = load_exact_nnunet_model(checkpoint_path, plans_json_path, configuration)
    print('Модель загружена!')

    # Создание dummy-входа
    dummy_input = torch.randn(*input_shape)

    # Экспорт в ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

    print(f'Модель экспортирована в: {output_onnx_path}')
    file_size_kb = os.path.getsize(output_onnx_path) / 1024
    print(f'Размер файла: {file_size_kb:.2f} КБ')

    # Проверка ONNX модели
    print('\n=== ПРОВЕРКА ONNX-МОДЕЛИ ===')
    session = ort.InferenceSession(output_onnx_path)
    inputs = session.get_inputs()
    print('Информация о входах модели:')
    for i, inp in enumerate(inputs):
        print(f'Вход {i}:')
        print(f'  Имя: {inp.name}')
        print(f'  Форма: {inp.shape}')
        print(f'  Тип: {inp.type}')

    # Дополнительная проверка через ONNX
    onnx_model = onnx.load(output_onnx_path)
    print('Входные параметры ONNX-модели:')
    for input in onnx_model.graph.input:
        print(f'  {input.name}:')
        dim_info = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.HasField('dim_param'):
                dim_info.append(f'? ({dim.dim_param})')
            elif dim.HasField('dim_value'):
                dim_info.append(str(dim.dim_value))
            else:
                dim_info.append('?')
        print(f'    Форма: [{", ".join(dim_info)}]')

    onnx_created = True  # Устанавливаем флаг после успешного экспорта и проверки

except Exception as e:
    print(f'❌ Ошибка при создании ONNX-модели: {e}')
    onnx_created = False
    raise

# Шаг 2: Конвертация ONNX → OpenVINO
print("\n=== ШАГ 2: КОНВЕРТАЦИЯ ONNX → OPENVINO ===")
os.makedirs(openvino_output_dir, exist_ok=True)

# Команда с поддерживаемыми параметрами из вашей версии OpenVINO
command = [
    sys.executable, "-m", "openvino.tools.ovc",
    output_onnx_path,
    "--output_model", f"{openvino_output_dir}\\model",
    "--input", f"input[1,1,{patch_size[0]},{patch_size[1]},{patch_size[2]}]",
    #"--compress_to_fp16", "False",
    "--verbose"  # Дополнительная информация о процессе конвертации
]

print("Переконвертация модели с оптимизацией...")
result = subprocess.run(command, capture_output=True, text=True)
print("stdout:", result.stdout)
if result.stderr:
    print("stderr:", result.stderr)

if result.returncode == 0:
    print("✅ Модель успешно переконвертирована с оптимизацией!")
    print(f"✅ Сохранена в: {openvino_output_dir}")

    # Удаляем ONNX-файл после успешной конвертации
    try:
        os.remove(output_onnx_path)
        print(f"✅ Промежуточный ONNX-файл удалён: {output_onnx_path}")
    except OSError as e:
        print(f"⚠️ Не удалось удалить ONNX-файл: {e}")
else:
    print("❌ Ошибка при переконвертации")
    # Если конвертация не удалась, но файл создан, оставляем его для диагностики
    if onnx_created:
        print(f"⚠️ ONNX-файл сохранён для диагностики: {output_onnx_path}")
    raise RuntimeError("Конвертация в OpenVINO не удалась")

print("\n=== ШАГ 3: ПРОВЕРКА OPENVINO-МОДЕЛИ ===")

# Освобождаем память перед проверкой
gc.collect()

# Загружаем модель
core = ov.Core()
model_path = os.path.join(openvino_output_dir, "model.xml")
print(f"Загружаем модель из: {model_path}")

try:
    model = core.read_model(model_path)
    print("✅ Модель прочитана успешно")

    # Получаем информацию о входе
    input_layer = model.input(0)
    input_shape = input_layer.shape
    print(f"✅ Форма входа модели: {input_shape}")

    available_devices = core.available_devices
    print(f"Доступные устройства OpenVINO: {available_devices}")

    compiled_model = None
    used_device = None

    # Проверка доступной памяти
    memory_info = psutil.virtual_memory()
    available_ram_gb = memory_info.available / (1024**3)
    print(f"Доступная оперативная память: {available_ram_gb:.2f} ГБ")

    if available_ram_gb < 4:
        print("⚠️ Внимание: доступно менее 4 ГБ ОЗУ. Возможны проблемы при компиляции.")

    # Приоритет: CPU с минимальной нагрузкой
    if "CPU" in available_devices:
        print("\nПопытка компиляции на CPU с минимальной нагрузкой")
        try:
            config = {"INFERENCE_NUM_THREADS": "1"}
            compiled_model = core.compile_model(model, "CPU", config)
            used_device = "CPU (1 поток)"
            print("✅ Модель успешно скомпилирована на CPU")
        except Exception as e:
            print(f"❌ Ошибка компиляции на CPU: {e}")

    # Резерв: GPU.0 (Intel HD520) с минимальной конфигурацией
    if compiled_model is None and "GPU.0" in available_devices:
        print("\nПопытка компиляции на GPU.0 с минимальной нагрузкой")
        try:
            config = {"PERFORMANCE_HINT": "LATENCY"}
            compiled_model = core.compile_model(model, "GPU.0", config)
            used_device = "GPU.0 (LATENCY)"
            print("✅ Модель успешно скомпилирована на GPU.0")
        except Exception as e:
            print(f"❌ Ошибка компиляции на GPU.0: {e}")

    # Финальная проверка результата компиляции
    if compiled_model is not None:
        print("\n🎉 Проверка OpenVINO модели завершена успешно!")
        print(f"   Устройство для инференса: {used_device}")
        print("   Модель готова к использованию в основном пайплайне.")
    else:
        raise RuntimeError("Не удалось скомпилировать модель ни на одном доступном устройстве")

except Exception as e:
    print(f"❌ Критическая ошибка при проверке OpenVINO модели: {e}")
    raise RuntimeError("Проверка OpenVINO модели не удалась")

print("\n🎉 ПОЛНЫЙ КОНВЕЙЕР ЗАВЕРШЁН УСПЕШНО!")
print("Модель готова к использованию для 32‑классовой 3D‑сегментации!")