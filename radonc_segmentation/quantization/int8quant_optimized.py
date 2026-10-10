import os
import numpy as np
from openvino import Core
from nncf import Dataset, quantize, QuantizationPreset, IgnoredScope

def create_memory_efficient_calibration_dataset(folder_path, prefix="resized_calib_center_"):
    def data_generator():
        files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith('.npy')]
        print(f"Найдено калибровочных файлов: {len(files)}")
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            if os.path.exists(filepath):
                data = np.load(filepath, mmap_mode='r')
                if data.shape != (1, 1, 28, 96, 96):
                    print(f"Предупреждение: неожиданный размер данных в {filename}: {data.shape}")
                    continue
                yield {'data': data.copy()}
    return Dataset(data_generator())

def main():
    original_model_path = r"D:\openvino\openvino_model_fixed\model.xml"
    quantized_model_path = r"D:\openvino\openvino_model_int8\quantized_model.xml"
    calibration_folder = r"D:\openvino\calibration_data_resized"

    # Вывод имён всех узлов модели (для отладки)
    print("Анализ структуры модели...")
    print_model_nodes(original_model_path)

    core = Core()
    model = core.read_model(original_model_path)
    print("Модель загружена на CPU.")

    calibration_dataset = create_memory_efficient_calibration_dataset(calibration_folder)
    total_files = len([f for f in os.listdir(calibration_folder) if f.startswith('resized_calib_center_') and f.endswith('.npy')])
    print(f"Всего калибровочных файлов: {total_files}")

    # Безопасный IgnoredScope — без строгих проверок
    ignored_operations = IgnoredScope(
        types=["BatchNormalization", "Add"],  # часто тяжёлые операции
        names=[],
        validate=False
    )

    print("Запуск квантизации с оптимизированными параметрами...")
    try:
        quantized_model = quantize(
            model=model,
            calibration_dataset=calibration_dataset,
            preset=QuantizationPreset.MIXED,
            subset_size=min(1, total_files),  # уменьшаем выборку
            fast_bias_correction=True,
            ignored_scope=ignored_operations
        )
        print("Квантизация завершена успешно.")
    except RuntimeError as e:
        if "memory" in str(e).lower():
            raise MemoryError(
                "Недостаточно памяти. Попробуйте:\n"
                "1. Уменьшить subset_size до 5\n"
                "2. Расширить ignored_operations\n"
                "3. Проверить размерность файлов (1, 1, 28, 96, 96)"
            )
        else:
            raise e

    os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)
    core.write_model(
        model=quantized_model,
        xml_path=quantized_model_path,
        bin_path=quantized_model_path.replace('.xml', '.bin')
    )
    print(f"Модель сохранена: {quantized_model_path}")

def print_model_nodes(model_path):
    """Выводит все имена узлов модели для выбора игнорируемых слоёв."""
    core = Core()
    model = core.read_model(model_path)

    print("\n--- Узлы модели (OpenVINO IR) ---")
    for layer in model.get_ordered_ops():
        print(f"Имя: {layer.get_friendly_name()}, Тип: {layer.get_type_name()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()