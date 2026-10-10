import os
import numpy as np
from openvino import Core
from nncf import Dataset, quantize, QuantizationPreset

# --- ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ .NPY ФАЙЛОВ ДЛЯ КАЛИБРОВКИ ---
def load_calibration_from_npy(folder_path, prefix="calib_center_", num_files=20):
    """
    Загружает калибровочные данные из .npy файлов.

    Args:
        folder_path: путь к папке с файлами
        prefix: префикс имени файла (по умолчанию "calib_center_")
        num_files: количество файлов для загрузки (0000–0019)

    Returns:
        Список массивов NumPy из всех файлов
    """
    calibration_data = []

    for i in range(num_files):
        filename = f"{prefix}{i:04d}.npy"
        filepath = os.path.join(folder_path, filename)

        if os.path.exists(filepath):
            data = np.load(filepath)
            # Убедимся, что данные имеют нужную размерность (batch, H, W, C)
            if data.ndim == 3:
                data = np.expand_dims(data, axis=0)  # добавляем batch_size=1
            if data.ndim == 2:
                data = np.expand_dims(data, axis=(0, -1))  # добавляем batch и channel
            calibration_data.append({'data': data})
            print(f"Загружен файл: {filename}")
        else:
            print(f"Файл не найден: {filename}")

    print(f"Всего загружено калибровочных файлов: {len(calibration_data)}")
    return calibration_data

# --- ОСНОВНАЯ ФУНКЦИЯ КВАНТИЗАЦИИ ---
def main():
    # --- НАСТРОЙКИ ---
    original_model_path = r"D:\openvino\openvino_model_fixed\model.xml"
    quantized_model_path = r"D:\openvino\openvino_model_int8\quantized_model.xml"
    calibration_folder = r"D:\openvino\calibration_data"

    # Параметры квантизации
    subset_size = 1  # минимальный размер подмножества для экономии памяти

    # --- ЗАГРУЗКА МОДЕЛИ ---
    print("Загрузка модели...")
    core = Core()
    model = core.read_model(original_model_path)
    print("Модель загружена успешно.")

    # --- СОЗДАНИЕ КАЛИБРОВОЧНОГО ДАТАСЕТА ИЗ .NPY ФАЙЛОВ ---
    print("Загрузка калибровочных данных из .npy файлов...")
    calibration_raw_data = load_calibration_from_npy(
        folder_path=calibration_folder,
        prefix="calib_center_",
        num_files=20
    )

    if not calibration_raw_data:
        raise ValueError("Не удалось загрузить калибровочные данные. Проверьте путь и наличие файлов.")

    # Создаём NNCF Dataset из загруженных .npy данных
    def transform_fn_calibration(sample):
        """Функция преобразования для калибровочного датасета."""
        return {'data': sample['data']}

    calibration_dataset = Dataset(calibration_raw_data, transform_fn_calibration)
    print(f"Калибровочный датасет создан из {len(calibration_raw_data)} .npy файлов.")

    # --- ЭТАП: КВАНТИЗАЦИЯ БЕЗ КОНТРОЛЯ ТОЧНОСТИ ---
    print("Запуск квантизации модели...")
    try:
        quantized_model = quantize(
            model=model,
            calibration_dataset=calibration_dataset,
            preset=QuantizationPreset.MIXED,  # смешанная квантизация (веса INT8, активации динамические)
            subset_size=subset_size,
            fast_bias_correction=True  # быстрая коррекция смещений
        )
        print("Квантизация завершена успешно.")
    except RuntimeError as e:
        if "memory" in str(e).lower():
            raise MemoryError(
                "Недостаточно памяти для квантизации. Попробуйте:\n"
                "1. Уменьшить subset_size (уже установлено 1)\n"
                "2. Использовать GPU (если доступно)\n"
                "3. Запустить на машине с большим объёмом RAM"
            )
        else:
            raise e
    except Exception as e:
        print(f"Ошибка при квантизации: {e}")
        raise e

    # --- СОХРАНЕНИЕ КВАНТИЗОВАННОЙ МОДЕЛИ ---
    print("Сохранение квантизованной модели...")
    os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)

    core.write_model(
        model=quantized_model,
        xml_path=quantized_model_path,
        bin_path=quantized_model_path.replace('.xml', '.bin')
    )

    print(f"Квантизованная модель сохранена:")
    print(f"XML: {quantized_model_path}")
    print(f"BIN: {quantized_model_path.replace('.xml', '.bin')}")
    print("Процесс квантизации успешно завершён!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Критическая ошибка при выполнении квантизации: {e}")
        import traceback
        traceback.print_exc()
