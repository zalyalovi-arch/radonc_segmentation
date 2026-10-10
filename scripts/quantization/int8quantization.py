import os
import nibabel as nib
import numpy as np
from openvino import Core
from nncf import Dataset, quantize, quantize_with_accuracy_control, QuantizationPreset


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

# --- ФУНКЦИИ ДЛЯ ВАЛИДАЦИИ (используют NIfTI) ---
def load_nifti_volume(path):
    """Загружает NIfTI-файл и возвращает numpy-массив."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine, img.header

def preprocess_ct_volume(volume, clip_min=-1000, clip_max=1000):
    """Предобработка КТ-данных: клиппинг и нормализация в [0, 1]."""
    volume = np.clip(volume, clip_min, clip_max)
    volume = (volume - clip_min) / (clip_max - clip_min)  # в [0, 1]
    return volume

def prepare_batch_for_validation(ct_paths, seg_paths=None, use_2d_slices=True):
    """Создаёт батч из КТ-файлов и масок (только для валидации)."""
    batch_data = []
    batch_labels = [] if seg_paths else None

    for i, ct_path in enumerate(ct_paths):
        # Загружаем КТ
        ct_volume, _, _ = load_nifti_volume(ct_path)
        ct_volume = preprocess_ct_volume(ct_volume)

        if use_2d_slices and ct_volume.ndim == 3:
            # Нарезаем 3D объём на 2D срезы вдоль оси Z
            for z in range(ct_volume.shape[0]):
                slice_data = ct_volume[z]  # форма (H, W)
                # Добавляем размерности: batch=1, channel=1
                slice_data = np.expand_dims(slice_data, axis=(0, -1))
                batch_data.append(slice_data)

                if seg_paths:
                    seg_volume, _, _ = load_nifti_volume(seg_paths[i])
            slice_label = seg_volume[z].astype(np.int64)
            slice_label = np.expand_dims(slice_label, axis=(0, -1))
            batch_labels.append(slice_label)
        else:
            # Для 3D модели
            ct_volume = np.expand_dims(ct_volume, axis=0)  # batch_size=1
            batch_data.append(ct_volume)
            if seg_paths:
                seg_volume, _, _ = load_nifti_volume(seg_paths[i])
                seg_volume = np.expand_dims(seg_volume.astype(np.int64), axis=0)
                batch_labels.append(seg_volume)

    # Конкатенируем в батч
    batch_data = np.concatenate(batch_data, axis=0)
    if batch_labels is not None:
        batch_labels = np.concatenate(batch_labels, axis=0)

    return {'data': batch_data, 'label': batch_labels}


def create_validation_dataset(ct_file_list, seg_file_list=None, num_samples=None, use_2d_slices=True):
    """Создаёт NNCF Dataset для валидации из списка КТ-файлов и масок."""
    if num_samples:
        ct_file_list = ct_file_list[:num_samples]
        if seg_file_list:
            seg_file_list = seg_file_list[:num_samples]


    def transform_fn(sample):
        """Функция преобразования для NNCF."""
        return {'data': sample['data']}


    data_items = []
    for i in range(len(ct_file_list)):
        ct_path = ct_file_list[i]
        seg_path = seg_file_list[i] if seg_file_list else None
        sample = prepare_batch_for_validation([ct_path], [seg_path] if seg_path else None, use_2d_slices)
        data_items.append(sample)

    return Dataset(data_items, transform_fn)

# --- ФУНКЦИЯ ДЛЯ РАСЧЁТА МЕТРИКИ КАЧЕСТВА (DICE SCORE) ---
def dice_score(pred, target, smooth=1e-6):
    """Расчёт Dice score между предсказанием и истинной маской."""
    num_classes = max(np.max(pred), np.max(target)) + 1
    dice_scores = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)

        intersection = np.sum(pred_cls * target_cls)
        union = np.sum(pred_cls) + np.sum(target_cls)

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

    return np.mean(dice_scores)

def validation_fn(model, dataset):
    """Функция для расчёта метрики качества (Dice score) на валидационном датасете."""
    core = Core()
    compiled_model = core.compile_model(model)
    output_layer = compiled_model.output(0)

    dice_scores = []

    for data_item in dataset:
        input_data = data_item['data']
        result = compiled_model([input_data])[output_layer]
        predicted_mask = np.argmax(result, axis=1)  # shape: (batch, H, W)
        true_mask = data_item['label'].squeeze(1)  # убираем канал, shape: (batch, H, W)

        for i in range(predicted_mask.shape[0]):
            dice = dice_score(predicted_mask[i], true_mask[i])
            dice_scores.append(dice)

    return float(np.mean(dice_scores))

# --- ОСНОВНАЯ ФУНКЦИЯ С ПОСТЕПЕННОЙ КВАНТИЗАЦИЕЙ ---
def main():
    # --- НАСТРОЙКИ ---
    original_model_path = r"D:\openvino\openvino_model_fixed\model.xml"
    quantized_model_path = r"D:\openvino\openvino_model_int8\quantized_model.xml"

    calibration_folder = r"D:\openvino\calibration_data"

    validation_ct_files = [
        r"D:\patients\P101\gt_folder\imagesTr\medimagetools_0001_0000.nii.gz"
    ]
    validation_seg_files = [
        r"D:\patients\P101\gt_folder\labelsTr\medimagetools_0001.nii.gz"
    ]

    # Параметры квантизации
    num_validation_samples = 1
    max_accuracy_drop = 0.05  # допустимое падение точности
    subset_size = 1  # минимальный размер подмножества для экономии памяти

    # --- ЗАГРУЗКА МОДЕЛИ ---
    core = Core()
    model = core.read_model(original_model_path)

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

    # --- СОЗДАНИЕ ВАЛИДАЦИОННОГО ДАТАСЕТА (на основе NIfTI) ---
    print("Создание датасета для валидации (КТ + сегментации)...")
    validation_dataset = create_validation_dataset(
        validation_ct_files,
        validation_seg_files,
        num_samples=num_validation_samples,
        use_2d_slices=True
    )
    print("Валидационный датасет создан.")

    # --- ЭТАП 1: КВАНТИЗАЦИЯ ТОЛЬКО ВЕСОВ (БЕЗ КОНТРОЛЯ ТОЧНОСТИ) ---
    print("Этап 1: Квантизация только весов модели...")
    try:
        quantized_model = quantize(
            model=model,
            calibration_dataset=calibration_dataset,
            preset=QuantizationPreset.MIXED,
            subset_size=subset_size,
            fast_bias_correction=True  # включаем быструю коррекцию смещений
        )
        print("Квантизация весов завершена успешно.")
    except RuntimeError as e:
        if "memory" in str(e).lower():
            print("Недостаточно памяти для квантизации даже с subset_size=1.")
            raise MemoryError("Недостаточно памяти для квантизации. Попробуйте использовать GPU или машину с большим объёмом RAM.")
        else:
            raise e
    except Exception as e:
        print(f"Ошибка при квантизации весов: {e}")
        raise e

    # --- ЭТАП 2: ТОНКАЯ НАСТРОЙКА С КОНТРОЛЕМ ТОЧНОСТИ ---
    print("Этап 2: Тонкая настройка с контролем точности...")
    try:
        final_quantized_model = quantize_with_accuracy_control(
            model=quantized_model,  # используем модель после первого этапа
            calibration_dataset=calibration_dataset,
            validation_dataset=validation_dataset,
            validation_fn=validation_fn,
            max_drop=max_accuracy_drop,
            drop_type="absolute",
            subset_size=subset_size
        )
        print("Тонкая настройка завершена успешно.")
        used_model = final_quantized_model
    except RuntimeError as e:
        if "memory" in str(e).lower():
            print("Недостаточно памяти для тонкой настройки. Используем модель после квантизации весов.")
            used_model = quantized_model
        else:
            raise e
    except Exception as e:
        print(f"Другая ошибка при тонкой настройке: {e}")
        used_model = quantized_model

    # --- СОХРАНЕНИЕ МОДЕЛИ ---
    print("Сохранение квантизованной модели...")
    os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)

    core.write_model(
        model=used_model,
        xml_path=quantized_model_path,
        bin_path=quantized_model_path.replace('.xml', '.bin')
    )

    print(f"Квантизованная модель сохранена:")
    print(f"XML: {quantized_model_path}")
    print(f"BIN: {quantized_model_path.replace('.xml', '.bin')}")
    print("Квантизация завершена!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ошибка при выполнении квантизации: {e}")
        import traceback
        traceback.print_exc()
