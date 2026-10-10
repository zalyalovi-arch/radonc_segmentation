# ============================================================
# ov_infer_core.py — общий движок скользящего окна для OpenVINO
# ------------------------------------------------------------
# Используется infer_ov.py (legacy-преобработка) и
# aligned/infer_ov_aligned.py (nnUNet-преобработка).
#
# Что умеет:
#   - sliding window по (Z,Y,X) с гауссовым смешиванием перекрытий
#   - много-фолдовый ансамбль (накопление логитов, затем argmax)
#   - sync- и async-режим (AsyncInferQueue перекрывает преп/пост-
#     обработку с вычислением)
#   - накопление в np.memmap (32 класса × объём не влезает в 8 ГБ RAM)
#   - тайминги по фолдам и патчам
# ============================================================
import gc
import os
import queue as _queue
import shutil
import time

import numpy as np
import openvino as ov
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def compute_gaussian_map(patch_size, sigma_scale=1 / 8):
    """Гауссова карта важности для смешивания перекрывающихся патчей (как в nnUNet)."""
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_map = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    return gaussian_map.astype(np.float32)


def compute_steps(shape, patch_size, stride_ratio):
    """Начала патчей по каждой оси; последний патч прижимается к краю (как в legacy)."""
    steps_list = []
    for i in range(3):
        d, p = shape[i], patch_size[i]
        if d < p:
            steps_list.append([0])
            continue
        step = max(1, int(p * stride_ratio))
        steps = list(range(0, d - p + 1, step))
        if not steps or steps[-1] + p < d:
            steps.append(d - p)
        steps_list.append(steps)
    return steps_list  # [z_steps, y_steps, x_steps]


def pad_to_patch(data, patch_size, pad_value=0.0):
    """Добивает объём до размера патча, если какая-то ось короче патча."""
    pad_needed = []
    for i in range(3):
        if data.shape[i] < patch_size[i]:
            pad = patch_size[i] - data.shape[i]
            pad_needed.append((pad // 2, pad - pad // 2))
        else:
            pad_needed.append((0, 0))
    if all(p == (0, 0) for p in pad_needed):
        return data, pad_needed
    data = np.pad(data, pad_needed, mode="constant", constant_values=pad_value)
    return data, pad_needed


def unpad(data, pad_needed):
    slices = tuple(
        slice(lo, data.shape[i] - hi) for i, (lo, hi) in enumerate(pad_needed)
    )
    return data[slices]


def body_bbox_zyx(data_zyx, threshold_hu=-300, margin=10):
    """Bounding box тела по HU в кадре (Z, Y, X). Возвращает (z0,z1,y0,y1,x0,x1) или None."""
    mask = data_zyx > threshold_hu
    if not mask.any():
        return None
    axes_ranges = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        present = np.any(mask, axis=other)
        idx = np.where(present)[0]
        lo = max(0, int(idx[0]) - margin)
        hi = min(data_zyx.shape[axis], int(idx[-1]) + 1 + margin)
        axes_ranges.append((lo, hi))
    z0, z1 = axes_ranges[0]
    y0, y1 = axes_ranges[1]
    x0, x1 = axes_ranges[2]
    return z0, z1, y0, y1, x0, x1


def crop_bbox(data_zyx, bbox):
    z0, z1, y0, y1, x0, x1 = bbox
    return data_zyx[z0:z1, y0:y1, x0:x1]


def uncrop_bbox(seg_zyx, original_shape, bbox):
    """Возвращает сегментацию в полный размер, вклеивая кроп в нулевое поле."""
    full = np.zeros(original_shape, dtype=seg_zyx.dtype)
    z0, z1, y0, y1, x0, x1 = bbox
    full[z0:z1, y0:y1, x0:x1] = seg_zyx
    return full


def probe_model(core, model_path, default_patch_size=(56, 192, 192)):
    """Читает модель и возвращает (model_ref, patch_size (Z,Y,X), num_classes).
    Вход может быть [N,C,Z,Y,X] или [C,Z,Y,X] — пространственные размеры всегда
    ПОСЛЕДНИЕ три измерения; статические берём из модели, динамические — из
    default_patch_size."""
    model = core.read_model(model_path)
    p_in = model.input(0).partial_shape
    if len(p_in) >= 3:
        spatial = list(p_in)[-3:]  # (Z, Y, X)
        patch_size = tuple(
            int(s.get_length()) if s.is_static else default_patch_size[i]
            for i, s in enumerate(spatial)
        )
    else:
        patch_size = tuple(default_patch_size)
    p_out = model.output(0).partial_shape
    num_classes = int(p_out[1].get_length()) if len(p_out) >= 2 and p_out[1].is_static else 32
    return model, patch_size, num_classes


def make_compile_config(num_threads, cache_dir, performance_hint=None, device="CPU"):
    """Конфиг компиляции OpenVINO. CPU-специфичные ключи добавляются только для CPU."""
    cfg = {}
    if str(device).lower().startswith("cpu"):
        cfg["INFERENCE_NUM_THREADS"] = str(num_threads)
    if performance_hint and str(performance_hint).upper() != "NONE":
        cfg["PERFORMANCE_HINT"] = str(performance_hint).upper()
    cfg["CACHE_DIR"] = cache_dir
    return cfg


def ensure_device(core, device):
    """Падаем с понятным сообщением, если устройство недоступно."""
    avail = core.available_devices
    base = str(device).upper().split(".")[0]
    if base not in [str(d).upper().split(".")[0] for d in avail]:
        raise SystemExit(
            f"Устройство {device!r} недоступно. OpenVINO видит устройства: {avail}\n"
            f"Если нужен gpu (Intel UHD 630): установите свежий драйвер Intel Graphics\n"
            f"с поддержкой OpenCL и проверьте: python check_devices.py"
        )


def _open_memmaps(tmp_dir, num_classes, shape):
    logits_path = os.path.join(tmp_dir, "ensemble_logits.dat")
    weights_path = os.path.join(tmp_dir, "ensemble_weights.dat")
    logits_bytes = num_classes * int(np.prod(shape)) * 4
    free = shutil.disk_usage(tmp_dir).free
    if free < logits_bytes * 1.1:
        raise RuntimeError(
            f"Недостаточно места на диске для временных файлов: нужно ~{logits_bytes / 1e9:.1f} ГБ, "
            f"свободно {free / 1e9:.1f} ГБ ({tmp_dir})"
        )
    logits_mm = np.memmap(logits_path, dtype=np.float32, mode="w+", shape=(num_classes,) + shape)
    weights_mm = np.memmap(weights_path, dtype=np.float32, mode="w+", shape=shape)
    logits_mm[:] = 0
    weights_mm[:] = 0
    return logits_mm, weights_mm, logits_path, weights_path


def _close_and_remove(mm, path):
    try:
        mm.flush()
    except Exception:
        pass
    try:
        mm._mmap.close()
    except Exception:
        pass
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def run_sliding_window(
    core,
    data_zyx,
    patch_size,
    stride_ratio,
    folds,
    *,
    model_base_path,
    model_filename,
    device,
    compile_config,
    tmp_dir,
    pad_value=0.0,
    n_async=1,
    accumulator="disk",
    batch_size=1,
    log=print,
):
    """
    Основной прогон: пэддинг -> скользящее окно по фолдам -> argmax.
    data_zyx: нормализованный объём (Z, Y, X) float32.
    accumulator: "disk" (memmap-файлы) | "ram" (в ОЗУ; нужен запас ОЗУ:
    ~num_classes * объём * 4 байт, для медианного тома ~7 ГБ).
    batch_size > 1: модель пересобирается на динамический батч, патчи
    подаются пачками (эксперимент с THROUGHPUT-пропускной способностью).
    Возвращает (seg_zyx uint8 в кадре data_zyx, stats dict).
    """
    t_start = time.perf_counter()

    # patch_size и num_classes берём из модели первого фолда
    model0_path = os.path.join(model_base_path.format(folds[0]), model_filename)
    model_ref, patch_size, num_classes = probe_model(core, model0_path)
    del model_ref
    log(f"   Размер патча (Z,Y,X): {patch_size}, классов: {num_classes}")

    data_padded, pad_needed = pad_to_patch(data_zyx, patch_size, pad_value)
    if any(p != (0, 0) for p in pad_needed):
        log(f"   Объём добит до размера патча: {data_padded.shape}")

    steps_list = compute_steps(data_padded.shape, patch_size, stride_ratio)
    patch_indices = [
        (z, y, x) for z in steps_list[0] for y in steps_list[1] for x in steps_list[2]
    ]
    n_patches = len(patch_indices)
    log(f"   Патчей на фолд: {n_patches} (шаг {stride_ratio})")

    use_ram = str(accumulator).lower() == "ram"
    if not use_ram:
        os.makedirs(tmp_dir, exist_ok=True)
    logits_mm = weights_mm = None
    logits_path = weights_path = None
    fold_times = {}

    try:
        if use_ram:
            log("   Накопление логитов в ОЗУ (ACCUMULATOR=ram)")
            logits_mm = np.zeros((num_classes,) + data_padded.shape, dtype=np.float32)
            weights_mm = np.zeros(data_padded.shape, dtype=np.float32)
        else:
            logits_mm, weights_mm, logits_path, weights_path = _open_memmaps(
                tmp_dir, num_classes, data_padded.shape
            )
        gaussian_map = compute_gaussian_map(patch_size)
        # Постоянный буфер взвешивания: экономит аллокацию ~250 МБ на каждый патч
        acc_buf = np.empty((num_classes,) + tuple(patch_size), dtype=np.float32)

        def accumulate(logits_c, z0, y0, x0):
            # logits_c: карта классов (32, Pz, Py, Px) для ОДНОГО патча
            z1 = z0 + patch_size[0]
            y1 = y0 + patch_size[1]
            x1 = x0 + patch_size[2]
            np.multiply(logits_c, gaussian_map, out=acc_buf)
            logits_mm[:, z0:z1, y0:y1, x0:x1] += acc_buf
            weights_mm[z0:z1, y0:y1, x0:x1] += gaussian_map

        for fold in folds:
            t_fold = time.perf_counter()
            model_path = os.path.join(model_base_path.format(fold), model_filename)
            log(f"   Фолд {fold}: загрузка {model_path}")
            model = core.read_model(model_path)
            if batch_size > 1:
                # динамический батч: переэкспорт модели не нужен
                pz, py, px = patch_size
                model.reshape({model.input(0).any_name:
                               ov.PartialShape(f"[1..{batch_size},1,{pz},{py},{px}]")})
            compiled = core.compile_model(model, device, config=compile_config)
            infer_request = compiled.create_infer_request()
            in_name = compiled.input(0).any_name
            out_name = compiled.output(0).any_name

            # патчи группируются в пачки по batch_size
            batches = [patch_indices[i:i + batch_size]
                       for i in range(0, n_patches, batch_size)] if batch_size > 1 \
                else [[idx] for idx in patch_indices]

            pbar = tqdm(
                total=n_patches,
                desc=f"Фолд {fold}",
                unit="патч",
                ncols=100,
                bar_format="{l_bar}{bar:30}{r_bar}",
                dynamic_ncols=True,
            )
            done_count = 0

            def _extract(z0, y0, x0):
                return np.ascontiguousarray(
                    data_padded[z0:z0 + patch_size[0],
                                y0:y0 + patch_size[1],
                                x0:x0 + patch_size[2]]
                )

            if n_async > 1:
                done_q: _queue.Queue = _queue.Queue()

                def _cb(infer_request, userdata):
                    # Копируем выход, пока запрос не переиспользован
                    out = infer_request.get_output_tensor(0).data
                    done_q.put((userdata, np.array(out, copy=True)))

                async_q = ov.AsyncInferQueue(compiled, n_async)
                async_q.set_callback(_cb)

                for batch_indices in batches:
                    if len(batch_indices) == 1:
                        z0, y0, x0 = batch_indices[0]
                        tensor = _extract(z0, y0, x0).reshape((1, 1) + patch_size)
                    else:
                        tensor = np.stack([_extract(*idx) for idx in batch_indices])[:, None]
                    # start_async блокируется, пока нет свободного запроса —
                    # естественный backpressure, очередь результатов не растёт
                    async_q.start_async({in_name: tensor}, userdata=tuple(batch_indices))
                    # Разбираем готовые результаты, пока основной поток готовился к следующей пачке
                    while done_q.qsize() >= n_async:
                        batch_ud, logits = done_q.get()
                        for i, (z0, y0, x0) in enumerate(batch_ud):
                            accumulate(logits[i], z0, y0, x0)
                        done_count += len(batch_ud)
                        pbar.update(len(batch_ud))
                async_q.wait_all()
                while not done_q.empty():
                    batch_ud, logits = done_q.get()
                    for i, (z0, y0, x0) in enumerate(batch_ud):
                        accumulate(logits[i], z0, y0, x0)
                    done_count += len(batch_ud)
                    pbar.update(len(batch_ud))
                del async_q
            else:
                for batch_indices in batches:
                    if len(batch_indices) == 1:
                        z0, y0, x0 = batch_indices[0]
                        tensor = _extract(z0, y0, x0).reshape((1, 1) + patch_size)
                    else:
                        tensor = np.stack([_extract(*idx) for idx in batch_indices])[:, None]
                    result = infer_request.infer(inputs={in_name: tensor})
                    logits = result[out_name]
                    for i, (z0, y0, x0) in enumerate(batch_indices):
                        accumulate(logits[i], z0, y0, x0)
                    done_count += len(batch_indices)
                    pbar.update(len(batch_indices))

            pbar.close()
            if done_count != n_patches:
                raise RuntimeError(f"Фолд {fold}: обработано {done_count}/{n_patches} патчей")

            fold_times[fold] = time.perf_counter() - t_fold
            log(f"   Фолд {fold} готов за {fold_times[fold]:.1f} с "
                f"({fold_times[fold] / n_patches:.2f} с/патч)")
            del model, compiled, infer_request
            gc.collect()

        # --- Финализация ---
        log("   Нормировка по гауссовым весам и усреднение фолдов...")
        safe_weights = np.where(weights_mm > 1e-8, weights_mm, 1.0)
        logits_mm /= safe_weights
        del safe_weights
        logits_mm /= len(folds)

        seg = np.empty(data_padded.shape, dtype=np.uint8)
        z_chunk = 16
        for z0 in range(0, data_padded.shape[0], z_chunk):
            z1 = min(z0 + z_chunk, data_padded.shape[0])
            block = np.asarray(logits_mm[:, z0:z1], dtype=np.float32)
            seg[z0:z1] = np.argmax(block, axis=0).astype(np.uint8)
            del block

        seg = unpad(seg, pad_needed)
        total_s = time.perf_counter() - t_start
        stats = {
            "n_patches_per_fold": n_patches,
            "folds": list(folds),
            "patch_size": tuple(int(p) for p in patch_size),
            "num_classes": num_classes,
            "fold_times_s": fold_times,
            "total_s": total_s,
            "sec_per_patch": sum(fold_times.values()) / max(1, n_patches * len(folds)),
            "shape_zyx": list(data_zyx.shape),
            "padded": any(p != (0, 0) for p in pad_needed),
            "batch_size": int(batch_size),
        }
        return seg, stats

    finally:
        gc.collect()
        if logits_mm is not None:
            if logits_path is not None:
                _close_and_remove(logits_mm, logits_path)
            else:
                del logits_mm
        if weights_mm is not None:
            if weights_path is not None:
                _close_and_remove(weights_mm, weights_path)
            else:
                del weights_mm
        # пустую временную папку тоже убираем (если не пользовательская)
        try:
            if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass
        gc.collect()
