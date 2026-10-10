# ============================================================
# test_core_functions.py — юнит-тесты чистых функций ядра
# ------------------------------------------------------------
# Без GUI и реальных данных: метрика Dice, геометрия патчей
# инференса, кроп тела, парсер спеков версий, нормализация ID
# пациента, целостность словаря структур.
# Запуск: python -m pytest tests (из корня проекта).
# ============================================================
import numpy as np
import pytest

from radonc_segmentation import config
from radonc_segmentation.compare_dice import dice
from radonc_segmentation.env_check import _spec_ok
from radonc_segmentation.ov_infer_core import body_bbox_zyx, compute_steps
from radonc_segmentation.pipeline import normalize_patient_id


# ---------- dice: метрика, на которой принимаются все решения ----------

def test_dice_identical_masks_is_one():
    a = np.zeros((10, 10, 10), dtype=np.uint8)
    a[2:5, 2:5, 2:5] = 1
    assert dice(a, a.copy()) == 1.0


def test_dice_disjoint_masks_is_zero():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[0, 0, 0] = True
    b[9, 9, 9] = True
    assert dice(a, b) == 0.0


def test_dice_both_empty_is_none():
    # орган отсутствует в обеих масках — не 0.0 и не деление на ноль
    empty = np.zeros((5, 5, 5), dtype=np.uint8)
    assert dice(empty, empty.copy()) is None


def test_dice_one_empty_is_zero():
    a = np.zeros((5, 5, 5), dtype=np.uint8)
    b = np.zeros((5, 5, 5), dtype=np.uint8)
    b[1:3, 1:3, 1:3] = 1
    assert dice(a, b) == 0.0


def test_dice_known_partial_overlap():
    # 10 vs 10 вокселей, пересечение 5 -> 2*5/(10+10) = 0.5
    a = np.zeros((2, 2, 20), dtype=bool)
    b = np.zeros((2, 2, 20), dtype=bool)
    a[0, 0, :10] = True
    b[0, 0, 5:15] = True
    assert dice(a, b) == pytest.approx(0.5)


def test_dice_normalizes_dtype():
    # uint8-маска со значением 255 — не 255 «единиц», а одна
    a = np.zeros((4, 4, 4), dtype=np.uint8)
    b = np.zeros((4, 4, 4), dtype=np.uint8)
    a[0, 0, 0] = 255
    b[0, 0, 0] = 255
    assert dice(a, b) == 1.0


# ---------- compute_steps: геометрия инференса ----------

def _patches_cover_volume(shape, patch, stride_ratio):
    """ union патчей покрывает каждый воксель объёма."""
    steps = compute_steps(shape, patch, stride_ratio)
    covered = np.zeros(shape, dtype=bool)
    for z in steps[0]:
        for y in steps[1]:
            for x in steps[2]:
                covered[z:z + patch[0], y:y + patch[1], x:x + patch[2]] = True
    return covered.all()


def test_compute_steps_full_coverage():
    for shape, patch, ratio in [
        ((100, 200, 300), (56, 192, 192), 0.5),   # типичный КТ-том
        ((128, 256, 256), (56, 192, 192), 0.25),  # плотная сетка
        ((56, 192, 192), (56, 192, 192), 0.5),    # том ровно в патч
        ((30, 64, 64), (56, 192, 192), 0.5),      # ось Z короче патча
        ((1, 1, 1), (56, 192, 192), 0.5),         # вырожденный объём
    ]:
        assert _patches_cover_volume(shape, patch, ratio), (shape, patch, ratio)


def test_compute_steps_axis_shorter_than_patch():
    # ось короче патча -> единственное начало [0] (добивает pad_to_patch)
    steps = compute_steps((30, 200, 200), (56, 192, 192), 0.5)
    assert steps[0] == [0]


def test_compute_steps_last_step_clamped_to_edge():
    # последний патч прижимается к краю: конец последнего патча == размер оси
    d, p, ratio = 300, 64, 0.5
    steps = compute_steps((d, 64, 64), (p, 64, 64), ratio)
    assert steps[0][-1] + p == d
    # и обычный шаг соблюдается до прижатия
    assert steps[0][0] == 0


def test_compute_steps_exact_fit_single_patch():
    assert compute_steps((64, 64, 64), (64, 64, 64), 0.5) == [[0], [0], [0]]


# ---------- body_bbox_zyx: кроп тела ----------

def test_body_bbox_covers_body_with_margin():
    vol = np.full((50, 60, 70), -1000.0, dtype=np.float32)  # воздух
    vol[10:20, 20:30, 30:40] = 100.0                        # «тело»
    bbox = body_bbox_zyx(vol, threshold_hu=-300, margin=10)
    z0, z1, y0, y1, x0, x1 = bbox
    assert (z0, z1) == (0, 30)      # 10-margin=0 (клип), 20+margin=30
    assert (y0, y1) == (10, 40)
    assert (x0, x1) == (20, 50)


def test_body_bbox_clips_margin_at_borders():
    vol = np.full((20, 20, 20), -1000.0, dtype=np.float32)
    vol[0:5, 0:5, 0:5] = 0.0        # тело в самом углу
    bbox = body_bbox_zyx(vol, threshold_hu=-300, margin=10)
    assert bbox == (0, 15, 0, 15, 0, 15)


def test_body_bbox_empty_volume_is_none():
    vol = np.full((10, 10, 10), -1000.0, dtype=np.float32)
    assert body_bbox_zyx(vol) is None


def test_body_bbox_threshold_exclusive():
    # воксель ровно на пороге НЕ считается телом (строгое сравнение)
    vol = np.full((10, 10, 10), -300.0, dtype=np.float32)
    assert body_bbox_zyx(vol, threshold_hu=-300) is None


# ---------- _spec_ok: парсер спеков версий в env_check ----------

@pytest.mark.parametrize("spec,installed,expected", [
    ("", "2025.2.0", True),                 # пустая спека — не блокируем
    ("==2025.2.0", "2025.2.0", True),
    ("==2025.2.0", "2025.3.0", False),
    ("!=2025.2.0", "2025.3.0", True),
    (">=1.20.0", "1.19.0", False),          # числовое сравнение, не строковое
    (">=1.20.0", "1.20.1", True),
    ("<2.8.0", "2.8.0", False),
    ("<=2.8.0", "2.8.0", True),
    (">3.9", "3.10", True),
    ("мусор без оператора", "1.0", True),   # непонятная спека — не блокируем
    (">=1.0, <2.0", "1.5", True),           # список через запятую: всё должно пройти
    (">=1.0, <2.0", "2.5", False),
])
def test_spec_ok_matrix(spec, installed, expected):
    assert _spec_ok(spec, installed) is expected


# ---------- normalize_patient_id: вход CLI ----------

@pytest.mark.parametrize("raw,expected", [
    ("7", "P007"),
    ("110", "P110"),
    ("P007", "P007"),
    ("p007", "P007"),
    ("  P110  ", "P110"),       # пробелы срезаются
    ("P12", None),              # нужно ровно 3 цифры
    ("abc", None),
    ("", None),
])
def test_normalize_patient_id(raw, expected):
    assert normalize_patient_id(raw) == expected


# ---------- config.LABELS: целостность медицинского словаря ----------

def test_labels_are_contiguous_1_to_31():
    values = sorted(config.LABELS.values())
    assert values == list(range(1, 32)), "нумерация меток должна быть 1..31 без дыр"


def test_labels_names_unique():
    # имя = ROI в RTSTRUCT; дубликат молча перезаписал бы орган
    assert len(set(config.LABELS)) == len(config.LABELS) == 31
