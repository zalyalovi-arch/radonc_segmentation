# ============================================================
# test_model_lifecycle.py — юнит-тесты жизненного цикла модели
# ------------------------------------------------------------
# Стейджинг чекпоинтов (import_checkpoint.locate_and_stage) и
# гейт качества (validate_model.gate_verdict) — шаги 0-1 и 3
# из kaggle/FINAL_STEPS.md. Без GUI и реальных весов.
# Запуск: python -m pytest tests (из корня проекта).
# ============================================================
import os
import zipfile

import pytest

from radonc_segmentation import config
from radonc_segmentation.quantization import import_checkpoint as ic
from radonc_segmentation.quantization import validate_model as vm


# ---------- locate_and_stage: раскладка чекпоинтов ----------

@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(config, "MODELS_ROOT", str(root))
    return root


def _make_plans(path):
    path.write_text('{"configurations": {"3d_fullres": {"patch_size": [56, 192, 192]}}}',
                    encoding="utf-8")


def test_stage_single_pth_with_nearby_plans(models_root, tmp_path):
    src = tmp_path / "download"
    src.mkdir()
    (src / "checkpoint_best.pth").write_bytes(b"fake")
    _make_plans(src / "plans.json")

    staged, folds = ic.locate_and_stage(str(src / "checkpoint_best.pth"), "distilled")

    assert folds == [(0, os.path.join(staged, "fold_0", "checkpoint_best.pth"))]
    assert os.path.isfile(os.path.join(staged, "fold_0", "checkpoint_best.pth"))
    assert os.path.isfile(os.path.join(staged, "plans.json"))
    assert staged.startswith(str(models_root / "_src" / "distilled"))


def test_stage_nested_kaggle_folder(models_root, tmp_path):
    # layout Kaggle Output: nnUNet_results\Dataset...\trainer\fold_N\checkpoint_*.pth
    trainer = tmp_path / "out" / "nnUNet_results" / "Dataset001_headandneck" / "trainer"
    for fold in (0, 1):
        d = trainer / f"fold_{fold}"
        d.mkdir(parents=True)
        (d / "checkpoint_final.pth").write_bytes(b"fake" * fold)
    _make_plans(trainer / "plans.json")

    staged, folds = ic.locate_and_stage(str(tmp_path / "out"), "distilled")

    assert [f for f, _p in folds] == [0, 1]
    assert os.path.isfile(os.path.join(staged, "plans.json"))
    for fold, ck in folds:
        assert os.path.isfile(ck)
        assert f"fold_{fold}" in ck


def test_stage_zip_output(models_root, tmp_path):
    trainer = tmp_path / "nnUNet_results" / "trainer"
    (trainer / "fold_0").mkdir(parents=True)
    (trainer / "fold_0" / "checkpoint_best.pth").write_bytes(b"fake")
    _make_plans(trainer / "plans.json")
    zpath = tmp_path / "output.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for root, _d, files in os.walk(tmp_path / "nnUNet_results"):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, tmp_path))

    staged, folds = ic.locate_and_stage(str(zpath), "distilled")

    assert [f for f, _p in folds] == [0]
    assert os.path.isfile(os.path.join(staged, "fold_0", "checkpoint_best.pth"))
    assert os.path.isfile(os.path.join(staged, "plans.json"))
    # временная распаковка удаляется после раскладки (экономия диска)
    assert not os.path.isdir(os.path.join(staged, "_raw"))


def test_stage_picks_one_checkpoint_per_fold(models_root, tmp_path):
    # в Output Kaggle в fold_0 лежат и final, и latest — нужен ОДИН,
    # приоритет best > final > latest (иначе экспорт перезаписывает сам себя)
    trainer = tmp_path / "out" / "trainer"
    d = trainer / "fold_0"
    d.mkdir(parents=True)
    (d / "checkpoint_final.pth").write_bytes(b"final")
    (d / "checkpoint_latest.pth").write_bytes(b"latest")
    _make_plans(trainer / "plans.json")

    staged, folds = ic.locate_and_stage(str(tmp_path / "out"), "distilled")

    assert folds == [(0, os.path.join(staged, "fold_0", "checkpoint_final.pth"))]
    assert not os.path.isfile(os.path.join(staged, "fold_0", "checkpoint_latest.pth"))


def test_stage_prefers_best_over_final(models_root, tmp_path):
    trainer = tmp_path / "out" / "trainer"
    d = trainer / "fold_0"
    d.mkdir(parents=True)
    (d / "checkpoint_best.pth").write_bytes(b"best")
    (d / "checkpoint_final.pth").write_bytes(b"final")
    _make_plans(trainer / "plans.json")

    staged, folds = ic.locate_and_stage(str(tmp_path / "out"), "distilled")

    assert folds == [(0, os.path.join(staged, "fold_0", "checkpoint_best.pth"))]


def test_stage_without_plans_fails_with_instruction(models_root, tmp_path):
    src = tmp_path / "download"
    src.mkdir()
    (src / "checkpoint_best.pth").write_bytes(b"fake")

    with pytest.raises(SystemExit, match="plans.json"):
        ic.locate_and_stage(str(src / "checkpoint_best.pth"), "distilled")


def test_stage_bad_source(models_root, tmp_path):
    with pytest.raises(SystemExit, match="Не найден источник"):
        ic.locate_and_stage(str(tmp_path / "нет_такого_файла.pth"), "distilled")


def test_import_name_rejects_slash():
    # имя импорта — папка, слэш недопустим (в отличие от имени квантизации)
    assert ic.NAME_RE.fullmatch("distilled_v2")
    assert not ic.NAME_RE.fullmatch("headandneck/distilled")


# ---------- gate_verdict: гейт качества из FINAL_STEPS ----------

def test_gate_passes_above_threshold():
    ok, lines = vm.gate_verdict(0.87, {"Lens_R": 0.9})
    assert ok
    assert any("0.8700" in ln for ln in lines)


def test_gate_fails_below_threshold():
    ok, lines = vm.gate_verdict(0.80, {})
    assert not ok
    assert any("НИЖЕ ПОРОГА" in ln for ln in lines)


def test_gate_no_mean_is_failure():
    ok, _lines = vm.gate_verdict(None, {})
    assert not ok


def test_gate_small_structures_within_drop():
    small = {s: 0.90 for s in vm.SMALL_STRUCTURES}
    ref = {s: 0.91 for s in vm.SMALL_STRUCTURES}   # падение 0.01 ≤ 0.03
    ok, lines = vm.gate_verdict(0.87, small, ref)
    assert ok
    assert any("дельта=-0.0100" in ln for ln in lines)


def test_gate_small_structures_excess_drop_fails():
    small = {s: 0.80 for s in vm.SMALL_STRUCTURES}
    ref = {s: 0.90 for s in vm.SMALL_STRUCTURES}   # падение 0.10 > 0.03
    ok, lines = vm.gate_verdict(0.87, small, ref)
    assert not ok
    assert any("ПАДЕНИЕ" in ln for ln in lines)


def test_gate_missing_organ_in_both_masks_skipped():
    # орган отсутствует в обеих масках — не влияет на гейт
    small = {"Lens_R": 0.9}
    ref = {"Lens_R": 0.9}
    ok, _lines = vm.gate_verdict(0.87, small, ref)
    assert ok


def test_gate_without_reference_reports_values_only():
    small = {s: 0.90 for s in vm.SMALL_STRUCTURES}
    ok, lines = vm.gate_verdict(0.87, small, None)
    assert ok
    assert any("эталон не задан" in ln for ln in lines)


# ---------- sanitize: имя модели в имени файла ----------

def test_sanitize_slash_and_spaces():
    assert vm.sanitize("headandneck/distilled_int8") == "headandneck_distilled_int8"
    assert vm.sanitize("a b/c.d") == "a_b_c.d"
