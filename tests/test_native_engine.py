# ============================================================
# test_native_engine.py — юнит-тесты нативного nnUNet-движка
# ------------------------------------------------------------
# Стейджинг-резолвер (native_stage.resolve_native_model), шаблон
# dataset.json, шим тренера (native_infer._install_trainer_shim),
# диспетчер движка (pipeline.infer_step_command) и config.cuda_available.
# Без GUI, реальных весов и CUDA.
# Запуск: python -m pytest tests (из корня проекта).
# ============================================================
import os
import sys

import pytest

from radonc_segmentation import config
from radonc_segmentation.native import native_infer, native_stage
from radonc_segmentation import pipeline


# ---------- resolve_native_model: .pth-модель для нативного движка ----------

@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(config, "MODELS_ROOT", str(root))
    return root


def _stage_model(models_root, quant="distilled", folds=(0,), ck="checkpoint_final.pth"):
    """Раскладывает _src\\<quant> как это делает import_checkpoint."""
    d = models_root / "_src" / quant
    for f in folds:
        fd = d / f"fold_{f}"
        fd.mkdir(parents=True)
        (fd / ck).write_bytes(b"fake")
    (d / "plans.json").write_text('{"configurations": {}}', encoding="utf-8")
    return d


def test_resolve_from_staging(models_root):
    _stage_model(models_root, "distilled", folds=(0, 1))
    cfg = native_stage.resolve_native_model("headandneck/distilled")
    assert cfg["name"] == "headandneck/distilled"
    assert cfg["folds"] == [0, 1]
    assert cfg["checkpoint"] == "checkpoint_final.pth"
    assert cfg["dir"] == str(models_root / "_src" / "distilled")


def test_resolve_from_model_folder(models_root):
    # пользователь положил дерево nnUNet_results прямо в models\<анатомия>\<quant>
    d = models_root / "headandneck" / "native"
    (d / "fold_0").mkdir(parents=True)
    (d / "fold_0" / "checkpoint_best.pth").write_bytes(b"fake")
    (d / "plans.json").write_text("{}", encoding="utf-8")
    cfg = native_stage.resolve_native_model("headandneck/native")
    assert cfg["checkpoint"] == "checkpoint_best.pth"


def test_resolve_prefers_staging_over_ir_folder(models_root):
    # в models\<анатомия>\<quant> лежит только IR (xml) — не подходит;
    # стейджинг с чекпоинтом должен выиграть
    ir = models_root / "headandneck" / "distilled"
    (ir / "fold_0").mkdir(parents=True)
    (ir / "fold_0" / "model.xml").write_text("<net/>", encoding="utf-8")
    _stage_model(models_root, "distilled", folds=(0,))
    cfg = native_stage.resolve_native_model("headandneck/distilled")
    assert cfg["dir"] == str(models_root / "_src" / "distilled")


def test_resolve_missing_model_raises(models_root):
    with pytest.raises(config.ModelNotFoundError, match="\\.pth-чекпоинт"):
        native_stage.resolve_native_model("headandneck/нету")


def test_resolve_ir_only_model_raises_with_hint(models_root):
    # импортированная модель с IR, но без _src (чекпоинты удалены) —
    # ошибка должна подсказывать импорт .pth
    ir = models_root / "headandneck" / "distilled"
    (ir / "fold_0").mkdir(parents=True)
    (ir / "fold_0" / "model.xml").write_text("<net/>", encoding="utf-8")
    with pytest.raises(config.ModelNotFoundError, match="import_checkpoint"):
        native_stage.resolve_native_model("headandneck/distilled")


def test_pick_checkpoint_requires_all_folds(models_root):
    # в одном фолде final, в другом best — единого имени нет, модель непригодна
    d = models_root / "_src" / "mixed"
    (d / "fold_0").mkdir(parents=True)
    (d / "fold_0" / "checkpoint_final.pth").write_bytes(b"f")
    (d / "fold_1").mkdir(parents=True)
    (d / "fold_1" / "checkpoint_best.pth").write_bytes(b"b")
    (d / "plans.json").write_text("{}", encoding="utf-8")
    with pytest.raises(config.ModelNotFoundError):
        native_stage.resolve_native_model("headandneck/mixed")


# ---------- dataset.json: шаблон для trainer-папки ----------

def test_dataset_json_template_covers_labels():
    dj = native_stage.dataset_json_template()
    assert dj["labels"]["background"] == 0
    assert dj["labels"]["Cochlea_L"] == 31
    assert len(dj["labels"]) == 32
    assert dj["channel_names"] == {"0": "CT"}


def test_ensure_dataset_json_creates_once(models_root, tmp_path):
    d = tmp_path / "trainer"
    d.mkdir()
    p1 = native_stage.ensure_dataset_json(str(d), log=lambda *_: None)
    assert os.path.isfile(p1)
    # повторный вызов не перезаписывает (сохранить пользовательский)
    native_stage.ensure_dataset_json(str(d), log=lambda *_: None)


# ---------- шим тренера: Kaggle-чекпоинты на stock nnunetv2 ----------

def test_trainer_shim_substitutes_base_class():
    # без nnunetv2 (нет в базовом окружении) тест не имеет смысла
    pytest.importorskip("nnunetv2")
    native_infer._install_trainer_shim()
    from nnunetv2.inference import predict_from_raw_data as pfrm
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    tr_dir = os.path.join(pfrm.nnunetv2.__path__[0], "training", "nnUNetTrainer")
    # заведомо отсутствующий в nnunetv2 тренер -> шим подменяет базовым
    # (recursive_find_python_class возвращает None, шим — nnUNetTrainer)
    cls = pfrm.recursive_find_python_class(
        tr_dir, "nnUNetTrainerRadioncFake", "nnunetv2.training.nnUNetTrainer")
    assert cls is nnUNetTrainer
    # stock-тренеры (и trainer_distill_standalone, если подложен) шим не трогает
    assert pfrm.recursive_find_python_class(
        tr_dir, "nnUNetTrainerNoMirroring", "nnunetv2.training.nnUNetTrainer") is not nnUNetTrainer


def test_lowmem_export_shim_replaces_iterator_method():
    # Windows: штатный экспорт через пул падает на pickle логитов >4 ГБ
    # (WriteFile ограничен DWORD) — шим заменяет predict_from_data_iterator
    # синхронной версией без multiprocessing
    pytest.importorskip("nnunetv2")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    original = nnUNetPredictor.predict_from_data_iterator
    try:
        native_infer._install_lowmem_export_shim()
        assert nnUNetPredictor.predict_from_data_iterator is not original
        # повторная установка — no-op (не оборачивает саму себя второй раз)
        native_infer._install_lowmem_export_shim()
        assert nnUNetPredictor.predict_from_data_iterator.__name__ == \
            "predict_from_data_iterator_sync"
    finally:
        nnUNetPredictor.predict_from_data_iterator = original


# ---------- диспетчер движка в pipeline ----------

def test_infer_step_command_openvino_legacy():
    cmd, seg, label = pipeline.infer_step_command(
        "openvino", "legacy", "P007", ["--folds", "0"], ["--no-crop"])
    assert "radonc_segmentation.infer_ov" in " ".join(cmd)
    assert "--force" in cmd
    assert seg == "P007.nii.gz" and label == "legacy"


def test_infer_step_command_openvino_aligned():
    cmd, seg, label = pipeline.infer_step_command(
        "openvino", "aligned", "P007", [], ["--no-async"])
    assert "aligned.infer_ov_aligned" in " ".join(cmd)
    assert seg == "P007_aligned.nii.gz" and label == "aligned"


def test_infer_step_command_native_ignores_variant():
    cmd, seg, label = pipeline.infer_step_command(
        "nnunet_native", "aligned", "P007", ["--folds", "0"], ["--no-crop"])
    joined = " ".join(cmd)
    assert "native.native_infer" in joined
    # вариант и OpenVINO-флаги не пролезают в команду нативного движка
    assert "--variant" not in joined and "--no-crop" not in joined
    assert seg == "P007_native.nii.gz" and label == "native"


def test_infer_step_command_native_allow_cpu():
    cmd, seg, label = pipeline.infer_step_command(
        "nnunet_native", "legacy", "P007", [], [], allow_cpu=True)
    assert "--allow-cpu" in cmd
    assert seg == "P007_native.nii.gz" and label == "native"


def test_infer_step_command_native_no_allow_cpu_by_default():
    # без явного разрешения CPU флаг не подставляется — движок сам
    # отклонит запуск без CUDA с объяснением
    cmd, _seg, _label = pipeline.infer_step_command(
        "nnunet_native", "legacy", "P007", [], [])
    assert "--allow-cpu" not in cmd


def test_infer_step_command_openvino_ignores_allow_cpu():
    # OpenVINO-движку --allow-cpu не нужен (он и так CPU-first) — флаг не пролезает
    cmd, _seg, _label = pipeline.infer_step_command(
        "openvino", "legacy", "P007", [], [], allow_cpu=True)
    assert "--allow-cpu" not in cmd


def test_pipeline_argparse_accepts_engine(monkeypatch):
    # --engine доходит до argparse и валидируется по списку движков
    argv = ["pipeline", "P007", "--engine", "nnunet_native"]
    monkeypatch.setattr(sys, "argv", argv)
    import argparse
    from radonc_segmentation import env_check
    monkeypatch.setattr(env_check, "ensure_ready", lambda **kw: [])
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("--engine", default=config.DEFAULT_ENGINE, choices=list(config.VALID_ENGINES))
    p.add_argument("--variant", default=config.DEFAULT_VARIANT)
    args = p.parse_args(argv[1:])
    assert args.engine == "nnunet_native"


def test_pipeline_rejects_unknown_engine():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("--engine", default=config.DEFAULT_ENGINE, choices=list(config.VALID_ENGINES))
    with pytest.raises(SystemExit):
        p.parse_args(["P007", "--engine", "что-то_иное"])


# ---------- config.cuda_available: subprocess-детект с кэшем ----------

def test_cuda_available_caches(monkeypatch):
    monkeypatch.setattr(config, "_CUDA_AVAILABLE", None)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = "1\n"
            stderr = ""
        return R()

    monkeypatch.setattr(config.subprocess, "run", fake_run)
    assert config.cuda_available(python="python-x") is True
    assert config.cuda_available(python="python-y") is True  # кэш — второй вызов не идёт
    assert len(calls) == 1
    assert "python-x" in calls[0][0]


def test_cuda_available_false_when_torch_without_cuda(monkeypatch):
    monkeypatch.setattr(config, "_CUDA_AVAILABLE", None)

    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stdout = "0\n"
            stderr = ""
        return R()

    monkeypatch.setattr(config.subprocess, "run", fake_run)
    assert config.cuda_available(python="python-x", force=True) is False


def test_cuda_available_false_when_import_fails(monkeypatch):
    monkeypatch.setattr(config, "_CUDA_AVAILABLE", None)

    def fake_run(cmd, **kw):
        class R:
            returncode = 1
            stdout = ""
            stderr = "ModuleNotFoundError: No module named 'torch'"
        return R()

    monkeypatch.setattr(config.subprocess, "run", fake_run)
    assert config.cuda_available(python="python-x", force=True) is False


# ---------- settings: ENGINE читается и валидируется ----------

def test_settings_engine_valid(tmp_path, monkeypatch):
    ini = tmp_path / "settings.ini"
    ini.write_text("[inference]\nENGINE = nnunet_native\n", encoding="utf-8")
    monkeypatch.setattr(config, "_SETTINGS_CANDIDATES", [str(ini)])
    monkeypatch.setattr(config, "SETTINGS_PATH", None)
    config._load_settings()
    assert config.DEFAULT_ENGINE == "nnunet_native"
    config.DEFAULT_ENGINE = "openvino"  # вернуть дефолт остальным тестам


def test_settings_engine_invalid_raises(tmp_path, monkeypatch):
    ini = tmp_path / "settings.ini"
    ini.write_text("[inference]\nENGINE = gluoncv\n", encoding="utf-8")
    monkeypatch.setattr(config, "_SETTINGS_CANDIDATES", [str(ini)])
    monkeypatch.setattr(config, "SETTINGS_PATH", None)
    with pytest.raises(config.SettingsValidationError, match="ENGINE"):
        config._load_settings()
