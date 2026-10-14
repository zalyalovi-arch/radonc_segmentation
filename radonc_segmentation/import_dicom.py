# C:\nnunet reboot\scripts\import_dicom.py
# Импорт DICOM: копирует серию в patients\PXXX\ct_dicom, создаёт подпапки,
# записывает patient.json (исходный путь — для авто-экспорта RTSTRUCT).

import json
import os
import shutil
import sys
import time

from radonc_segmentation import config


def get_next_patient_id(patients_root):
    """
    Find the next free patient ID in the form P001, P002, ...
    Учитывает пациентов во ВСЕХ анатомических подпапках (patients\<модель>\PXXX),
    чтобы ID не пересекались между моделями.
    """
    existing = []
    roots = [patients_root]
    if os.path.isdir(patients_root):
        for sub in os.listdir(patients_root):
            sub_path = os.path.join(patients_root, sub)
            if os.path.isdir(sub_path):
                roots.append(sub_path)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.startswith("P") and name[1:].isdigit():
                existing.append(int(name[1:]))
    if not existing:
        return "P001"
    next_num = max(existing) + 1
    return f"P{next_num:03d}"


def find_first_dcm(src_dicom_folder):
    for root, _dirs, files in os.walk(src_dicom_folder):
        for f in files:
            if f.lower().endswith(".dcm"):
                return os.path.join(root, f)
    return None


def count_dcm(src_dicom_folder):
    n = 0
    for _root, _dirs, files in os.walk(src_dicom_folder):
        n += sum(1 for f in files if f.lower().endswith(".dcm"))
    return n


def check_modality(dcm_path):
    """Мягкая проверка, что это CT (предупреждение, не ошибка)."""
    try:
        import pydicom
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        modality = str(getattr(ds, "Modality", "?"))
        if modality != "CT":
            print(f"ВНИМАНИЕ: Modality = {modality} (ожидался CT). "
                  f"Продолжаю, но проверьте, ту ли папку вы импортируете.")
        else:
            print("Modality: CT (OK)")
    except Exception as e:
        print(f"(не удалось прочитать DICOM-заголовок: {e})")


def copy_dicom_folder(src_dicom_folder, dst_dicom_folder):
    """Copy all files from src_dicom_folder into dst_dicom_folder."""
    if not os.path.isdir(src_dicom_folder):
        raise RuntimeError(f"Source DICOM folder does not exist: {src_dicom_folder}")
    shutil.copytree(src_dicom_folder, dst_dicom_folder, dirs_exist_ok=True)


def import_dicom(src_dicom_folder):
    """
    High-level function to import a DICOM folder, create a new patient ID,
    copy DICOMs and create subfolders.
    Returns the new patient_id and base_dir.
    Папка пациента — config.patient_dir(): patients\<MODEL_NAME>\PXXX
    (та же, которую ищет pipeline/infer — единый стандарт путей).
    """
    # 1) Папка пациентов текущей анатомической модели
    patients_root = config.patients_dir()
    os.makedirs(patients_root, exist_ok=True)

    # 2) Валидация источника ДО копирования
    if not os.path.isdir(src_dicom_folder):
        raise RuntimeError(f"Папка не найдена: {src_dicom_folder}")
    n_dcm = count_dcm(src_dicom_folder)
    if n_dcm == 0:
        raise RuntimeError(f"В папке нет .dcm файлов: {src_dicom_folder}")
    first_dcm = find_first_dcm(src_dicom_folder)
    check_modality(first_dcm)
    print(f"DICOM файлов: {n_dcm}")

    # 3) Determine next available patient ID (P001, P002, ...)
    patient_id = get_next_patient_id(patients_root)
    base_dir = os.path.join(patients_root, patient_id)

    # 4) Define target subfolders
    ct_dicom_dir   = os.path.join(base_dir, "ct_dicom")
    nnunet_input   = os.path.join(base_dir, "nnunet_input")
    seg_nnunet     = os.path.join(base_dir, "seg_nnunet")

    print(f"Creating new patient: {patient_id}")
    print(f"Base dir: {base_dir}")
    os.makedirs(base_dir, exist_ok=True)

    # Copy DICOM
    print(f"Copying DICOM from {src_dicom_folder} to {ct_dicom_dir} ...")
    copy_dicom_folder(src_dicom_folder, ct_dicom_dir)

    # Create needed subfolders
    os.makedirs(nnunet_input, exist_ok=True)
    os.makedirs(seg_nnunet, exist_ok=True)

    # 5) patient.json: исходная папка (для авто-экспорта RTSTRUCT) + метаданные
    with open(os.path.join(base_dir, "patient.json"), "w", encoding="utf-8") as f:
        json.dump({
            "patient_id": patient_id,
            "source_path": os.path.abspath(src_dicom_folder),
            "imported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dicom_files": n_dcm,
        }, f, ensure_ascii=False, indent=2)

    print(f"Import finished for {patient_id}.")
    print(f"CT DICOM:     {ct_dicom_dir}")
    print(f"nnUNet input: {nnunet_input}")
    print(f"seg_nnunet:   {seg_nnunet}")
    print(f"Исходная папка запомнена (patient.json): {os.path.abspath(src_dicom_folder)}")

    return patient_id, base_dir


def main():
    # ===== HOW TO RUN FROM COMMAND LINE =====
    # python -m radonc_segmentation.import_dicom "D:\some\path\to\ct_dicom"
    # or a network path like:
    # python -m radonc_segmentation.import_dicom "\\SERVER\share\Patient123\CT"
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python import_dicom.py <SOURCE_DICOM_FOLDER>")
    src_dicom_folder = sys.argv[1]
    import_dicom(src_dicom_folder)


if __name__ == "__main__":
    main()
