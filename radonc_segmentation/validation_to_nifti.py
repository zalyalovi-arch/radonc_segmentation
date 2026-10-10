# ============================================================
# validation_to_nifti.py — конвертация валидационного кейса в NIfTI
# ------------------------------------------------------------
# На вход: папка с КТ-серией (DICOM) + RTSTRUCT с GT (RTSTRUCT*.dcm /
# RS*.dcm). На выход: пара для benchmark_dice.py / compare_dice.py:
#   <out>\imagesTr\<case>_0000.nii.gz  — КТ
#   <out>\labelsTr\<case>.nii.gz       — GT-маска (метки из config.LABELS)
#
# Запуск:
#   python validation_to_nifti.py "C:\nnunet reboot\validation" val001
# По умолчанию out = <проект>\validation (imagesTr/labelsTr внутри).
# ============================================================
import os
import sys

import numpy as np
import SimpleITK as sitk

from radonc_segmentation import config
from radonc_segmentation import env_check

env_check.ensure_ready()


def find_rtstruct(folder):
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".dcm") and f.lower().split(".")[0].upper() in ("RS", "RTSTRUCT"):
            return os.path.join(folder, f)
    raise SystemExit(f"RTSTRUCT (RS*.dcm / RTSTRUCT*.dcm) не найден в {folder}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python validation_to_nifti.py <папка с DICOM+RTSTRUCT> [case_name] [out_root]")
    dicom_folder = sys.argv[1]
    case = sys.argv[2] if len(sys.argv) > 2 else "val001"
    out_root = sys.argv[3] if len(sys.argv) > 3 else config.validation_cases_dir()

    # новая per-case структура: validation\<модель>\<кейс>\{nnunet_input,gt}
    case_dir = os.path.join(config.validation_cases_dir(), case)
    images_dir = os.path.join(case_dir, "nnunet_input")
    labels_dir = os.path.join(case_dir, "gt")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 1) КТ-серия -> sitk image -> NIfTI
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        raise SystemExit(f"КТ-серия не найдена в {dicom_folder}")
    files = reader.GetGDCMSeriesFileNames(dicom_folder, series_ids[0])
    print(f"КТ: {len(files)} срезов (серия {series_ids[0][:20]}...)")
    reader.SetFileNames(files)
    ct = reader.Execute()
    img_path = os.path.join(images_dir, f"{case}_0000.nii.gz")
    sitk.WriteImage(ct, img_path)
    print(f"КТ сохранён: {img_path}  размер {ct.GetSize()}, spacing {ct.GetSpacing()}")

    # 2) RTSTRUCT -> маска GT (метки из config.LABELS)
    from rt_utils import RTStructBuilder
    rt_path = find_rtstruct(dicom_folder)
    print(f"RTSTRUCT: {rt_path}")
    rt = RTStructBuilder.create_from(dicom_series_path=dicom_folder, rt_struct_path=rt_path)
    roi_names = rt.get_roi_names()  # rt_utils 1.2.7 -> список имён (str)

    gt = None  # (z, y, x) uint8
    matched, unmatched = [], []
    for name in roi_names:
        if name not in config.LABELS:
            unmatched.append(name)
            continue
        lbl = config.LABELS[name]
        mask_yxz = rt.get_roi_mask_by_name(name)  # (y, x, z) bool
        m_zyx = np.transpose(mask_yxz, (2, 0, 1))
        if gt is None:
            gt = np.zeros(m_zyx.shape, dtype=np.uint8)
        gt[m_zyx] = lbl  # больший номер метки перекрывает меньший (мелкие структуры важнее)
        matched.append((name, lbl, int(m_zyx.sum())))

    if gt is None:
        raise SystemExit(f"Ни один ROI не совпал с config.LABELS.\nROI в RTSTRUCT: {roi_names}")
    print(f"Совпало ROI: {len(matched)} из {len(roi_names)}")
    for name, lbl, n in matched:
        print(f"  {lbl:>2} {name:<20} {n:>9} вокселей")
    if unmatched:
        print(f"Без метки (пропущены): {unmatched}")

    gt_img = sitk.GetImageFromArray(gt)
    gt_img.CopyInformation(ct)  # та же геометрия, что у КТ
    gt_path = os.path.join(labels_dir, f"{case}.nii.gz")
    sitk.WriteImage(gt_img, gt_path)
    print(f"GT сохранён: {gt_path}")

    # проверка пересечений в GT (Lens внутри Eye и т.п.)
    vals, counts = np.unique(gt, return_counts=True)
    print("Гистограмма меток:", dict(zip(vals.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
