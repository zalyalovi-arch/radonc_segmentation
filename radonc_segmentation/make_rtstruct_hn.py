import os
import sys


import SimpleITK as sitk
import numpy as np
from rt_utils import RTStructBuilder
import colorsys

from radonc_segmentation import env_check

# предупреждаем о проблемах окружения ДО трейсбеков (см. env_check.py)
env_check.ensure_ready()

def load_ct_from_dicom(dicom_series_path):
    """Load CT DICOM series as SimpleITK image."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_series_path)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_series_path}")
    series_files = reader.GetGDCMSeriesFileNames(dicom_series_path, series_ids[0])
    reader.SetFileNames(series_files)
    ct_img = reader.Execute()
    return ct_img

def resample_seg_to_ct(seg_img, ct_img):
    """
    Resample segmentation image to match CT geometry (size, spacing, origin, direction).
    Uses nearest neighbor to preserve labels.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    seg_resampled = resampler.Execute(seg_img)
    return seg_resampled

def main():
    # ===== PATIENT ID FROM COMMAND LINE =====
    # Run this script as:
    # python make_rtstruct.py P001
    # python make_rtstruct.py P002
    #
    # Where "P001", "P002", ... are patient IDs and each patient has a folder
    # C:\nnunet\patients\P001
    # C:\nnunet\patients\P002
    #
    # containing:
    # - ct_dicom\ (CT DICOM series)
    # - seg_nnunet\ (nnU-Net output NIfTI, e.g. P001.nii.gz)

    if len(sys.argv) < 1 + 1:
        raise SystemExit("Usage: python make_rtstruct_hn.py P001 [SEG_FILENAME.nii.gz]\n"
                         "По умолчанию берётся seg_nnunet\\P001.nii.gz")
    patient_id = sys.argv[1]
    seg_filename = sys.argv[2] if len(sys.argv) > 2 else f"{patient_id}.nii.gz"
    # ========================================

    # ========= ПАПКА ПАЦИЕНТОВ (из settings.ini / config.py) =========
    from radonc_segmentation import config
    base_dir = config.patient_dir(patient_id)   # patients\<MODEL_NAME>\<ID>
    # =================================================

    # ========= PATHS FOR THIS PATIENT =========
    dicom_series_path = os.path.join(base_dir, "ct_dicom") # CT DICOM folder
    seg_nifti_path = os.path.join(base_dir, "seg_nnunet", seg_filename) # nnU-Net output NIfTI
    out_rtstruct_path = os.path.join(base_dir, "rtstruct.dcm") # output RTSTRUCT
    # =========================================

    # ========= EDIT LABEL MAP IF NEEDED =========
    # Integer labels in nnU-Net mask (check with np.unique)
    # единый источник имён/номеров структур — config.LABELS (как в GUI и метриках)
    label_map = config.LABELS
    # ============================================

    # Load CT
    ct_img = load_ct_from_dicom(dicom_series_path)

    # Load segmentation and resample to CT geometry
    seg_img = sitk.ReadImage(seg_nifti_path)
    seg_img_rs = resample_seg_to_ct(seg_img, ct_img)

    # seg_array has shape [z, y, x]
    seg_array = sitk.GetArrayFromImage(seg_img_rs)
    print("Seg array shape [z,y,x]:", seg_array.shape)

    # rt-utils expects [rows, cols, slices] = [y, x, z]
    seg_array_yxz = np.transpose(seg_array, (1, 2, 0)) # [y, x, z]
    print("Seg array_yxz shape [y,x,z]:", seg_array_yxz.shape)

    # Debug info
    #print("CT size (x,y,z):", ct_img.GetSize())
    #print("Seg size (x,y,z) after resample:", seg_img_rs.GetSize())

    # ===== OPTIONAL ORIENTATION FIXES =====
    # If RTSTRUCT still appears flipped/rotated, uncomment ONE of these blocks
    # and regenerate RTSTRUCT to see which matches best.
    #
    # NOTE: operations must be applied on seg_array_yxz and keep shape [y,x,z].
    # Example 1: left-right flip in X
    # seg_array_yxz = seg_array_yxz[:, ::-1, :]
    # Example 2: anterior-posterior flip in Y
    # seg_array_yxz = seg_array_yxz[::-1, :, :]
    # Example 3: 90° rotation in-plane (Y,X)
    # seg_array_yxz = np.rot90(seg_array_yxz, k=1, axes=(0, 1))
    # ======================================

    # Create RTSTRUCT linked to CT
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_series_path)

    # Add ROIs with automatically generated unique colors
    for name, lbl in label_map.items():
        roi_mask = (seg_array_yxz == lbl) # [y, x, z]
        if not np.any(roi_mask):
            print(f"Warning: label {lbl} for '{name}' is empty, skipping.")
            continue

        # Generate unique color for each structure (HSV spaced for distinction)
        num_structures = len(label_map)
        structure_idx = list(label_map.values()).index(lbl)  # 0-based index for this label
        hue = structure_idx / num_structures
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)  # Saturation 0.8, Value 0.9 for visibility
        color = [int(255 * c) for c in rgb]

        print(f"Adding ROI '{name}' with label {lbl} and color {color}")
        rtstruct.add_roi(
            mask=roi_mask,
            name=name,
            color=color,
        )

    # Save RTSTRUCT
    os.makedirs(os.path.dirname(out_rtstruct_path), exist_ok=True)
    rtstruct.save(out_rtstruct_path)
    print(f"{patient_id}: Saved RTSTRUCT: {out_rtstruct_path}")

if __name__ == "__main__":
    main()
