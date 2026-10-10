# C:\nnunet\scripts\export_rtstruct.py

import os
import sys
import shutil

from radonc_segmentation import config


def main():
    # ===== HOW TO RUN =====
    # python export_rtstruct.py P002 "D:\path\to\original\dicom_folder"
    #
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python export_rtstruct.py PXXX <ORIGINAL_DICOM_FOLDER>\n"
            "Example: python export_rtstruct.py P002 \"D:\\CTs\\Patient123\""
        )

    patient_id = sys.argv[1]              # e.g. "P002"
    original_dicom_folder = sys.argv[2]   # where you originally imported from

    # 1) Where nnU-Net patient folders live (папка patients из settings.ini)
    patients_root = config.PATIENTS_ROOT
    base_dir = os.path.join(patients_root, patient_id)

    # 2) Source RTSTRUCT file inside nnU-Net patient folder
    src_rtstruct = os.path.join(base_dir, "rtstruct.dcm")

    if not os.path.isfile(src_rtstruct):
        raise FileNotFoundError(f"RTSTRUCT not found for {patient_id}: {src_rtstruct}")

    # 3) Destination folder: <original_dicom_folder>\rtstruct
    dst_rtstruct_folder = os.path.join(original_dicom_folder, "rtstruct")
    os.makedirs(dst_rtstruct_folder, exist_ok=True)

    # 4) Copy RTSTRUCT there, keep same name
    dst_rtstruct = os.path.join(dst_rtstruct_folder, os.path.basename(src_rtstruct))

    print(f"Copying {src_rtstruct}")
    print(f"    -> {dst_rtstruct}")
    shutil.copy2(src_rtstruct, dst_rtstruct)

    print(f"{patient_id}: RTSTRUCT exported to {dst_rtstruct}")


if __name__ == "__main__":
    main()