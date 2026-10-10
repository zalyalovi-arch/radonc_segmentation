import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import SimpleITK as sitk

import env_check

# предупреждаем о проблемах окружения ДО трейсбеков (см. env_check.py)
env_check.ensure_ready()


def select_series(dicom_folder):
    """
    Выбирает DICOM-серию для конвертации. Если в папке несколько серий
    (напр. скаут/локалайзер + CT), предпочитает Modality=CT и наибольшую
    по числу файлов. Возвращает список имён файлов выбранной серии.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {dicom_folder}")

    if len(series_ids) == 1:
        return reader.GetGDCMSeriesFileNames(dicom_folder, series_ids[0])

    print(f"ВНИМАНИЕ: в папке {len(series_ids)} серий — выбираю CT с максимальным числом срезов:")
    best_key = None
    best_files = None
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(dicom_folder, sid)
        modality = "?"
        try:
            import pydicom
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            modality = str(getattr(ds, "Modality", "?"))
        except Exception:
            pass
        print(f"  серия {sid[:16]}...: {len(files)} файлов, Modality={modality}")
        key = (1 if modality == "CT" else 0, len(files))
        if best_key is None or key > best_key:
            best_key, best_files = key, files
    print(f"Выбрана серия из {len(best_files)} файлов")
    return best_files


def dicom_to_nifti(dicom_folder, output_nifti):
    """
    Convert a DICOM CT series (dicom_folder) to a single NIfTI file (output_nifti).
    """
    series_files = select_series(dicom_folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_files)

    # Load the DICOM series as a 3D image
    image = reader.Execute()

    # Save as NIfTI (.nii.gz)
    sitk.WriteImage(image, output_nifti)
    print(f"Saved NIfTI: {output_nifti}")
    print(f"Размер (x,y,z): {image.GetSize()}, spacing: {tuple(round(s, 3) for s in image.GetSpacing())}")


if __name__ == "__main__":
    # ========== HOW TO RUN ==========
    #
    # Run this script as:
    #   python dicom_to_nifti.py P001
    #
    # Where "P001", "P002", ... are patient IDs and each patient has a folder
    #   C:\nnunet\patients\P001
    #
    # containing a "ct_dicom" subfolder.

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

    if len(sys.argv) < 1 + 1:
        raise SystemExit("Usage: python dicom_to_nifti.py P001")
    patient_id = sys.argv[1]

    base_root = config.PATIENTS_ROOT
    base_dir = os.path.join(base_root, patient_id)

    # Subfolder that contains CT DICOM files inside base_dir
    dicom_folder = os.path.join(base_dir, "ct_dicom")

    # NIfTI for nnU-Net input: {CASE}_0000.nii.gz
    out_folder = os.path.join(base_dir, "nnunet_input")
    os.makedirs(out_folder, exist_ok=True)
    seg_folder = os.path.join(base_dir, "seg_nnunet")
    os.makedirs(seg_folder, exist_ok=True)
    out_nifti = os.path.join(out_folder, f"{patient_id}_0000.nii.gz")

    dicom_to_nifti(dicom_folder, out_nifti)
