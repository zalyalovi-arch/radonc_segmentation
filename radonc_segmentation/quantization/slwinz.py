import numpy as np
from openvino import Core
import nibabel as nib
import os
import gc
from scipy.ndimage import gaussian_filter
import openvino as ov
import time

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_IMAGE_PATH = r"D:\nnunet\patients\P001\nnunet_input\original\P001_0000.nii.gz"
MODEL_PATH = r"D:\openvino\openvino_model_fixed\model.xml"
OUTPUT_DIR = r"D:\nnunet\results"
CACHE_PATH = r"D:\openvino\openvino_model_fixed\cache"
OUTPUT_FILENAME = "P001_prediction_sw_optimized.nii.gz"

# Temporary directory for memmap files
TEMP_DIR = r"D:\nnunet\temp_buffer"

# Patch size expected by the model (Z, Y, X)
PATCH_SIZE = (56, 192, 192) 
STEP_SIZE_RATIO = 0.5 
DEVICE = "GPU.0" 

# ============================================================
# DEVICE OPTIMIZATION CONFIGURATION
# ============================================================
if DEVICE == "CPU":
    COMPILE_CONFIG = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "4"} 
elif DEVICE == "GPU.0":
    COMPILE_CONFIG = {"PERFORMANCE_HINT": "LATENCY"}
else:
    raise ValueError("DEVICE must be 'CPU' or 'GPU.0'")
# ============================================================

def compute_gaussian_map(patch_size, sigma_scale=1/8):
    """Creates a Gaussian importance map for blending."""
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    return gaussian_map.astype(np.float32)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(CACHE_PATH, exist_ok=True)

    # 1. Load Image
    print(f"📂 Loading image: {INPUT_IMAGE_PATH}")
    nii = nib.load(INPUT_IMAGE_PATH)
    # Load as float64 (default) or float32. Keeping float64 as in your snippet.
    data = nii.get_fdata(dtype=np.float32) 
    affine = nii.affine
    original_shape = data.shape
    
    print(f"   Original Shape (X, Y, Z): {original_shape}")
    print(f"   Data type: {data.dtype}")

    # 2. Global Preprocessing
    print("🔧 Applying global Z-score normalization...")
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        data = (data - mean) / std
    else:
        data = data - mean

    # 3. Transpose for Model (X,Y,Z -> Z,Y,X)
    print("🔄 Transposing volume to (Z, Y, X)...")
    data_transposed = np.transpose(data, (2, 0, 1))
    transposed_shape = data_transposed.shape
    print(f"   Transposed Shape (Z, Y, X): {transposed_shape}")

    # Clear original large data from RAM
    del data
    gc.collect()

    # 4. Padding
    print("🔲 Padding volume...")
    pad_needed = []
    for i in range(3):
        if transposed_shape[i] < PATCH_SIZE[i]:
            pad = PATCH_SIZE[i] - transposed_shape[i]
            pad_needed.append((pad // 2, pad - pad // 2))
        else:
            pad_needed.append((0, 0))
    
    data_padded = np.pad(data_transposed, pad_needed, mode='constant', constant_values=-1000)
    print(f"   Padded Shape (Z, Y, X): {data_padded.shape}")

    # Clear transposed data
    del data_transposed
    gc.collect()

    # 5. Load Model
    print(f"⚙️  Loading OpenVINO model for {DEVICE}...")
    print(f"   Optimization Config: {COMPILE_CONFIG}")
    
    core = Core()
    model = core.read_model(MODEL_PATH)
    COMPILE_CONFIG["CACHE_DIR"] = CACHE_PATH
    compiled_model = core.compile_model(model, DEVICE, config=COMPILE_CONFIG)

    # OPTIMIZATION: Create InferRequest ONCE
    infer_request = compiled_model.create_infer_request()

    # ============================================================
    # 6. SETUP MEMMAP AND STEPS (The missing part)
    # ============================================================
    num_classes = 32
    gaussian_map = compute_gaussian_map(PATCH_SIZE)
    
    probs_path = os.path.join(TEMP_DIR, "probs_buffer.dat")
    counts_path = os.path.join(TEMP_DIR, "counts_buffer.dat")

    # Delete old temp files if exist
    if os.path.exists(probs_path): os.remove(probs_path)
    if os.path.exists(counts_path): os.remove(counts_path)

    print(f"💾 Creating Memory-Mapped arrays on disk...")
    aggregated_probs = np.memmap(probs_path, dtype=np.float32, mode='w+', 
                                  shape=(num_classes,) + data_padded.shape)
    count_map = np.memmap(counts_path, dtype=np.float32, mode='w+', 
                          shape=data_padded.shape)

    # Calculate steps for sliding window
    step_sizes = [int(p * STEP_SIZE_RATIO) for p in PATCH_SIZE]
    
    steps_list = []
    for i in range(3):
        steps = list(range(0, data_padded.shape[i] - PATCH_SIZE[i] + 1, step_sizes[i]))
        # Ensure the very last patch is included if there's a remainder
        if not steps or steps[-1] + PATCH_SIZE[i] < data_padded.shape[i]:
            steps.append(data_padded.shape[i] - PATCH_SIZE[i])
        steps_list.append(steps)
        
    print(f"🚀 Starting Sliding Window Inference...")
    print(f"   Total patches: {len(steps_list[0]) * len(steps_list[1]) * len(steps_list[2])}")
    # ============================================================

    # 7. Sliding Window Loop
    total_patches = 0
    for z in steps_list[0]:
        for y in steps_list[1]:
            for x in steps_list[2]:
                
                # Extract patch
                patch = data_padded[z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]]
                
                # Prepare input
                input_tensor = patch[np.newaxis, np.newaxis, ...]
                
                # OPTIMIZATION: Use ov.Tensor (Zero-Copy)
                ov_tensor = ov.Tensor(array=input_tensor)
                
                # ============================================================
                # TIMER 1: INFERENCE (GPU/CPU Work)
                # ============================================================
                start_inference = time.perf_counter()
                
                # Set input and run inference
                infer_request.set_input_tensor(ov_tensor)
                infer_request.start_async()
                infer_request.wait()
                
                end_inference = time.perf_counter()
                # ============================================================
                
                # Get output
                prediction = infer_request.get_output_tensor().data[0]
                
                # Weighted Prediction
                weighted_pred = prediction.astype(np.float32) * gaussian_map
                
                # ============================================================
                # TIMER 2: WRITE (Math + Disk I/O)
                # ============================================================
                start_write = time.perf_counter()
                
                # Aggregation (This triggers the write to the memmap/disk)
                aggregated_probs[:, z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]] += weighted_pred
                count_map[z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]] += gaussian_map
                
                end_write = time.perf_counter()
                # ============================================================

                # Calculate times in milliseconds
                t_inf = (end_inference - start_inference) * 1000
                t_write = (end_write - start_write) * 1000
                t_total = t_inf + t_write

                total_patches += 1
                
                # Print detailed timing
                print(f"Patch {total_patches:03d} | Inf: {t_inf:6.2f}ms | Write: {t_write:6.2f}ms | Total: {t_total:6.2f}ms")

    # 8. Finalize
    print(f"✅ Processed {total_patches} patches.")
    print("🔗 Merging patches...")
    aggregated_probs /= np.maximum(count_map[np.newaxis, ...], 1e-8)
    
    print("✂️  Calculating Argmax...")
    segmentation_map = np.argmax(aggregated_probs, axis=0)

    # Close and delete memmaps
    del aggregated_probs, count_map
    if os.path.exists(probs_path): os.remove(probs_path)
    if os.path.exists(counts_path): os.remove(counts_path)
    gc.collect()

    # 9. Remove Padding and Transpose Back
    z_slice = slice(pad_needed[0][0], segmentation_map.shape[0] - pad_needed[0][1])
    y_slice = slice(pad_needed[1][0], segmentation_map.shape[1] - pad_needed[1][1])
    x_slice = slice(pad_needed[2][0], segmentation_map.shape[2] - pad_needed[2][1])
    
    segmentation_cropped = segmentation_map[z_slice, y_slice, x_slice]
    final_output = np.transpose(segmentation_cropped, (1, 2, 0))

    print(f"✅ Final Output Shape (X, Y, Z): {final_output.shape}")

    # 10. Save
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    result_nii = nib.Nifti1Image(final_output.astype(np.int16), affine)
    nib.save(result_nii, output_path)
    
    print(f"💾 Saved result to: {output_path}")

if __name__ == "__main__":
    main()