import torch
import os
import json
import importlib
import onnx
import numpy as np
import subprocess
import sys
import gc
from openvino import Core

# ============================================================
# USER CONFIGURATION
# ============================================================
CHECKPOINT_PATH = r"C:\nnunet\model_headandneck\nnUNet_results\Dataset001_headandneck\nnUNetTrainerNoMirroring__nnUNetResEncUNetMPlans__3d_fullres\fold_0\checkpoint_latest.pth"
PLANS_JSON_PATH = r"C:\nnunet\model_headandneck\nnUNet_results\Dataset001_headandneck\nnUNetTrainerNoMirroring__nnUNetResEncUNetMPlans__3d_fullres\plans.json"
CONFIGURATION = "3d_fullres"

OUTPUT_ONNX_PATH = r"C:\nnunet\model_headandneck\onnx\model.onnx"
OPENVINO_OUTPUT_DIR = r"C:\nnunet\model_headandneck\openvino"
# ============================================================

def load_exact_nnunet_model(checkpoint_path: str, plans_json_path: str, configuration: str):
    """
    Reconstructs the model manually from weights and plans.json.
    """
    print(f"🔧 Loading Checkpoint: {checkpoint_path}")
    
    # 1. Load State Dict
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['network_weights']

    # 2. Detect Number of Classes dynamically
    num_classes = 32 # Default fallback
    for key in state_dict.keys():
        if 'seg_outputs' in key and 'weight' in key:
            num_classes = state_dict[key].shape[0]
            break
    print(f"   Detected Output Classes: {num_classes}")

    # 3. Load Plans
    with open(plans_json_path, 'r') as f:
        plans = json.load(f)

    # 4. Extract Architecture Parameters
    config = plans['configurations'][configuration]
    
    # Handle both nnU-Net v2 and potential older JSON structures
    if 'network_arch_init_kwargs' in config:
        network_config = config['network_arch_init_kwargs']
    elif 'architecture' in config and 'arch_kwargs' in config['architecture']:
        network_config = config['architecture']['arch_kwargs']
    else:
        raise ValueError("Could not find architecture parameters in plans.json")

    print("   Extracted architecture parameters.")

    # 5. Import Network Class
    try:
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
        network_class = ResidualEncoderUNet
    except ImportError as e:
        raise ImportError(f"Failed to import network class: {e}")

    # 6. Convert string arguments to classes
    def string_to_class(class_string):
        if class_string is None: return None
        if isinstance(class_string, str):
            module_path, class_name = class_string.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        return class_string

    valid_params = {'input_channels': 1, 'num_classes': num_classes}
    for param, value in network_config.items():
        if param in ['conv_op', 'norm_op', 'dropout_op', 'nonlin']:
            valid_params[param] = string_to_class(value)
        else:
            valid_params[param] = value

    # 7. Build Model
    model = network_class(**valid_params)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"   Missing Keys: {len(missing)} | Unexpected Keys: {len(unexpected)}")
    
    model.eval()
    return model

def get_patch_size(plans_json_path, configuration):
    with open(plans_json_path, 'r') as f:
        plans = json.load(f)
    return plans['configurations'][configuration]['patch_size']

# ============================================================
# MAIN PIPELINE
# ============================================================
print("🚀 Starting Conversion Pipeline...")

# Step 1: Load & Export to ONNX
print("\n=== STEP 1: PyTorch -> ONNX ===")
try:
    patch_size = get_patch_size(PLANS_JSON_PATH, CONFIGURATION)
except KeyError:
    print("   Warning: Patch size not found, using default.")
    patch_size = [56, 192, 192]

input_shape = [1, 1] + patch_size
print(f"   Input Shape: {input_shape}")

model = load_exact_nnunet_model(CHECKPOINT_PATH, PLANS_JSON_PATH, CONFIGURATION)
dummy_input = torch.randn(*input_shape)

print("   Exporting to ONNX (Opset 11)...")
try:
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            OUTPUT_ONNX_PATH,
            export_params=True,
            opset_version=11,              
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
    print(f"✅ Exported to: {OUTPUT_ONNX_PATH}")
except Exception as e:
    print(f"❌ Export Failed: {e}")
    sys.exit(1)

# Step 2: Aggressive Cleanup
print("\n🧹 Cleaning up PyTorch objects...")
del model        
del dummy_input 
gc.collect()
print("✅ Cleanup complete.")

# Step 3: Convert ONNX -> OpenVINO (Subprocess Isolation)
print("\n=== STEP 2: ONNX -> OPENVINO (Isolated Process) ===")
os.makedirs(OPENVINO_OUTPUT_DIR, exist_ok=True)

# Define shape string explicitly with quotes for subprocess safety
# NCDHW = Batch, Channel, Depth, Height, Width
shape_str = f"input[1,1,{patch_size[0]},{patch_size[1]},{patch_size[2]}]"
layout_str = "input(NCDHW)" # Explicit layout helps OpenVINO optimize 3D Convs

command = [
    sys.executable, "-m", "openvino.tools.ovc",
    OUTPUT_ONNX_PATH,
    "--output_model", os.path.join(OPENVINO_OUTPUT_DIR, "model"),
    "--input", shape_str,     # Defined shape
    "--compress_to_fp16", "False"
]

print("   Running OpenVINO Converter via subprocess...")
print(f"   Input: {shape_str}, Layout: {layout_str}")

result = subprocess.run(command, capture_output=True, text=True)

if result.stdout: print("   stdout:", result.stdout)
if result.stderr: print("   stderr:", result.stderr)

if result.returncode != 0:
    print("❌ OpenVINO Conversion Failed.")
    sys.exit(1)
print("✅ Conversion Successful!")

# Step 4: Verification
print("\n=== STEP 3: VERIFICATION ===")
core = Core()
model_xml = os.path.join(OPENVINO_OUTPUT_DIR, "model.xml")
model_bin = os.path.join(OPENVINO_OUTPUT_DIR, "model.bin")

if not os.path.exists(model_xml):
    print(f"❌ File not found: {model_xml}")
    sys.exit(1)

ov_model = core.read_model(model=model_xml)
inp_shape = ov_model.input(0).shape
out_shape = ov_model.output(0).shape
print(f"   Input Shape:  {inp_shape}")
print(f"   Output Shape: {out_shape}")

try:
    compiled = core.compile_model(ov_model, "CPU")
    print("✅ Model compiles successfully on CPU!")
except Exception as e:
    print(f"⚠️  Compilation Warning: {e}")

print("\n🎉 Pipeline Finished Successfully!")
print(f"   Model saved to: {OPENVINO_OUTPUT_DIR}")