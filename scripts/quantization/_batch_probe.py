import os
import sys

import numpy as np
import openvino as ov
import nibabel as nib

sys.path.insert(0, r"C:\nnunet reboot\scripts")
import config
from infer_ov import ct_normalize

xml = r"C:\nnunet reboot\models\original_dyn\fold_0\model.xml"
core = ov.Core()
model = core.read_model(xml)
model.reshape({model.input(0).any_name: ov.PartialShape("[1..2,1,56,192,192]")})

CONFIGS = {
    "plain": {},
    "LATENCY": {"PERFORMANCE_HINT": "LATENCY"},
    "full_engine": {"INFERENCE_NUM_THREADS": "6", "PERFORMANCE_HINT": "LATENCY"},
}

nii = nib.load(os.path.join(config.PATIENTS_ROOT, "P110", "nnunet_input", "P110_0000.nii.gz"))
data = ct_normalize(np.transpose(nii.get_fdata(dtype=np.float32), (2, 1, 0)))[:90, 129:471, :512]

def patch(z, y, x):
    return np.ascontiguousarray(data[z:z+56, y:y+192, x:x+192])

p1, p2 = patch(0, 100, 100), patch(20, 200, 150)
r1 = None
for name, cfg in CONFIGS.items():
    compiled = core.compile_model(model, "CPU", config=cfg) if cfg else core.compile_model(model, "CPU")
    out = compiled.output(0)
    r_batch = compiled(np.stack([p1, p2])[:, None])[out]
    r_single = compiled(p1.reshape(1, 1, 56, 192, 192))[out]
    d = np.max(np.abs(r_batch[0] - r_single[0]))
    print(f"{name:>12}: batch[0] vs single: max|diff|={d:.2e}, "
          f"argmax согласованы: {np.array_equal(np.argmax(r_batch[0], axis=0), np.argmax(r_single[0], axis=0))}")
