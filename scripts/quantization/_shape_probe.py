import numpy as np
import openvino as ov
import sys

for p in (r"C:\nnunet reboot\models\int8\fold_0\quantized_model.xml",
          r"C:\nnunet reboot\models\original_dyn\fold_0\model.xml"):
    m = ov.Core().read_model(p)
    print(p.split("models")[1], "| in:", m.input(0).partial_shape, "| out:", m.output(0).partial_shape)

# что делает multiply при несовпадении
acc = np.zeros((32, 56, 192, 192), dtype=np.float32)
try:
    np.multiply(np.zeros((56, 192, 192)), np.zeros((56, 192, 192)), out=acc)
    print("multiply (56,192,192)x(56,192,192) -> out (32,56,192,192): OK (без ошибки!)")
except ValueError as e:
    print("multiply ValueError:", str(e)[:80])
