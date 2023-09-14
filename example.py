from typing import List, Tuple

import onnx
import onnxruntime
import torch
import torch.jit
from torch import Tensor, nn


class ConvModule(nn.Module):
    def __init__(self, bias: bool = True):
        super().__init__()
        self.layer1 = nn.Conv1d(1, 16, 1, bias=bias)
        self.layer2 = nn.Conv1d(16, 1, 1, bias=bias)

    def forward(self, x: Tensor, aux: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        new_aux = []

        x = self.layer1.forward(x)
        aux_item = aux[0]
        x = x * aux_item
        aux_item = torch.rand_like(aux_item)
        new_aux.append(aux_item)

        x = self.layer2.forward(x)
        aux_item = aux[1]
        x = x * aux_item
        aux_item = torch.rand_like(aux_item)
        new_aux.append(aux_item)

        return x, new_aux


def export():
    module = ConvModule().to("cpu")
    sample_input = torch.rand(1, 1, 100)
    module.eval()
    aux = [
        torch.rand(1, 16, 100),
        torch.rand(1, 1, 100),
    ]
    _, aux = module.forward(sample_input, aux)
    module_jit = torch.jit.script(module)
    torch.onnx.export(
        module_jit,
        (sample_input, aux),
        "/tmp/convmodule.onnx",
        verbose=False,
        opset_version=18,
        input_names=["x", "aux"],
        output_names=["y", "new_aux"],
        dynamic_axes={
            "x": {2: "x_time"},
            "y": {2: "y_time"},
        },
    )
    onnx_model = onnx.load("/tmp/convmodule.onnx")
    onnx.checker.check_model(onnx_model)


def test_inference():
    sample_input = torch.rand(1, 1, 100).numpy()
    aux = [
        torch.rand(1, 16, 100).numpy(),
        torch.rand(1, 1, 100).numpy(),
    ]
    ort_session = onnxruntime.InferenceSession("/tmp/convmodule.onnx")
    for _ in range(10):
        ort_inputs = {"x": sample_input, "aux": aux}
        ort_outputs = ort_session.run(None, ort_inputs)
        aux = ort_outputs[1]
        for i in aux:
            print(aux.shape)


if __name__ == "__main__":
    export()
    test_inference()
