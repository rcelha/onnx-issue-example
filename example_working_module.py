import onnx
import onnxruntime
import torch
import torch.jit
from torch import Tensor, nn


class ConvModule(nn.Module):
    def __init__(self, bias: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(1, 16, 1, bias=bias),
                nn.Conv1d(16, 1, 1, bias=bias),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x


def export():
    module = ConvModule().to("cpu")
    sample_input = torch.rand(1, 1, 100)
    module.eval()

    module_jit = torch.jit.script(module)
    torch.onnx.export(
        module_jit,
        sample_input,
        "/tmp/convmodule.onnx",
        verbose=False,
        opset_version=18,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {2: "x_time"},
            "y": {2: "y_time"},
        },
    )
    onnx_model = onnx.load("/tmp/convmodule.onnx")
    onnx.checker.check_model(onnx_model)


def test_inference():
    ort_session = onnxruntime.InferenceSession("/tmp/convmodule.onnx")
    sample_input = torch.rand(1, 1, 100).detach().cpu().numpy()
    ort_inputs = {"x": sample_input}
    _ = ort_session.run(None, ort_inputs)


if __name__ == "__main__":
    export()
    test_inference()
