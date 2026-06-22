import json

import torch
from tensordict import TensorDict
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase


class _FlatTensorDictModule(torch.nn.Module):
    def __init__(self, module: ModBase, in_keys, out_keys, batch_size):
        super().__init__()
        self.module = module
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
        self.batch_size = torch.Size(batch_size)

    def forward(self, *args):
        td = TensorDict({}, batch_size=self.batch_size)
        for key, value in zip(self.in_keys, args):
            td.set(key, value)
        out = self.module(td)
        return tuple(out.get(key) for key in self.out_keys)


def _onnx_name(key, index: int) -> str:
    if isinstance(key, tuple):
        raw = "__".join(str(part) for part in key)
    else:
        raw = str(key)
    name = "".join(char if char.isalnum() or char == "_" else "_" for char in raw)
    return name or f"value_{index}"


def _onnx_names(keys) -> list[str]:
    names = []
    seen = {}
    for index, key in enumerate(keys):
        base = _onnx_name(key, index)
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count}")
    return names


@torch.inference_mode()
def export_onnx(module: ModBase, td: TensorDictBase, path: str, meta=None):
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")

    td = td.cpu().select(*module.in_keys, strict=True)
    module = module.cpu().eval()
    in_keys = list(module.in_keys)
    out_keys = list(module.out_keys)
    flat_module = _FlatTensorDictModule(module, in_keys, out_keys, td.batch_size).eval()
    input_tensors = tuple(td[key] for key in in_keys)
    input_names = _onnx_names(in_keys)
    output_names = _onnx_names(out_keys)
    torch.onnx.export(
        flat_module,
        args=input_tensors,
        f=path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model to {path}.")

    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["in_keys"] = in_keys
    meta["out_keys"] = out_keys
    meta["in_shapes"] = ([td[k].shape for k in in_keys],)

    json.dump(meta, open(meta_path, "w"), indent=4)
    print(f"Exported metadata to {meta_path}.")

    import onnxruntime as ort

    ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    onnx_input = tuple(td[k] for k in in_keys)
    onnxruntime_input = {
        k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
    }

    ort_output = ort_session.run(None, onnxruntime_input)
    assert len(ort_output) == len(out_keys)


def export_onnx_optional(module: ModBase, td: TensorDictBase, path: str, meta=None, *, required: bool = False) -> bool:
    try:
        export_onnx(module, td, path, meta)
    except Exception as exc:
        if required:
            raise
        print(f"[Warning] ONNX export failed for {path}: {exc}")
        return False
    return True
