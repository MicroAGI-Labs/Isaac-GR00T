"""Compose a GR00T N1.7 multipart export into the common raw-observation policy graph.

All processing is frozen in the ONNX artifact. The serving process requires only
NumPy and ONNX Runtime; it does not import GR00T, Torch, or a policy plug-in.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh


class Graph:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.nodes, self.initializers, self.opsets = [], [], {"": 18}
        self.weights = None
        self.counter = 0
        self.cache = {}

    def const(self, value, name=None):
        if name is None:
            name = f"constant_{self.counter}"
            self.counter += 1
        tensor = nh.from_array(np.asarray(value), name)
        self.initializers.append(tensor)
        return name

    def torch_const(self, value, name):
        import torch

        if value.dtype != torch.bfloat16:
            return self.const(value.cpu().numpy(), name)
        tensor = T()
        tensor.name = name
        tensor.dims.extend(value.shape)
        tensor.data_type = T.BFLOAT16
        tensor.raw_data = value.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()
        self.initializers.append(tensor)
        return name

    def node(self, op, *inputs, outputs=None, **attrs):
        if outputs is None:
            outputs = [f"{op}_{self.counter}"]
            self.counter += 1
        self.nodes.append(h.make_node(op, list(inputs), outputs, **attrs))
        return outputs[0] if len(outputs) == 1 else outputs

    def reshape(self, x, shape):
        return self.node("Reshape", x, self.const(np.array(shape, np.int64)))

    def cast(self, x, dtype):
        return self.node("Cast", x, to=dtype)

    def slice(self, x, start, end, axis):
        return self.node(
            "Slice",
            x,
            self.const(np.array([start], np.int64)),
            self.const(np.array([end], np.int64)),
            self.const(np.array([axis], np.int64)),
        )

    def inline(self, model_path, prefix, inputs):
        key = str(model_path)
        if key not in self.cache:
            model = onnx.load(model_path, load_external_data=False)
            if model.functions:
                raise ValueError("Export graph must use standard ONNX operators")
            for opset in model.opset_import:
                if opset.domain and opset.domain not in self.opsets:
                    raise ValueError(f"Unsupported custom operator domain {opset.domain}")
                self.opsets[opset.domain] = max(self.opsets.get(opset.domain, 0), opset.version)
            initial_names = {}
            for tensor in model.graph.initializer:
                name = f"weights_{len(self.cache)}_{tensor.name}"
                initial_names[tensor.name] = name
                tensor.name = name
                if tensor.data_location == T.EXTERNAL:
                    data = {e.key: e.value for e in tensor.external_data}
                    source = Path(model_path).parent / data["location"]
                    target = self.output / f"source_{len(self.cache)}_{source.name}"
                    if not target.exists():
                        os.link(source, target)
                    del tensor.external_data[:]
                    for k, v in {**data, "location": target.name}.items():
                        tensor.external_data.add(key=k, value=v)
                elif tensor.raw_data:
                    if self.weights is None:
                        self.weights = (self.output / "policy.weights").open("wb")
                    offset = self.weights.tell()
                    self.weights.write(tensor.raw_data)
                    length = len(tensor.raw_data)
                    tensor.ClearField("raw_data")
                    tensor.data_location = T.EXTERNAL
                    for k, v in {
                        "location": "policy.weights",
                        "offset": str(offset),
                        "length": str(length),
                    }.items():
                        tensor.external_data.add(key=k, value=v)
                self.initializers.append(tensor)
            del model.graph.initializer[:]
            self.cache[key] = (model, initial_names)
        model, initial_names = self.cache[key]
        mapping = dict(initial_names)
        mapping.update(inputs)

        def rename(name):
            return mapping.get(name, prefix + name) if name else name

        for original in model.graph.node:
            node = copy.deepcopy(original)
            node.name = prefix + node.name
            node.input[:] = [rename(n) for n in node.input]
            node.output[:] = [rename(n) for n in node.output]
            # Exported networks are flat; refusing subgraphs avoids accidental capture bugs.
            if any(
                a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for a in node.attribute
            ):
                raise ValueError("Multipart network contains an unexpected control-flow subgraph")
            self.nodes.append(node)
        return {out.name: rename(out.name) for out in model.graph.output}

    def save(self, inputs, outputs, metadata, name="policy.onnx"):
        if self.weights is not None:
            self.weights.flush()
        graph = h.make_graph(
            self.nodes, "raw_observation_policy", inputs, outputs, initializer=self.initializers
        )
        model = h.make_model(
            graph, opset_imports=[h.make_opsetid(k, v) for k, v in self.opsets.items()]
        )
        model.ir_version = 10
        h.set_model_props(model, {"microevals.policy": json.dumps(metadata)})
        onnx.save(model, self.output / name)
        return self.output / name


def cv_area_up_axis(source, target, *, horizontal):
    # OpenCV 4.11 resize.cpp: INTER_AREA upsampling uses fixed-point bilinear.
    # https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/resize.cpp
    inv_scale = target / source
    scale = 1.0 / inv_scale
    dest = np.arange(target, dtype=np.float64)
    index = np.floor(dest * scale).astype(np.int64)
    fraction = np.asarray((dest + 1) - (index + 1) * inv_scale, dtype=np.float32)
    fraction = np.where(fraction <= 0, np.float32(0), fraction - np.floor(fraction)).astype(
        np.float32
    )
    if horizontal:
        fraction = np.where(index >= source - 1, np.float32(0), fraction)
        index = np.minimum(index, source - 1)
    weights = np.stack(
        (
            np.rint((np.float32(1) - fraction) * np.float32(2048)),
            np.rint(fraction * np.float32(2048)),
        ),
        axis=-1,
    ).astype(np.float64)
    return np.minimum(index, source - 1), np.minimum(index + 1, source - 1), weights


def resize(g, x, source, target, mode="area"):
    if mode == "area":
        assert target[0] >= source[0] and target[1] >= source[1]
        x0, x1, a = cv_area_up_axis(source[1], target[1], horizontal=True)
        y0, y1, b = cv_area_up_axis(source[0], target[0], horizontal=False)
        left = g.node("Gather", x, g.const(x0), axis=3)
        right = g.node("Gather", x, g.const(x1), axis=3)
        horizontal = g.node(
            "Add",
            g.node("Mul", left, g.const(a[:, 0].astype(np.float32))),
            g.node("Mul", right, g.const(a[:, 1].astype(np.float32))),
        )
        horizontal = g.cast(
            g.node("Floor", g.node("Div", horizontal, g.const(np.float32(16)))), T.DOUBLE
        )
        upper = g.node("Gather", horizontal, g.const(y0), axis=2)
        lower = g.node("Gather", horizontal, g.const(y1), axis=2)
        upper = g.node(
            "Floor",
            g.node(
                "Div",
                g.node("Mul", upper, g.const(b[:, 0].reshape(1, 1, -1, 1))),
                g.const(np.float64(65536)),
            ),
        )
        lower = g.node(
            "Floor",
            g.node(
                "Div",
                g.node("Mul", lower, g.const(b[:, 1].reshape(1, 1, -1, 1))),
                g.const(np.float64(65536)),
            ),
        )
        x = g.node(
            "Floor",
            g.node(
                "Div",
                g.node("Add", g.node("Add", upper, lower), g.const(np.float64(2))),
                g.const(np.float64(4)),
            ),
        )
        return g.cast(x, T.FLOAT)
    else:
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2.functional import resize as tv_resize

        assert source[0] == target[0], "Current trained image pipeline only changes width"
        basis = torch.eye(source[1], dtype=torch.float64).reshape(source[1], 1, 1, source[1])
        coefficient = tv_resize(
            basis, [1, target[1]], interpolation=InterpolationMode.BICUBIC, antialias=True
        ).numpy()[:, 0, 0]
        maximum = float(coefficient.max())
        precision = 0
        while precision < 22:
            if int(0.5 + maximum * (1 << (precision + 1))) >= 32768:
                break
            precision += 1
        scale = 1 << precision
        coefficient = np.trunc(coefficient * scale + np.where(coefficient < 0, -0.5, 0.5)).astype(
            np.float32
        )
        x = g.node("MatMul", x, g.const(coefficient))
        x = g.node(
            "Floor",
            g.node(
                "Div", g.node("Add", x, g.const(np.float32(scale // 2))), g.const(np.float32(scale))
            ),
        )
        return g.node("Clip", x, g.const(np.float32(0)), g.const(np.float32(255)))


def preprocessing(g, processor, norm_params):
    config = processor.modality_configs["new_embodiment"]
    mins, maxs = [], []
    for key in config["state"].modality_keys:
        mins.extend(norm_params["state"][key]["min"])
        maxs.extend(norm_params["state"][key]["max"])
    # NumPy trained processor calculates float64 then assigns back to the float32 state.
    state = g.node(
        "Concat",
        "get_joint_positions",
        "get_commanded_positions",
        g.node("Div", "get_elapsed_time", g.const(np.float32(10))),
        axis=-1,
    )
    low, high = np.array(mins), np.array(maxs)
    valid = ~np.isclose(high, low)
    state = g.cast(state, T.DOUBLE)
    state = g.node(
        "Div", g.node("Sub", state, g.const(low)), g.const(np.where(valid, high - low, 1))
    )
    state = g.cast(state, T.FLOAT)
    state = g.node("Sub", g.node("Mul", state, g.const(np.float32(2))), g.const(np.float32(1)))
    state = g.node("Where", g.const(valid), state, g.const(np.zeros(len(low), np.float32)))
    state = g.node("Clip", state, g.const(np.float32(-1)), g.const(np.float32(1)))
    state = g.node("Concat", state, g.const(np.zeros((1, 132 - len(low)), np.float32)), axis=-1)
    state = g.cast(g.reshape(state, [1, 1, 132]), T.BFLOAT16)
    x = g.cast(g.reshape("get_camera_obs", [4, 480, 640, 3]), T.FLOAT)
    # PIL BOX 2x downsample rounds after each one-dimensional pass.
    x = g.reshape(x, [4, 480, 320, 2, 3])
    x = g.node("ReduceMean", x, g.const(np.array([3], np.int64)), keepdims=0)
    x = g.node("Floor", g.node("Add", x, g.const(np.float32(0.5))))
    x = g.reshape(x, [4, 240, 2, 320, 3])
    x = g.node("ReduceMean", x, g.const(np.array([2], np.int64)), keepdims=0)
    x = g.node("Floor", g.node("Add", x, g.const(np.float32(0.5))))
    x = g.node("Transpose", x, perm=[0, 3, 1, 2])
    first = (256, round(320 * 256 / 240))
    x = resize(g, x, (240, 320), first)
    crop = (int(first[0] * 0.95), int(first[1] * 0.95))
    top, left = ((first[i] - crop[i]) // 2 for i in range(2))
    x = g.slice(g.slice(x, top, top + crop[0], 2), left, left + crop[1], 3)
    second = (256, round(crop[1] * 256 / crop[0]))
    x = resize(g, x, crop, second)
    target = (256, 352)
    x = resize(g, x, second, target, "bicubic")
    x = g.node("Div", g.node("Sub", x, g.const(np.float32(127.5))), g.const(np.float32(127.5)))
    # Qwen repeats the single image into two temporal frames before patchification.
    x = g.reshape(x, [4, 1, 3, 256, 352])
    x = g.node("Concat", x, x, axis=1)
    patches = []
    for index in range(4):
        camera = g.node("Gather", x, g.const(np.int64(index)), axis=0)
        camera = g.reshape(camera, [2, 3, 8, 2, 16, 11, 2, 16])
        camera = g.node("Transpose", camera, perm=[2, 5, 3, 6, 1, 0, 4, 7])
        patches.append(g.reshape(camera, [352, 1536]))
    x = g.node("Concat", *patches, axis=0)
    return state, g.cast(g.cast(x, T.BFLOAT16), T.FLOAT)


def load_processor(source):
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    processor = Gr00tN1d7Processor.from_pretrained(
        source / "checkpoint",
        model_name=str(source / "backbone"),
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": False},
    )
    processor.eval()
    return processor


def validate_binding(bundle, processor):
    """Refuse a different producer interface before specializing any graph."""
    expected = {
        "get_joint_positions": {"shape": [1, 29], "dtype": "float32"},
        "get_commanded_positions": {"shape": [1, 29], "dtype": "float32"},
        "get_elapsed_time": {"shape": [1, 1], "dtype": "float32"},
        "get_camera_obs": {"shape": [1, 4, 480, 640, 3], "dtype": "uint8"},
    }
    data = bundle["groot"]
    selected = processor.modality_configs["new_embodiment"]
    if (
        bundle["inference"]["onnx"]["observations"] != expected
        or bundle["inference"]["control"]["hz"] != 10
        or bundle["inference"]["control"]["horizon"] != 16
        or data["image_resize"] != [240, 320]
        or data["elapsed_divisor"] != 10
        or data["padded_horizon"] != 40
        or data["diffusion_steps"] != 4
        or selected["state"].modality_keys
        != [
            "waist",
            "left_arm",
            "right_arm",
            "lift",
            "left_hand",
            "right_hand",
            "prev_cmd",
            "elapsed",
        ]
        or selected["action"].modality_keys != list(data["state_slices"])
        or selected["video"].modality_keys != data["camera_model_keys"]
        or len(data["camera_model_keys"]) != 4
        or processor.shortest_image_edge != 256
        or processor.crop_fraction != 0.95
        or not processor.use_albumentations
        or processor.letter_box_transform
        or processor.exclude_state
        or processor.apply_sincos_state_encoding
        or not processor.clip_outliers
        or any(c.rep.name != "ABSOLUTE" for c in selected["action"].action_configs)
    ):
        raise ValueError(
            "This producer export requires the qualified four-camera G1-D N1.7 binding"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()
    processor = load_processor(args.source)
    bundle = json.loads((args.source / "microevals-bundle.json").read_text())
    validate_binding(bundle, processor)
    g = Graph(args.output)
    state, pixels = preprocessing(
        g, processor, processor.state_action_processor.norm_params["new_embodiment"]
    )
    inputs = [
        h.make_tensor_value_info(
            name, {"float32": T.FLOAT, "uint8": T.UINT8}[spec["dtype"]], spec["shape"]
        )
        for name, spec in bundle["inference"]["onnx"]["observations"].items()
    ]
    if args.preprocess_only:
        g.node("Identity", g.cast(state, T.FLOAT), outputs=["state"])
        g.node("Identity", pixels, outputs=["pixel_values"])
        g.save(
            inputs,
            [
                h.make_tensor_value_info("state", T.FLOAT, [1, 1, 132]),
                h.make_tensor_value_info("pixel_values", T.FLOAT, [1408, 1536]),
            ],
            bundle["inference"],
            name="preprocess.onnx",
        )
        return
    from types import SimpleNamespace

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData
    from safetensors.torch import load_file
    import torch
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    data = bundle["groot"]
    raw = {k: np.zeros((1, b - a), np.float32) for k, (a, b) in data["state_slices"].items()}
    raw["prev_cmd"] = np.zeros((1, 29), np.float32)
    raw["elapsed"] = np.zeros((1, 1), np.float32)
    images = {k: np.zeros((1, 240, 320, 3), np.uint8) for k in data["camera_model_keys"]}
    step = VLAStepData(
        images=images,
        states=raw,
        actions={},
        text=bundle["inference"]["io"]["prompt"],
        embodiment=EmbodimentTag.NEW_EMBODIMENT,
    )
    processed = processor.collator(
        [processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])]
    )["inputs"]
    config = Qwen3VLConfig.from_pretrained(args.source / "backbone", local_files_only=True)
    ids = processed["input_ids"]
    mask = processed["attention_mask"]
    assert bool((mask == 1).all()), "Export prompt must have no padding"
    image_mask = ids == config.image_token_id
    positions, _ = Qwen3VLModel.get_rope_index(
        SimpleNamespace(config=config), ids, processed["image_grid_thw"], attention_mask=mask
    )
    weights = load_file(str(args.source / data["auxiliary_weights"]))
    embeddings = torch.nn.functional.embedding(
        ids, weights["backbone.model.model.language_model.embed_tokens.weight"]
    )
    vision = g.inline(args.source / "onnx/vit_fp32.onnx", "vit/", {"pixel_values": pixels})
    visual = g.cast(vision["image_embeds"], T.BFLOAT16)
    # Image placeholder positions are constant for the exported prompt and sensor layout.
    chunks = []
    visual_start = 0
    start = 0
    flags = image_mask[0].tolist()
    while start < len(flags):
        end = start + 1
        while end < len(flags) and flags[end] == flags[start]:
            end += 1
        if flags[start]:
            value = g.slice(visual, visual_start, visual_start + end - start, 0)
            value = g.reshape(value, [1, end - start, config.text_config.hidden_size])
            visual_start += end - start
        else:
            value = g.torch_const(embeddings[:, start:end], f"text_{start}")
        chunks.append(value)
        start = end
    embeds = g.node("Concat", *chunks, axis=1)
    attention = g.const(mask.numpy(), "attention_mask")
    image_positions = g.const(image_mask.numpy(), "image_mask")
    llm_inputs = {
        "inputs_embeds": embeds,
        "attention_mask": attention,
        "position_ids": g.const(positions.numpy(), "position_ids"),
        "visual_pos_masks": image_positions,
    }
    for index in range(3):
        part = g.node("Gather", vision["deepstack_features"], g.const(np.int64(index)), axis=0)
        llm_inputs[f"deepstack_{index}"] = g.cast(part, T.BFLOAT16)
    language = g.inline(args.source / "onnx/llm_bf16.onnx", "llm/", llm_inputs)["embeddings"]
    # Torch layer_norm on bf16 accumulates statistics in fp32 and rounds the result back.
    norm_weight = g.torch_const(weights["action_head.vlln.weight"], "vlln_weight")
    norm_bias = g.torch_const(weights["action_head.vlln.bias"], "vlln_bias")
    language = g.node(
        "LayerNormalization",
        language,
        norm_weight,
        norm_bias,
        axis=-1,
        epsilon=1e-5,
        stash_type=T.FLOAT,
    )
    vl = g.inline(args.source / "onnx/vl_self_attention.onnx", "vl/", {"hidden_states": language})[
        "output"
    ]
    embodiment = g.const(processed["embodiment_id"].numpy(), "embodiment_id")
    encoded_state = g.inline(
        args.source / "onnx/state_encoder.onnx",
        "state/",
        {"state": state, "embodiment_id": embodiment},
    )["output"]
    horizon = data["padded_horizon"]
    actions = g.node(
        "RandomNormal",
        shape=[1, horizon, 132],
        dtype=T.FLOAT,
        mean=0.0,
        scale=1.0,
        outputs=["diffusion_noise"],
    )
    actions = g.cast(actions, T.BFLOAT16)
    positional = g.torch_const(
        weights["action_head.position_embedding.weight"][:horizon][None], "action_position"
    )
    steps = data["diffusion_steps"]
    for index in range(steps):
        timestep = g.const(np.array([int(index / steps * 1000)], np.int64), f"timestep_{index}")
        encoded = g.inline(
            args.source / "onnx/action_encoder.onnx",
            f"encoder{index}/",
            {"actions": actions, "timesteps": timestep, "embodiment_id": embodiment},
        )["output"]
        sa = g.node("Concat", encoded_state, g.node("Add", encoded, positional), axis=1)
        dit = g.inline(
            args.source / "onnx/dit_bf16.onnx",
            f"dit{index}/",
            {
                "sa_embs": sa,
                "vl_embs": vl,
                "timestep": timestep,
                "image_mask": image_positions,
                "backbone_attention_mask": g.const(
                    (mask == 1).numpy(), "backbone_mask_" + str(index)
                ),
            },
        )["output"]
        velocity = g.inline(
            args.source / "onnx/action_decoder.onnx",
            f"decoder{index}/",
            {"model_output": dit, "embodiment_id": embodiment},
        )["output"]
        velocity = g.slice(velocity, 1, horizon + 1, 1)
        scaled = g.node(
            "Mul",
            velocity,
            g.torch_const(torch.tensor(1.0 / steps, dtype=torch.bfloat16), f"scale_{index}"),
        )
        actions = g.node("Add", actions, scaled)
    norm = processor.state_action_processor.norm_params["new_embodiment"]["action"]
    low = np.concatenate([norm[k]["min"] for k in data["state_slices"]])
    high = np.concatenate([norm[k]["max"] for k in data["state_slices"]])
    decoded = g.slice(g.slice(g.cast(actions, T.FLOAT), 0, 16, 1), 0, 29, 2)
    decoded = g.node("Clip", decoded, g.const(np.float32(-1)), g.const(np.float32(1)))
    decoded = g.node("Div", g.node("Add", decoded, g.const(np.float32(1))), g.const(np.float32(2)))
    decoded = g.cast(
        g.node("Add", g.node("Mul", g.cast(decoded, T.DOUBLE), g.const(high - low)), g.const(low)),
        T.FLOAT,
    )
    g.node(
        "Concat", decoded, g.const(np.zeros((1, 16, 2), np.float32)), axis=2, outputs=["actions"]
    )
    g.save(inputs, [h.make_tensor_value_info("actions", T.FLOAT, [1, 16, 31])], bundle["inference"])
    print("EXPORTED", args.output / "policy.onnx", flush=True)


if __name__ == "__main__":
    main()
