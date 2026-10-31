"""Regression checks for sharing external weights while composing graph calls."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from scripts.deployment.export_common_policy import Graph, resize


def test_composition_reuses_weights_and_keeps_calls_separate(tmp_path: Path):
    source = tmp_path / "network.onnx"
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "weight"], ["y"])],
        "network",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.array([2, 3], np.float32), "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, source)
    export = Graph(tmp_path / "out")
    first = export.inline(source, "first/", {"x": "input"})["y"]
    second = export.inline(source, "second/", {"x": first})["y"]
    export.node("Identity", second, outputs=["output"])
    output = export.save(
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        {},
    )
    exported = onnx.load(output, load_external_data=False)
    assert len(exported.graph.initializer) == 1
    assert exported.graph.initializer[0].data_location == TensorProto.EXTERNAL
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(
        session.run(None, {"input": np.array([[5, 7]], np.float32)})[0], [[20, 63]]
    )


def test_frozen_area_resize_matches_trained_uint8_kernel(tmp_path: Path):
    import cv2

    export = Graph(tmp_path / "area")
    value = resize(export, "input", (7, 11), (12, 17))
    export.node("Identity", value, outputs=["output"])
    output = export.save(
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 7, 11])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 12, 17])],
        {},
    )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1234)
    for image in [rng.integers(0, 256, (7, 11, 3), np.uint8), np.full((7, 11, 3), 127, np.uint8)]:
        candidate = session.run(None, {"input": image.transpose(2, 0, 1)[None].astype(np.float32)})[
            0
        ][0].transpose(1, 2, 0)
        reference = cv2.resize(image, (17, 12), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(candidate, reference)
