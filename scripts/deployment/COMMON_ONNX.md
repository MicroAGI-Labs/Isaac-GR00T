# Export a complete policy for common ONNX serving

`export_common_policy.py` composes the qualified G1-D N1.7 multipart export into
one `policy.onnx` with external weights. It includes the trained image processing,
state normalization, prompt embeddings, vision/language/action networks, all four
denoising steps and action decoding. The serving process needs only ONNX Runtime
and NumPy; no GR00T processor, Python callback or Torch runtime is loaded there.

The inputs use the existing raw-observation contract recorded in the bundle's
`inference` declaration: joint positions, previous commanded positions, elapsed
time and four uint8 RGB cameras. The sole output is float32
`actions[1,16,31]`: 29 joint targets and two zero base velocity channels. The
checkpoint has no learned base action. The prompt is frozen at export time.

Run in the pinned GR00T export environment with ONNX, OpenCV and Torch installed:

```sh
python scripts/deployment/export_common_policy.py \
  --source /path/to/qualified-multipart-export \
  --output /path/to/common-policy
```

The source folder contains the seven ONNX networks under `onnx/`, the trained
processor under `checkpoint/`, tokenizer configuration under `backbone/`, the
auxiliary checkpoint tensors, and the source export's binding manifest. The
result uses only standard ONNX operators. Large weights remain external files;
copy them together with `policy.onnx`.

The production graph samples denoising noise internally with ONNX `RandomNormal`.
Qualification compares it with the original inference implementation using an
identical injected noise tensor in a separate test graph. That test graph must
never be published. Qualify camera changes, previous-command and elapsed inputs,
finite decoded outputs, and exact zero base commands before publishing the common
`b1-onnx-v1` bundle. Use the unchanged common evaluator loader and test a native
simulation through both supported control paths.

The export specializes static sensor geometry and the trained processor settings.
It does not modify the environment's cameras, actuators, starting conditions,
control rate or task. Changing the checkpoint or prompt requires another export
and qualification.
