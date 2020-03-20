# SST-2 example

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "train",
    srcs = [
        "input_pipeline.py",
        "model.py",
        "train.py",
    ],
    main = "train.py",
    python_version = "PY3",
    deps = [
        "//learning/brain/research/jax:gpu_support",
        "//learning/brain/research/jax:tpu_support",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/flax",
        "//third_party/py/flax/metrics:tensorboard",
        "//third_party/py/flax/training",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow_datasets",
    ],
)
