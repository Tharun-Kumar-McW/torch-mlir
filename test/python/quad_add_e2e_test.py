# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
#
# End-to-end (e2e) pytest for `torch.quad_add` (Approach 1: decompose to
# `aten.mul.Tensor`/`aten.add.Tensor`, then lower through the standard Linalg
# RefBackend pipeline and execute via the MLIR ExecutionEngine JIT).
#
# Unlike the lit tests in test/Dialect/Torch/decompose-complex-ops.mlir and
# test/Conversion/TorchToLinalg/quad_add.mlir (which check the *shape* of the
# generated IR via FileCheck), this test proves the whole pipeline produces
# the *numerically correct* answer for `a*b + (a*a + b)`, by actually running
# compiled code and comparing against a NumPy reference.
#
# `quad_add` is not a real PyTorch/ATen op, so there is no `torch.nn.Module`
# that can call it through the normal FX-import path. Instead this test hand
# -writes the Torch-dialect IR directly (`torch.quad_add` is valid, parsable
# IR the moment `include/torch-mlir/Dialect/Torch/IR/TorchOps.td` declares
# it), then drives it through the same `torch-lower-to-backend-contract` +
# `torch-backend-to-linalg-on-tensors-backend-pipeline` + RefBackend machinery
# the e2e test framework uses internally for real PyTorch modules.
#
# Run with:
#   PYTHONPATH=$PYTHONPATH ./torch_venv_shlo/bin/python -m pytest test/python/quad_add_e2e_test.py -v
# (this file adds the required paths to sys.path itself, so a bare
#  `pytest test/python/quad_add_e2e_test.py` from the repo root also works
#  as long as the venv/interpreter has `torch_mlir` built for it.)

import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_BUILD_PYPATH = os.path.join(
    _REPO_ROOT, "build", "tools", "torch-mlir", "python_packages", "torch_mlir"
)
_PT1_PYPATH = os.path.join(_REPO_ROOT, "projects", "pt1", "python")
for _p in (_BUILD_PYPATH, _PT1_PYPATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_mlir.ir import Context, Module
from torch_mlir.passmanager import PassManager
import torch_mlir.dialects.torch as torch_d
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)

BACKEND_PIPELINE = (
    "builtin.module("
    "torch-lower-to-backend-contract,"
    "torch-backend-to-linalg-on-tensors-backend-pipeline"
    ")"
)


def _quad_add_mlir(shape: str, dtype: str) -> str:
    ty = f"!torch.vtensor<[{shape}],{dtype}>"
    return f"""
func.func @quad_add_e2e(%a: {ty}, %b: {ty}) -> {ty} {{
  %0 = torch.quad_add %a, %b : {ty}, {ty} -> {ty}
  return %0 : {ty}
}}
"""


def _compile_and_load(shape: str, dtype: str, *, check_decomposed: bool = False):
    """Parses hand-written `torch.quad_add` IR, lowers it through the same
    decompose -> Linalg -> LLVM pipeline used by Approach 1, and returns an
    invoker whose `.quad_add_e2e(a, b)` runs the JIT-compiled function.
    """
    with Context() as ctx:
        torch_d.register_dialect(ctx)
        module = Module.parse(_quad_add_mlir(shape, dtype))

        if check_decomposed:
            decompose_only = PassManager.parse(
                "builtin.module(func.func(torch-decompose-complex-ops))"
            )
            probe = Module.parse(_quad_add_mlir(shape, dtype))
            torch_d.register_dialect(probe.context)
            decompose_only.run(probe.operation)
            probe_text = str(probe)
            assert "torch.quad_add" not in probe_text, (
                "torch.quad_add should have been eliminated by "
                "-torch-decompose-complex-ops (Approach 1), but it wasn't"
            )
            assert "torch.aten.mul.Tensor" in probe_text
            assert "torch.aten.add.Tensor" in probe_text

        PassManager.parse(BACKEND_PIPELINE).run(module.operation)

        backend = RefBackendLinalgOnTensorsBackend()
        compiled = backend.compile(module)
        return backend.load(compiled)


def _reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b + (a * a + b)


@pytest.mark.parametrize(
    "a_vals,b_vals",
    [
        ([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 2.5]),
        ([-2.0, -1.5, 0.5, 3.0], [5.0, -5.0, 0.0, 2.0]),
    ],
)
def test_quad_add_1d_f32(a_vals, b_vals):
    invoker = _compile_and_load("4", "f32", check_decomposed=True)
    a = np.array(a_vals, dtype=np.float32)
    b = np.array(b_vals, dtype=np.float32)
    result = invoker.quad_add_e2e(a, b)
    np.testing.assert_allclose(result, _reference(a, b), rtol=1e-5, atol=1e-6)


def test_quad_add_2d_f32():
    invoker = _compile_and_load("3,4", "f32")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    result = invoker.quad_add_e2e(a, b)
    np.testing.assert_allclose(result, _reference(a, b), rtol=1e-5, atol=1e-6)


def test_quad_add_f64():
    invoker = _compile_and_load("8", "f64")
    rng = np.random.default_rng(1)
    a = rng.standard_normal(8).astype(np.float64)
    b = rng.standard_normal(8).astype(np.float64)
    result = invoker.quad_add_e2e(a, b)
    np.testing.assert_allclose(result, _reference(a, b), rtol=1e-12, atol=1e-12)


def test_quad_add_matches_manual_decomposition():
    """Sanity-checks the *formula* itself against the naive expression, so a
    future change to the decomposition pattern's operand order (e.g. an
    accidental `b*a*a` vs `a*a*b` swap, which would still be numerically
    identical for commutative ops but could hide a real bug for a
    differently-ordered rewrite) is caught against both the JIT result and a
    plain NumPy expression evaluated independently.
    """
    invoker = _compile_and_load("5", "f32")
    a = np.array([1.0, -2.0, 3.5, 0.0, -0.25], dtype=np.float32)
    b = np.array([2.0, 0.5, -1.0, 4.0, 10.0], dtype=np.float32)
    result = invoker.quad_add_e2e(a, b)
    naive = (a * b) + ((a * a) + b)
    np.testing.assert_allclose(result, naive, rtol=1e-5, atol=1e-6)
