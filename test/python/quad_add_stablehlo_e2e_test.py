# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
#
# End-to-end (e2e) pytest for `torch.quad_add` -- Approach 2: direct
# `TorchToStablehlo` lowering (`ConvertQuadAddOp` in
# lib/Conversion/TorchToStablehlo/Basic.cpp), with NO decomposition into
# `aten.mul.Tensor`/`aten.add.Tensor` anywhere in the pipeline.
#
# `quad_add` is not a real PyTorch/ATen op, so there is no `torch.nn.Module`
# that can call it through the normal FX-import path. This test hand-writes
# the Torch-dialect IR directly, then drives it through:
#
#   torch-lower-to-backend-contract{backend-legal-ops=quad_add}
#       -> torch-backend-to-stablehlo-backend-pipeline
#       -> LinalgOnTensorsStablehloBackend (stablehlo -> linalg -> LLVM JIT)
#
# The `backend-legal-ops=quad_add` option is the crucial bit: `DecomposeQuadAddOp`
# (Approach 1) is still registered in this tree, so without telling
# `torch-lower-to-backend-contract` to treat `quad_add` as backend-legal, it
# would decompose `torch.quad_add` into `aten.mul.Tensor`/`aten.add.Tensor`
# before the new direct pattern ever got a chance to run -- silently testing
# Approach 1 instead of Approach 2. This test explicitly checks the
# intermediate IR to prove `torch.quad_add` survives decomposition and is
# converted straight into `chlo`/`stablehlo` ops by `ConvertQuadAddOp`.
#
# Run with:
#   ./torch_venv_shlo/bin/python -m pytest test/python/quad_add_stablehlo_e2e_test.py -v

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
from torch_mlir_e2e_test.stablehlo_backends.linalg_on_tensors import (
    LinalgOnTensorsStablehloBackend,
)

# `backend-legal-ops=quad_add` keeps `torch.quad_add` intact through
# `torch-lower-to-backend-contract` -- without it, Approach 1's
# `DecomposeQuadAddOp` would still eat the op here.
BACKEND_PIPELINE = (
    "builtin.module("
    "torch-lower-to-backend-contract{backend-legal-ops=quad_add},"
    "torch-backend-to-stablehlo-backend-pipeline"
    ")"
)
DECOMPOSE_ONLY_LEGAL_PIPELINE = (
    "builtin.module(func.func(torch-decompose-complex-ops{legal-ops=quad_add}))"
)


def _quad_add_mlir(shape: str, dtype: str) -> str:
    ty = f"!torch.vtensor<[{shape}],{dtype}>"
    return f"""
func.func @quad_add_e2e(%a: {ty}, %b: {ty}) -> {ty} {{
  %0 = torch.quad_add %a, %b : {ty}, {ty} -> {ty}
  return %0 : {ty}
}}
"""


def _compile_and_load(shape: str, dtype: str, *, check_not_decomposed: bool = False):
    """Parses hand-written `torch.quad_add` IR, lowers it through the direct
    StableHLO pipeline (Approach 2, no decomposition), and returns an invoker
    whose `.quad_add_e2e(a, b)` runs the JIT-compiled function.
    """
    with Context() as ctx:
        torch_d.register_dialect(ctx)
        module = Module.parse(_quad_add_mlir(shape, dtype))

        if check_not_decomposed:
            probe = Module.parse(_quad_add_mlir(shape, dtype))
            torch_d.register_dialect(probe.context)
            PassManager.parse(DECOMPOSE_ONLY_LEGAL_PIPELINE).run(probe.operation)
            probe_text = str(probe)
            assert "torch.quad_add" in probe_text, (
                "torch.quad_add should have survived "
                "-torch-decompose-complex-ops{legal-ops=quad_add} untouched "
                "(Approach 2 must not decompose it), but it didn't"
            )
            assert "torch.aten.mul.Tensor" not in probe_text
            assert "torch.aten.add.Tensor" not in probe_text

        PassManager.parse(BACKEND_PIPELINE).run(module.operation)
        module_text = str(module)
        assert "stablehlo.multiply" in module_text or "chlo" in module_text
        assert "torch.quad_add" not in module_text

        backend = LinalgOnTensorsStablehloBackend()
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
def test_quad_add_stablehlo_1d_f32(a_vals, b_vals):
    invoker = _compile_and_load("4", "f32", check_not_decomposed=True)
    a = np.array(a_vals, dtype=np.float32)
    b = np.array(b_vals, dtype=np.float32)
    result = invoker.quad_add_e2e(a, b)
    np.testing.assert_allclose(result, _reference(a, b), rtol=1e-5, atol=1e-6)


def test_quad_add_stablehlo_2d_f32():
    invoker = _compile_and_load("3,4", "f32")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    result = invoker.quad_add_e2e(a, b)
    np.testing.assert_allclose(result, _reference(a, b), rtol=1e-5, atol=1e-6)


def test_quad_add_stablehlo_f64():
    invoker = _compile_and_load("8", "f64")
    rng = np.random.default_rng(1)
    a = rng.standard_normal(8).astype(np.float64)
    b = rng.standard_normal(8).astype(np.float64)
    result = invoker.quad_add_e2e(a, b)
    np.testing.assert_allclose(result, _reference(a, b), rtol=1e-12, atol=1e-12)


def test_quad_add_stablehlo_matches_manual_formula():
    """Cross-checks the JIT result against the naive expression evaluated
    independently in NumPy, so an accidental operand-order bug in
    `ConvertQuadAddOp` (e.g. swapping which operand feeds `aSquared` vs `ab`)
    would be caught even though `mul` is commutative and might otherwise hide
    a real ordering mistake for a hypothetical non-commutative variant.
    """
    invoker = _compile_and_load("5", "f32")
    a = np.array([1.0, -2.0, 3.5, 0.0, -0.25], dtype=np.float32)
    b = np.array([2.0, 0.5, -1.0, 4.0, 10.0], dtype=np.float32)
    result = invoker.quad_add_e2e(a, b)
    naive = (a * b) + ((a * a) + b)
    np.testing.assert_allclose(result, naive, rtol=1e-5, atol=1e-6)
