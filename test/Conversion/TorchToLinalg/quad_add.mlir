// RUN: torch-mlir-opt <%s -torch-decompose-complex-ops -convert-torch-to-linalg -split-input-file | FileCheck %s

// `torch.quad_add` has no TorchToLinalg pattern of its own -- it must first be
// decomposed (by -torch-decompose-complex-ops) into `aten.mul.Tensor`/`aten.add.Tensor`,
// which already have Linalg lowerings. This test proves the whole chain compiles.

// CHECK-LABEL:   func.func @quad_add_to_linalg(
// CHECK-NOT:       torch.quad_add
// CHECK-COUNT-4:   linalg.generic
func.func @quad_add_to_linalg(%a: !torch.vtensor<[?],f32>, %b: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.quad_add %a, %b : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}
