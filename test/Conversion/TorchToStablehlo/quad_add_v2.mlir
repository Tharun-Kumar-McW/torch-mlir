// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @quad_add_to_stablehlo(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.vtensor<[?],f32>, %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[B:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK:           %[[A:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK-NOT:       torch.quad_add
// CHECK:           %[[AA:.*]] = chlo.broadcast_multiply %[[A]], %[[A]] : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[AAB:.*]] = chlo.broadcast_add %[[AA]], %[[B]] : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[AB:.*]] = chlo.broadcast_multiply %[[A]], %[[B]] : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[RESULT:.*]] = chlo.broadcast_add %[[AB]], %[[AAB]] : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK:           torch_c.from_builtin_tensor %[[RESULT]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
func.func @quad_add_to_stablehlo(%a: !torch.vtensor<[?],f32>, %b: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.quad_add %a, %b : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}