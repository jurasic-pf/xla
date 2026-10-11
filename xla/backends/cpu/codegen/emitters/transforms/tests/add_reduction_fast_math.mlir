// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: emitters_opt %s -split-input-file -xla-cpu-add-reduction-fast-math-flags | FileCheck %s

func.func @caller(%x: f32, %y: f32) -> f32
{
  %z = func.call @reducer(%x, %y) { xla.is_reduction }: (f32, f32) -> f32
  func.return %z : f32
}

func.func @reducer(%x: f32, %y: f32) -> f32
{
  %z = arith.addf %x, %y : f32
  func.return %z : f32
}

// CHECK-LABEL: func.func @caller
// CHECK-LABEL: func.func @reducer
// CHECK: arith.addf {{.*}} fastmath<reassoc> : f32

// -----


func.func @caller(%x: f32, %y: f32) -> f32
{
  %z = func.call @reducer(%x, %y) { xla.is_reduction }: (f32, f32) -> f32
  func.return %z : f32
}

func.func @reducer(%x: f32, %y: f32) -> f32
{
  %w = arith.addf %x, %y : f32
  %z = arith.mulf %w, %y : f32
  func.return %z : f32
}

// CHECK-LABEL: func.func @caller
// CHECK-LABEL: func.func @reducer
// CHECK-NOT: fastmath

// -----

func.func @caller(%x: f32, %y: f64, %a: f32, %b: f64) -> (f32, f64)
{
  %z:2 = func.call @reducer(%x, %y, %a, %b) { xla.is_reduction }
      : (f32, f64, f32, f64) -> (f32, f64)
  func.return %z#0, %z#1 : f32, f64
}

func.func @reducer(%x: f32, %y: f64, %a: f32, %b: f64) -> (f32, f64)
{
  %s = arith.addf %x, %a : f32
  %p = arith.mulf %y, %b : f64
  func.return %s, %p : f32, f64
}

// CHECK-LABEL: func.func @caller
// CHECK-LABEL: func.func @reducer
// CHECK: arith.addf {{.*}} fastmath<reassoc> : f32
// CHECK: arith.mulf {{.*}} fastmath<reassoc> : f64

// -----

func.func @caller(%x: f32, %y: f32, %a: f32, %b: f32) -> (f32, f32)
{
  %z:2 = func.call @reducer(%x, %y, %a, %b) { xla.is_reduction }
      : (f32, f32, f32, f32) -> (f32, f32)
  func.return %z#0, %z#1 : f32, f32
}

// Both outputs read %a, so the accumulators are not independent.
func.func @reducer(%x: f32, %y: f32, %a: f32, %b: f32) -> (f32, f32)
{
  %s = arith.addf %x, %a : f32
  %t = arith.addf %y, %a : f32
  func.return %s, %t : f32, f32
}

// CHECK-LABEL: func.func @caller
// CHECK-LABEL: func.func @reducer
// CHECK-NOT: fastmath

// -----

func.func @caller(%x: f32, %y: f32, %a: f32, %b: f32) -> (f32, f32)
{
  %z:2 = func.call @reducer(%x, %y, %a, %b) { xla.is_reduction }
      : (f32, f32, f32, f32) -> (f32, f32)
  func.return %z#0, %z#1 : f32, f32
}

// Only adds and multiplies are treated as independent accumulators.
func.func @reducer(%x: f32, %y: f32, %a: f32, %b: f32) -> (f32, f32)
{
  %s = arith.addf %x, %a : f32
  %m = arith.maximumf %y, %b : f32
  func.return %s, %m : f32, f32
}

// CHECK-LABEL: func.func @caller
// CHECK-LABEL: func.func @reducer
// CHECK-NOT: fastmath
