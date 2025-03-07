module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @_anonymous5() {addr_space = 0 : i32} : !llvm.array<1 x i32>
  llvm.mlir.global external @_anonymous4() {addr_space = 0 : i32} : !llvm.array<1 x i32>
  llvm.mlir.global external @_anonymous3() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @_anonymous2() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @_anonymous1() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.mlir.global external @_anonymous0() {addr_space = 0 : i32} : !llvm.array<1024 x i16>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @vector_scalar_mul_int16_vector(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @_anonymous5 : !llvm.ptr
    %1 = llvm.mlir.addressof @_anonymous3 : !llvm.ptr
    %2 = llvm.mlir.addressof @_anonymous1 : !llvm.ptr
    %3 = llvm.mlir.addressof @_anonymous4 : !llvm.ptr
    %4 = llvm.mlir.addressof @_anonymous2 : !llvm.ptr
    %5 = llvm.mlir.constant(32 : index) : i64
    %6 = llvm.mlir.constant(true) : i1
    %7 = llvm.mlir.addressof @_anonymous0 : !llvm.ptr
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(49 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(51 : i32) : i32
    %13 = llvm.mlir.constant(53 : i32) : i32
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.mlir.constant(-1 : i32) : i32
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(2 : index) : i64
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(1024 : i32) : i32
    %20 = llvm.mlir.constant(9223372036854775806 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb8
    %22 = llvm.icmp "slt" %21, %20 : i64
    llvm.cond_br %22, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%13, %15) : (i32, i32) -> ()
    llvm.br ^bb3(%16 : i64)
  ^bb3(%23: i64):  // 2 preds: ^bb2, ^bb4
    %24 = llvm.icmp "slt" %23, %18 : i64
    llvm.cond_br %24, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%12, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %15) : (i32, i32) -> ()
    %25 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x i16>
    llvm.intr.assume %6 ["align"(%25, %5 : !llvm.ptr, i64)] : i1
    %26 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x i16>
    llvm.intr.assume %6 ["align"(%26, %5 : !llvm.ptr, i64)] : i1
    %27 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    llvm.intr.assume %6 ["align"(%27, %5 : !llvm.ptr, i64)] : i1
    llvm.call @vector_scalar_mul_int16_vector(%26, %25, %27, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %15) : (i32, i32) -> ()
    %28 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x i16>
    llvm.intr.assume %6 ["align"(%28, %5 : !llvm.ptr, i64)] : i1
    %29 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x i16>
    llvm.intr.assume %6 ["align"(%29, %5 : !llvm.ptr, i64)] : i1
    llvm.intr.assume %6 ["align"(%27, %5 : !llvm.ptr, i64)] : i1
    llvm.call @vector_scalar_mul_int16_vector(%29, %28, %27, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    %30 = llvm.add %23, %17 : i64
    llvm.br ^bb3(%30 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%8, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %15) : (i32, i32) -> ()
    llvm.br ^bb6(%16 : i64)
  ^bb6(%31: i64):  // 2 preds: ^bb5, ^bb7
    %32 = llvm.icmp "slt" %31, %18 : i64
    llvm.cond_br %32, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%12, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %15) : (i32, i32) -> ()
    %33 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x i16>
    llvm.intr.assume %6 ["align"(%33, %5 : !llvm.ptr, i64)] : i1
    %34 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x i16>
    llvm.intr.assume %6 ["align"(%34, %5 : !llvm.ptr, i64)] : i1
    %35 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i32>
    llvm.intr.assume %6 ["align"(%35, %5 : !llvm.ptr, i64)] : i1
    llvm.call @vector_scalar_mul_int16_vector(%34, %33, %35, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %15) : (i32, i32) -> ()
    %36 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x i16>
    llvm.intr.assume %6 ["align"(%36, %5 : !llvm.ptr, i64)] : i1
    %37 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x i16>
    llvm.intr.assume %6 ["align"(%37, %5 : !llvm.ptr, i64)] : i1
    llvm.intr.assume %6 ["align"(%35, %5 : !llvm.ptr, i64)] : i1
    llvm.call @vector_scalar_mul_int16_vector(%37, %36, %35, %19) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %14) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %14) : (i32, i32) -> ()
    %38 = llvm.add %31, %17 : i64
    llvm.br ^bb6(%38 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.release(%8, %14) : (i32, i32) -> ()
    %39 = llvm.add %21, %17 : i64
    llvm.br ^bb1(%39 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}

