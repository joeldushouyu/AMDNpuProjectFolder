; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@_anonymous5 = external global [1 x i32]
@_anonymous4 = external global [1 x i32]
@_anonymous3 = external global [1024 x i16]
@_anonymous2 = external global [1024 x i16]
@_anonymous1 = external global [1024 x i16]
@_anonymous0 = external global [1024 x i16]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

declare void @vector_scalar_mul_int16_vector(ptr, ptr, ptr, i32)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %16, %0
  %2 = phi i64 [ %17, %16 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %18

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %9, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 4
  br i1 %7, label %8, label %10

8:                                                ; preds = %5
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 48, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous2, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous4, i64 32) ]
  call void @vector_scalar_mul_int16_vector(ptr @_anonymous2, ptr @_anonymous0, ptr @_anonymous4, i32 1024)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 49, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 48, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous3, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous4, i64 32) ]
  call void @vector_scalar_mul_int16_vector(ptr @_anonymous3, ptr @_anonymous1, ptr @_anonymous4, i32 1024)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 49, i32 1)
  %9 = add i64 %6, 2
  br label %5

10:                                               ; preds = %5
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  br label %11

11:                                               ; preds = %14, %10
  %12 = phi i64 [ %15, %14 ], [ 0, %10 ]
  %13 = icmp slt i64 %12, 4
  br i1 %13, label %14, label %16

14:                                               ; preds = %11
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 48, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous2, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous5, i64 32) ]
  call void @vector_scalar_mul_int16_vector(ptr @_anonymous2, ptr @_anonymous0, ptr @_anonymous5, i32 1024)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 49, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 48, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous3, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @_anonymous5, i64 32) ]
  call void @vector_scalar_mul_int16_vector(ptr @_anonymous3, ptr @_anonymous1, ptr @_anonymous5, i32 1024)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 49, i32 1)
  %15 = add i64 %12, 2
  br label %11

16:                                               ; preds = %11
  call void @llvm.aie2.release(i32 52, i32 1)
  %17 = add i64 %2, 2
  br label %1

18:                                               ; preds = %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
