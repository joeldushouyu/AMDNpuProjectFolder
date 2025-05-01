

#include "host_helper.hpp"

bool are_results_close(
    buffer<float>& y_cpu,
    buffer<float>& y_npu,
    float rtol ,  // Relative tolerance (1e-5 = 0.001%)
    float atol,   // Absolute tolerance (1e-6 = 0.0001%),
    uint32_t size_to_check 
) {
    // First check vector sizes match
    if (y_cpu.size() != y_npu.size()) {
        return false;
    }
    if(size_to_check == 0){
        size_to_check =  y_cpu.size();
    }

    // Component-wise comparison
    for (size_t i = 0; i < size_to_check; ++i) {
        const float a = y_cpu[i];
        const float b = y_npu[i];
        const float abs_diff = std::fabs(a - b);

        // Calculate acceptable tolerance threshold
        const float threshold = atol + rtol * std::max(std::fabs(a), std::fabs(b));

        // Early exit if any element fails
        if (abs_diff > threshold) {
            std::cout << "Failed at index" << i << std::endl;
            return false;
        }
    }

    return true;
}








// Helper function to convert a single matrix slice from row-major to column-major
void convertMatrix(
    const buffer<float> &in,
    buffer<float> &out,
    std::size_t in_offset,
    std::size_t &out_offset,
    std::int32_t rows,
    std::int32_t cols
) {
    for (std::int32_t c = 0; c < cols; ++c) {
        for (std::int32_t r = 0; r < rows; ++r) {
            // Correct index: row-major input to column-major output
            out[out_offset++] = in[in_offset + r * cols + c];
        }
    }
}
// Transforms a flattened array of matrices from row-major to column-major order.
// Structure of `in`:
// 1) First block: switch_size repetitions of matrices of size
//    (3 * diode_size) rows x (state_size + input_size) cols.
// 2) Second block: switch_size repetitions of matrices of size
//    (state_size + 2 * output_size) rows x (state_size + input_size) cols.
// All matrices in `in` are stored in row-major order.
// This function returns a new buffer where each individual matrix
// has been converted into column-major order.
buffer<float> transform_to_column_major_order(
    const buffer<float> &in,
    std::int32_t switch_size
) {
    const std::int32_t m1_rows =  C1_DSW_ROW_SIZE;
    const std::int32_t m1_cols = C1_DSW_COL_SIZE;
    const std::int32_t m2_rows = A_B_C_D_ROW_SIZE;
    const std::int32_t m2_cols = A_B_C_D_COL_SIZE;

    const std::size_t mat1_elems = static_cast<std::size_t>(m1_rows) * m1_cols;
    const std::size_t mat2_elems = static_cast<std::size_t>(m2_rows) * m2_cols;
    const std::size_t expected_size = static_cast<std::size_t>(switch_size) * (mat1_elems + mat2_elems);

    if (in.size() != expected_size) {
        throw std::invalid_argument("Input buffer size does not match expected dimensions");
    }

    buffer<float> out(expected_size);
    std::size_t in_offset = 0;
    std::size_t out_offset = 0;

    // Process first block matrices
    for (std::int32_t s = 0; s < switch_size; ++s) {
        convertMatrix(in, out, in_offset, out_offset, m1_rows, m1_cols);
        in_offset += mat1_elems;
    }

    // Process second block matrices
    for (std::int32_t s = 0; s < switch_size; ++s) {
        convertMatrix(in, out, in_offset, out_offset, m2_rows, m2_cols);
        in_offset += mat2_elems;
    }

    return out;
}






 void debug_print_reorder(
    const buffer<float>& row_buf,
    const buffer<float>& col_buf,
    std::size_t row_off,
    std::size_t col_off,
    int        rows,
    int        cols,
    float      eps     // tolerance
) {
    assert(row_buf.size() >= row_off + std::size_t(rows)*cols);
    assert(col_buf.size() >= col_off + std::size_t(rows)*cols);

    printf(" (r,c) | orig_idx →  val  || new_idx →  val   | diff    \n");
    printf("-------+------------------++-------------------+---------\n");
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::size_t orig_idx = row_off + r*cols + c;
            std::size_t new_idx  = col_off + c*rows + r;
            float a = row_buf[orig_idx];
            float b = col_buf[new_idx];
            float diff = std::fabs(a - b);
            bool  bad  = diff > eps;

            printf(" (%2d,%2d) | %4zu → %8.3f || %4zu → %8.3f | %8.3e %s\n",
                   r, c,
                   orig_idx, a,
                   new_idx,  b,
                   diff,
                   bad ? "<-- mismatch" : "");
        }
    }
    printf("\n");
}

// ----------------------------------------------------------------------------
// Walk through *all* your slices in the two blocks and call debug_print_reorder
// ----------------------------------------------------------------------------
void debug_inspect_all(
    const buffer<float>& row_buf,
    const buffer<float>& col_buf,
    int switch_size
) {
    int m1r = C1_DSW_ROW_SIZE,        m1c = C1_DSW_COL_SIZE;
    int m2r = A_B_C_D_ROW_SIZE, 
        m2c = A_B_C_D_COL_SIZE;
    std::size_t elems1 = std::size_t(m1r)*m1c;
    std::size_t elems2 = std::size_t(m2r)*m2c;

    std::size_t row_off = 0, col_off = 0;
    for (int s = 0; s < switch_size; ++s) {
        printf("=== slice %d, BLOCK 1 (%dx%d) ===\n", s, m1r, m1c);
        debug_print_reorder(row_buf, col_buf, row_off, col_off, m1r, m1c);
        row_off += elems1;
        col_off += elems1;
    }
    for (int s = 0; s < switch_size; ++s) {
        printf("=== slice %d, BLOCK 2 (%dx%d) ===\n", s, m2r, m2c);
        debug_print_reorder(row_buf, col_buf, row_off, col_off, m2r, m2c);
        row_off += elems2;
        col_off += elems2;
    }
}






// MV, matrix is in column major 

// Performs y = A * x
// A: m x n matrix stored in column-major order
// x: vector of size n
// Returns: vector y of size m
std::vector<float> matvec_mul_col_major(
    float * A, // matrix A in column-major order
    float * x, // input vector x
   size_t m,                    // number of rows
   size_t n                     // number of columns
) {
    // assert(A.size() == m * n);
    // assert(x.size() == n);

    std::vector<float> y(m, 0.0f);

    // Iterate over columns
    for (size_t col = 0; col < n; ++col) {
        float x_val = *(x+col);
        for (size_t row = 0; row < m; ++row) {
            y[row] += A[col * m + row] * x_val;
        }
    }

    return y;
}