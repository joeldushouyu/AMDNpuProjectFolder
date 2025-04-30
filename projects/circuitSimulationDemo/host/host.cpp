#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include "debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils.hpp"
#include "vm_args.hpp"
#include "utils.hpp"
#include "mvm_sequence.hpp"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"
#include <nlohmann/json.hpp> // Include the nlohmann/json header
using json = nlohmann::json;
namespace po = boost::program_options;
#include <vector>
#include <cmath>    // For std::fabs, std::max
#include <algorithm> // For std::max

#include "circuitConfig.h"

using int32 = std::int32_t;
bool are_results_close(
    buffer<float>& y_cpu,
    buffer<float>& y_npu,
    float rtol = 1e-5f,  // Relative tolerance (1e-5 = 0.001%)
    float atol = 1e-6f,   // Absolute tolerance (1e-6 = 0.0001%),
    uint32_t size_to_check = 0
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
buffer<float> test_func(){
    buffer<float> w(100);
    for (int i = 0; i < 100; i++){
        w[i] = i;
    }
    return w;
}


// Helper function to convert a single matrix slice from row-major to column-major
static void convertMatrix(
    const buffer<float> &in,
    buffer<float> &out,
    std::size_t in_offset,
    std::size_t &out_offset,
    int32 rows,
    int32 cols
) {
    for (int32 c = 0; c < cols; ++c) {
        for (int32 r = 0; r < rows; ++r) {
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
    int32 diode_size,
    int32 state_size,
    int32 input_size,
    int32 output_size,
    int32 switch_size
) {
    const int32 m1_rows = 3 * diode_size;
    const int32 m1_cols = state_size + input_size;
    const int32 m2_rows = state_size + 2 * output_size;
    const int32 m2_cols = state_size + input_size;

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
    for (int32 s = 0; s < switch_size; ++s) {
        convertMatrix(in, out, in_offset, out_offset, m1_rows, m1_cols);
        in_offset += mat1_elems;
    }

    // Process second block matrices
    for (int32 s = 0; s < switch_size; ++s) {
        convertMatrix(in, out, in_offset, out_offset, m2_rows, m2_cols);
        in_offset += mat2_elems;
    }

    return out;
}

void print_matrices_side_by_side(
    const buffer<float>& row_major,
    const buffer<float>& col_major,
    int32 rows,
    int32 cols
) {
    for (int32 r = 0; r < rows; ++r) {
        // Print row-major row
        for (int32 c = 0; c < cols; ++c) {
            printf("%8.3f ", row_major[r * cols + c]);
        }

        printf("    ||    ");

        // Print column-major row
        for (int32 c = 0; c < cols; ++c) {
            printf("%8.3f ", col_major[c * rows + r]);
        }

        printf("\n");
    }
}

static void debug_print_reorder(
    const buffer<float>& row_buf,
    const buffer<float>& col_buf,
    std::size_t row_off,
    std::size_t col_off,
    int        rows,
    int        cols,
    float      eps = 1e-6f    // tolerance
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
    int diode_size,
    int state_size,
    int input_size,
    int output_size,
    int switch_size
) {
    int m1r = 3*diode_size,        m1c = state_size + input_size;
    int m2r = state_size + 2*output_size, 
        m2c = state_size + input_size;
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

void linear(buffer<dtype_out>& y, buffer<dtype_in>& w, buffer<dtype_in>& x);

int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);





    // std::ifstream file("final_config.json");
    // if (!file.is_open()) {
    //     std::cerr << "Error: Could not open file 'final_config.json'" << std::endl;
    //     return 1;
    // }

    // // 2. Read the JSON data from the file
    // json data;
    // try {
    //     file >> data; // Use stream extraction to read JSON
    // }
    // catch (json::parse_error& e) {
    //     std::cerr << "Error: Parse error - " << e.what() << std::endl;
    //     return 1;
    // }
    // file.close();

    // int TRACE_SIZE, STATE_SIZE, U_SIZE, Y_SIZE;
    // int DIODE_SIZE, SWITCH_SIZE;
    // int C1_DSW_ROW_SIZE, C1_DSW_COL_SIZE, C1_DSW_MATRIX_SIZE, C1_DSW_BUFFER_SIZE;
    // int A_B_C_D_ROW_SIZE, A_B_C_D_COL_SIZE, A_B_C_D_MATRIX_SIZE, A_B_C_D_BUFFER_SIZE;
    // int INPUT_SWITCH_SIZE, INPUT_SIZE;
    // int ITERATION_STEP_PER_PING_PONG_BUFFER;
    // int BUFFER_SIZE_OF_IN_PING_POING, BUFFER_SIZE_OF_OUT_PING_PONG;
    // int PING_PONG_BUFFER_ITERATION;
    
    // try {
    //     trace_size = data["trace_size"];
    //     state_size = data["state_size"];
    //     u_size = data["u_size"];
    //     y_size = data["y_size"];
    //     diode_size = data["diode_size"];
    //     switch_size = data["switch_size"];
    //     C1_DSW_row_size = data["C1_DSW_row_size"];
    //     C1_DSW_col_size = data["C1_DSW_col_size"];
    //     C1_DSW_matrix_size = data["C1_DSW_matrix_size"];
    //     C1_DSW_buffer_size = data["C1_DSW_buffer_size"];
    //     A_B_C_D_row_size = data["A_B_C_D_row_size"];
    //     A_B_C_D_col_size = data["A_B_C_D_col_size"];
    //     A_B_C_D_matrix_size = data["A_B_C_D_matrix_size"];
    //     A_B_C_D_buffer_size = data["A_B_C_D_buffer_size"];
    //     input_switch_size = data["input_switch_size"];
    //     input_size = data["input_size"];
    //     iteration_step_per_ping_pong_buffer = data["iteration_step_per_ping_pong_buffer"];
    //     buffer_size_of_in_ping_poing = data["buffer_size_of_in_ping_poing"];
    //     buffer_size_of_out_ping_pong = data["buffer_size_of_out_ping_pong"];
    //     ping_pong_buffer_iteration = data["ping_pong_buffer_iteration"];
    // }
    // catch (json::out_of_range& e) {
    //     std::cerr << "Error: Key not found - " << e.what() << std::endl;
    //     return 1;
    // }
    // catch (json::type_error& e){
    //     std::cerr << "Error: Type error - " << e.what() << std::endl;
    //     return 1;
    // }
    





    int in_size = C1_DSW_BUFFER_SIZE  +A_B_C_D_BUFFER_SIZE;
    int Iterations = 1; // NOTE: only can run one time due to matrix balance transfer on s2mm
                    // once transfer matrix, the s2mm-1 will never go back to transfer matrix mode

    // NPU instance
    npu_app npu_instance(1);
    if (VERBOSE >= 1){
        npu_instance.get_npu_power(true);
        npu_instance.print_npu_info();
    }

    accel_user_desc accel_desc_0 = {
        .xclbin_name = "build/xclbins/mv.xclbin",
        .instr_seq = npu_sequence("build/insts/mv.txt", true),
    };


    int app_id_0 = npu_instance.register_accel_app(accel_desc_0);

    npu_instance.interperate_bd(0);
    // npu_instance.interperate_bd(1);

    // compare the two sequences
    int input_iteration_size = PING_PONG_BUFFER_ITERATION * ITERATION_STEP_PER_PING_PONG_BUFFER * INPUT_SIZE;
    int output_iteration_size = PING_PONG_BUFFER_ITERATION * ITERATION_STEP_PER_PING_PONG_BUFFER * Y_SIZE;
    buffer<int32_t> seq_0 = accel_desc_0.instr_seq.to_bo().cast_to<int32_t>();
    buffer<dtype_in> w_0 = npu_instance.create_bo_buffer<dtype_in>(in_size, 3, app_id_0);
    buffer<dtype_out> y_0 = npu_instance.create_bo_buffer<dtype_out>(in_size, 4, app_id_0);
    buffer<dtype_in> in_0 = npu_instance.create_bo_buffer<dtype_in>( input_iteration_size, 5, app_id_0);
    buffer<dtype_out> out_0 = npu_instance.create_bo_buffer<dtype_out>(output_iteration_size, 6, app_id_0);

    int tmp_trace_size = (TRACE_SIZE > 0) ? TRACE_SIZE :1;
    buffer<char> trace_res = npu_instance.create_bo_buffer<char>(tmp_trace_size,7, app_id_0 );


    // random float, not in in this case
    std::random_device rd;
    std::mt19937                  gen(rd());
    std::uniform_real_distribution<float> dist(-500.123f, 1000.12333f);  
    buffer<dtype_out> out_ref_0(output_iteration_size);
    for (int i = 0; i < in_size; i++){
        w_0[i] = i;//dist(gen);

    }

    // answer 
    
    // transform y_ref_0 to colum major
    buffer<float> y_ref_col = transform_to_column_major_order( w_0, DIODE_SIZE, STATE_SIZE, U_SIZE, Y_SIZE, std::pow(2, SWITCH_SIZE + DIODE_SIZE) );

    uint32_t C1_SWD_matrix_index = 0;


    for (int i = 0, n = std::min(input_iteration_size, 16); i < n; ++i) {
        // build the 32‐bit pattern we want
        uint32_t bits = static_cast<uint32_t>(i);
        // bit_cast that into a float (so its bit‐pattern becomes exactly 'bits')
        in_0[i] = std::bit_cast<float>(bits);
    }
  
    

    w_0.sync_to_device();
    in_0.sync_to_device();
    char *bufTrace = trace_res.data();
    if(TRACE_SIZE>0){
        memset(bufTrace, 0, TRACE_SIZE);
        trace_res.sync_to_device();
    }

    int C1_SWD_debug_mat_count = 0;
    for(int i = 0; i <   ITERATION_STEP_PER_PING_PONG_BUFFER* PING_PONG_BUFFER_ITERATION; i++){
        if(i  <  C1_DSW_BUFFER_SIZE){
            out_ref_0[i] = y_ref_col[i];
        }
        else{
            out_ref_0[i ] = 0;
        }
    }

    // // generate out_ref_0
    // int32_t in_offset = 0;
    // int32_t out_offset = 0; 
    // for(int i = 0; i < iteration_step_per_ping_pong_buffer* ping_pong_buffer_iteration; i++){
    //     float acc = 0;
    //     for(int k = 0; k < input_size;k ++){
    //         acc += in_0[in_offset];
    //         in_offset++;
    //     }
    //     for(int l = 0; l < y_size; l++ ){
    //         out_ref_0[out_offset] = acc;
    //         out_offset++;
    //     }

    // }


    auto run_0 = npu_instance.create_run(app_id_0, w_0.bo(), y_0.bo(), in_0.bo(), out_0.bo(), trace_res.bo() );

	
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call.");
    time_utils::time_with_unit npu_time = {0.0, "us"};
	
    for (int i = 0; i < Iterations; i++) {
        time_utils::time_point start = time_utils::now();
        run_0.start();
        run_0.wait();

	    time_utils::time_point stop = time_utils::now();
	    npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= Iterations * 2.0;
    MSG_BONDLINE(40);
    MSG_BOX_LINE(40, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(40);

    y_0.sync_from_device();    
    out_0.sync_from_device();
    if(TRACE_SIZE > 0){
        trace_res.sync_from_device();
        npu_instance.write_out_trace(((char *)bufTrace), TRACE_SIZE,
        "trace.txt");
    }

    header_print("info", "Finished running kernel");




    bool pass = are_results_close(y_0, y_ref_col, 1e-4f, 1e-3f);
    // for (size_t i = 0; i < y_0.size(); i++) {
    //     std::cout << std::scientific      // Use exponential notation
    //               << std::setprecision(6) // Show 2 digits after decimal
    //               << "y[" << i << "] = " << y_0[i]
    //               << " ?= y_ref[" << i << "] = " << y_ref_0[i]
    //               << std::endl;
    // }

    // debug_inspect_all(
    //     y_ref_0, y_0,
    //     diode_size, state_size, u_size, output_size,
    //     int(std::pow(2, switch_diode_size))
    // );

    if (pass ==false){
        std::cout <<"Fail stage 1" << std::endl;
    }
    pass &= are_results_close( out_0, out_ref_0,1e-4f, 1e-3f, C1_DSW_BUFFER_SIZE );
    if(pass==false){
        std::cout << "FAil stage2" <<std::endl;
    }
    for (size_t i = 0; i < C1_DSW_BUFFER_SIZE; i++) {
        std::cout << std::scientific      // Use exponential notation
                  << std::setprecision(6) // Show 2 digits after decimal
                  << "out_0[" << i << "] = " << out_0[i]
                  << " ?= out_ref_0[" << i << "] = " << out_ref_0[i]
                  << std::endl;
    }

    // // run with runlist
    // xrt::runlist runlist = npu_instance.create_runlist(app_id_0);
    // y_0.memset(0);
    // for (int i = 0; i < Iterations; i++){
    //     xrt::run run_0 = npu_instance.create_run(app_id_0, w_0.bo(), x_0.bo(), y_0.bo());
    //     runlist.add(run_0);
    // }
    
    // npu_time.first = 0;

    // {
    //     time_utils::time_point start = time_utils::now();
    //     runlist.execute();
    //     runlist.wait();
    //     time_utils::time_point stop = time_utils::now();
    //     npu_time.first += time_utils::duration_us(start, stop).first;
    // }
    // npu_time.first /= Iterations * 2.0;
    // MSG_BONDLINE(40);
    // MSG_BOX_LINE(40, "NPU time with runlist: " << npu_time.first << " us");
    // MSG_BONDLINE(40);
    // y_0.sync_from_device();    

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }

    // utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N), 1000);
    return 0;
}

void linear(buffer<dtype_out>& y, buffer<dtype_in>& w, buffer<dtype_in>& x){
    int in_features = x.size();
    int out_features = y.size();
    assert((in_features * out_features) == w.size());
    int v = 0;
    for (int row = 0; row < out_features; row++){
        dtype_out sum = 0;
        for (int col = 0; col < in_features; col++){
            sum = sum + dtype_out(w[v++] * x[col]);
        }
        y[row] = sum;
    }
}
