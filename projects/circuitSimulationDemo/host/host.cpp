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

#include "host_helper.hpp"

using int32 = std::int32_t;




int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);


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
    int input_iteration_size = BUFFER_SIZE_OF_IN_PING_POING * PING_PONG_BUFFER_ITERATION ;
    int output_iteration_size = BUFFER_SIZE_OF_OUT_PING_PONG * PING_PONG_BUFFER_ITERATION ;

    buffer<int32_t> seq_0 = accel_desc_0.instr_seq.to_bo().cast_to<int32_t>();
    buffer<dtype_in> matrix_in = npu_instance.create_bo_buffer<dtype_in>(in_size, 3, app_id_0);
    buffer<dtype_out> matrix_out_col_major = npu_instance.create_bo_buffer<dtype_out>(in_size, 4, app_id_0);
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
        matrix_in[i] = i;//dist(gen);

    }

    // answer 
    
    // transform y_ref_0 to colum major
    buffer<float> matrix_out_ref_col = transform_to_column_major_order( matrix_in,  std::pow(2, SWITCH_SIZE + DIODE_SIZE) );

    uint32_t C1_SWD_matrix_index = 0;


    for (int i = 0; i < input_iteration_size; i+= INPUT_SIZE_PER_ITERATION) {

        for(int k = 0; k < INPUT_SIZE_PER_ITERATION; k++){
            if(k +1 == INPUT_SIZE_PER_ITERATION){
                        // build the 32‐bit pattern we want
                uint32_t bits = static_cast<uint32_t>(i);
                // bit_cast that into a float (so its bit‐pattern becomes exactly 'bits')
                in_0[i+k] = std::bit_cast<float>(bits);
            }else{
                in_0[i+k] = i+10;
            }
        }

    }
  
    // for(int i = 0; i < input_iteration_size; i++){
    //     in_0[i] =i;
    // }
    
    for(int i = 0; i < 10; i++){
        std::cout << in_0[i]<<std::endl;
    }


    matrix_in.sync_to_device();
    in_0.sync_to_device();
    char *bufTrace = trace_res.data();
    if(TRACE_SIZE>0){
        memset(bufTrace, 0, TRACE_SIZE);
        trace_res.sync_to_device();
    }



    // // generate out_ref_0
    // int32_t in_offset = 0;
    // int32_t out_offset = 0; 
    // for(int i = 0; i < ITERATION_STEP_PER_PING_PONG_BUFFER* PING_PONG_BUFFER_ITERATION; i++){
    //     float acc = 0;
    //     for(int k = 0; k < INPUT_SIZE_PER_ITERATION;k ++){
    //         acc += in_0[in_offset];
    //         in_offset++;
    //     }
    //     for(int l = 0; l < OUTPUT_SIZE_PER_ITERATION; l++ ){
    //         out_ref_0[out_offset] = acc;
    //         out_offset++;
    //     }
    // }

    // generate out_ref_0

    // only check partial result for now
    float* input_ptr = in_0.data();
    float* ref_res = out_ref_0.data();
    float *C1_DSW_ptr = matrix_out_ref_col.data();
    for(int i = 0; i < TOTAL_SWITCH_DIODE_STATE*2; i++){

        float x[C1_DSW_COL_SIZE] = {0};
        
        for(int l = 0; l < U_SIZE; l++){
            x[STATE_SIZE + l] = *input_ptr++;
        }

        std::vector<float>res  = matvec_mul_col_major(
            C1_DSW_ptr + (i*C1_DSW_MATRIX_SIZE),x, 
            C1_DSW_ROW_SIZE,
            C1_DSW_COL_SIZE 
        );

        // STORE the reference result
        for(auto v :res){
            *ref_res++= v;
        }
        input_ptr++; // the external switch bit that is not used for now


    }





    auto run_0 = npu_instance.create_run(app_id_0, matrix_in.bo(), matrix_out_col_major.bo(), in_0.bo(), out_0.bo(), trace_res.bo() );

	
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

    matrix_out_col_major.sync_from_device();    
    out_0.sync_from_device();
    if(TRACE_SIZE > 0){
        trace_res.sync_from_device();
        npu_instance.write_out_trace(((char *)bufTrace), TRACE_SIZE,
        "trace.txt");
    }

    header_print("info", "Finished running kernel");




    bool pass = are_results_close(matrix_out_col_major, matrix_out_ref_col, 1e-4f, 1e-3f);

    // debug_inspect_all(
    //     matrix_in, matrix_out_col_major, 
    //     std::pow(2, SWITCH_SIZE + DIODE_SIZE)
    // );

    if (pass ==false){
        std::cout <<"Fail stage 1" << std::endl;
    }
    pass &= are_results_close( out_0, out_ref_0,1e-4f, 1e-3f, 2* TOTAL_SWITCH_DIODE_STATE *  C1_DSW_COL_SIZE  );
    if(pass==false){
        std::cout << "FAil stage2" <<std::endl;
    }
    for (size_t i = 0; i < 32; i++) {
        std::cout << std::scientific      // Use exponential notation
                  << std::setprecision(6) // Show 2 digits after decimal
                  << "out_0[" << i << "] = " << out_0[i]
                  << " ?= out_ref_0[" << i << "] = " << out_ref_0[i]
                  << std::endl;
    }



    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }

    // utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N), 1000);
    return 0;
}

