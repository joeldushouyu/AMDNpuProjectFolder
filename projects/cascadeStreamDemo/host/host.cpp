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

namespace po = boost::program_options;
#include <vector>
#include <cmath>    // For std::fabs, std::max
#include <algorithm> // For std::max

template<typename T>
bool are_results_close(
    buffer<T>& y_cpu,
    buffer<T>& y_npu,
    float rtol = 1e-5f,  // Relative tolerance (default for float comparisons)
    float atol = 1e-6f   // Absolute tolerance (default for float comparisons)
) {
    // First check vector sizes match
    if (y_cpu.size() != y_npu.size()) {
        return false;
    }

    // Component-wise comparison
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        const T a = y_cpu[i];
        const T b = y_npu[i];
        const float abs_diff = std::abs(a - b);

        // Calculate acceptable tolerance threshold
        const float threshold = atol + rtol * std::max(std::abs(a), std::abs(b));

        // Early exit if any element fails
        if (abs_diff > threshold) {
            return false;
        }
    }

    return true;
}
template<typename T>
buffer<T> elementwise_product(buffer<T>& a, buffer<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length.");
    }

    buffer<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}



void linear(buffer<dtype_out>& y, buffer<dtype_in>& w, buffer<dtype_in>& x);

int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("V,v", po::value<int>()->default_value(512 ), "M");
    desc.add_options()("T,t", po::value<int>()->default_value(3072), "K");
    desc.add_options()("I,i", po::value<int>()->default_value(1), "Iterations");

    arg_utils::parse_options(argc, argv, desc, vm);
    
    // User logic
    int vectorSize = vm["V"].as<int>();
    int trace_size = vm["T"].as<int>();
    int Iterations = vm["I"].as<int>();
    // int N = 1;
    // int Y_VOLUME = M * 1;
    // int W_VOLUME = M * K;
    // int X_VOLUME = 1 * K;
    // int Iterations = 10;
    // int vectorSize = 1024;
    // int trace_size  = 4096;

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


    buffer<dtype_in> w_0 = npu_instance.create_bo_buffer<dtype_in>(vectorSize, 3, app_id_0);
    buffer<dtype_in> w_1 = npu_instance.create_bo_buffer<dtype_in>(vectorSize, 4, app_id_0);
    buffer<dtype_out> y_1 = npu_instance.create_bo_buffer<dtype_out>(vectorSize, 5, app_id_0);


    int tmp_trace_size  = (trace_size > 0)? trace_size: 1;
    buffer<char> trace_res = npu_instance.create_bo_buffer<char>(tmp_trace_size, 6, app_id_0);

    


    // DATA initalizatio
    std::random_device rd;
    std::mt19937                  gen(rd());
    std::uniform_real_distribution<float> dist(-500.123f, 1000.12333f);

    for (int i = 0; i < vectorSize; i++){
        w_0[i] = dist(gen);
        w_1[i] =dist(gen);
        
    }
    w_0[2] = 0.0;
    buffer<dtype_out> y_ref = elementwise_product(w_0, w_1);
    char *bufTrace = trace_res.data();

    w_0.sync_to_device();
    w_1.sync_to_device();
    if(trace_size > 0){ 
        
        memset(bufTrace, 0, trace_size); 
        trace_res.sync_to_device();  
    }
    
    // if(trace_size>0){
    auto run_0 = npu_instance.create_run(app_id_0, w_0.bo(), w_1.bo(), y_1.bo(),  trace_res.bo());
    // }else{
    //     auto run_0 = npu_instance.create_run(app_id_0, w_0.bo(), w_1.bo(), y_1.bo());
    // }
    


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


    y_1.sync_from_device();   
    if(trace_size > 0){
        trace_res.sync_from_device();
        npu_instance.write_out_trace(((char *)bufTrace), trace_size,
        "trace.txt");
    }

    
    header_print("info", "Finished running kernel");



    bool pass = are_results_close<dtype_out>(y_ref, y_1, 1e-4f, 1e-3f);
    // pass &= are_results_close<dtype_out>(y_1, w_1, 1e-4f, 1e-3f);
    for (size_t i = 0; i < 25; i++) { // first 25
        std::cout << std::scientific      // Use exponential notation
                  << std::setprecision(6) // Show 2 digits after decimal
                  << "y[" << i << "] = " << y_ref[i]
                  << " ?= y_ref[" << i << "] = " << y_1[i]
                  << std::endl;
    }


    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }


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
