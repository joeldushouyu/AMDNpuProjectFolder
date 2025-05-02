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


bool are_results_close(
    buffer<float>& y_cpu,
    buffer<float>& y_npu,
    float rtol = 1e-5f,  // Relative tolerance (1e-5 = 0.001%)
    float atol = 1e-6f   // Absolute tolerance (1e-6 = 0.0001%)
) {
    // First check vector sizes match
    if (y_cpu.size() != y_npu.size()) {
        return false;
    }

    // Component-wise comparison
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        const float a = y_cpu[i];
        const float b = y_npu[i];
        const float abs_diff = std::fabs(a - b);

        // Calculate acceptable tolerance threshold
        const float threshold = atol + rtol * std::max(std::fabs(a), std::fabs(b));

        // Early exit if any element fails
        if (abs_diff > threshold) {
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

void linear(buffer<dtype_out>& y, buffer<dtype_in>& w, buffer<dtype_in>& x);

int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("M,m", po::value<int>()->default_value(64 ), "M");
    desc.add_options()("K,k", po::value<int>()->default_value(256), "K");
    desc.add_options()("I,i", po::value<int>()->default_value(1), "Iterations");

    arg_utils::parse_options(argc, argv, desc, vm);
    
    // User logic
    int M = vm["M"].as<int>();
    int K = vm["K"].as<int>();
    int Iterations =1; // vm["I"].as<int>();
    int N = 1;
    int Y_VOLUME = M * 1;
    int W_VOLUME = M * K;
    int X_VOLUME = 1 * K;

    int trace_size = 8192;

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


    header_print("info", "Matrix size " << M << "x" << K << "x" << N);

    int app_id_0 = npu_instance.register_accel_app(accel_desc_0);

    npu_instance.interperate_bd(0);
    // npu_instance.interperate_bd(1);

    // compare the two sequences
    buffer<int32_t> seq_0 = accel_desc_0.instr_seq.to_bo().cast_to<int32_t>();


    
    buffer<dtype_in> w_0 = npu_instance.create_bo_buffer<dtype_in>(W_VOLUME, 3, app_id_0);
    buffer<dtype_in> x_0 = npu_instance.create_bo_buffer<dtype_in>(X_VOLUME, 4, app_id_0);
    buffer<dtype_out> y_0 = npu_instance.create_bo_buffer<dtype_out>(Y_VOLUME, 5, app_id_0);

    int tmp_trace_size  = (trace_size > 0)? trace_size: 1;
    buffer<char> trace_res = npu_instance.create_bo_buffer<char>(tmp_trace_size, 6, app_id_0);

    

    // random float, not in in this case
    std::random_device rd;
    std::mt19937                  gen(rd());
    std::uniform_real_distribution<float> dist(-500.123f, 1000.12333f);

    for (int i = 0; i < W_VOLUME; i++){
        w_0[i] = dist(gen);
    }

    for (int i = 0; i < X_VOLUME; i++){
        x_0[i] = dist(gen);
    }
    x_0[0] = 1.23334;
    // for (int i = 0; i < W_VOLUME; i++){
    //     w_0[i] = utils::getRandInt(-10, 10);
    // }

    // for (int i = 0; i < X_VOLUME; i++){
    //     x_0[i] = utils::getRandInt(-10, 10);
    // }
 
    header_print("info", "Calculate reference for " << M << "x" << K << "x" << N);
    buffer<dtype_out> y_ref_0(Y_VOLUME);    
    buffer<dtype_out> y_ref_1(Y_VOLUME);    
    linear(y_ref_0, w_0, x_0);

    char *bufTrace = trace_res.data();

    w_0.sync_to_device();

    x_0.sync_to_device();
    if(trace_size > 0){ 
        memset(bufTrace, 0, trace_size); 
        trace_res.sync_to_device();  
    }
    auto run_0 = npu_instance.create_run(app_id_0, w_0.bo(), x_0.bo(), y_0.bo(), trace_res.bo());
	
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
    if(trace_size > 0){
        trace_res.sync_from_device();
        npu_instance.write_out_trace(((char *)bufTrace), trace_size,
        "trace.txt");
    }

    header_print("info", "Finished running kernel");

    // bool pass = true;
    // if (utils::compare_vectors(y_0, y_ref_0,1, 1e-3) > 0){
    //     pass = false;
    // }

    bool pass = are_results_close(y_0, y_ref_0, 1e-4f, 1e-3f);
    for (size_t i = 0; i < y_0.size(); i++) {
        std::cout << std::scientific      // Use exponential notation
                  << std::setprecision(6) // Show 2 digits after decimal
                  << "y[" << i << "] = " << y_0[i]
                  << " ?= y_ref[" << i << "] = " << y_ref_0[i]
                  << std::endl;
    }
    // run with runlist
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
