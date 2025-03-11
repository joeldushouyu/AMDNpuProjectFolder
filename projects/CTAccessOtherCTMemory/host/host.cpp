
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
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"

namespace po = boost::program_options;

void callback_0(const void *data, ert_cmd_state state, void *user_data) {
    header_print("info", "Run 0 completed");
}
void callback_1(const void *data, ert_cmd_state state, void *user_data) {
    header_print("info", "Run 1 completed");
}
int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("M,m", po::value<int>()->default_value(8), "VectorSize"); // don't change it yet

    arg_utils::parse_options(argc, argv, desc, vm);
    
    // User logic
    int vectorSize = vm["M"].as<int>();
    int scalarSize = 1;

    // NPU instance
    npu_app npu_instance(1, 1, 0);
    if (VERBOSE >= 1){
        npu_instance.get_npu_power(true);
        npu_instance.print_npu_info();
    }

    accel_user_desc accel_desc_0 = {
        .xclbin_name = "build/xclbins/aie.xclbin",
        .instr_name = "build/insts/aie.txt",
    };


    int app_id_0 = npu_instance.register_accel_app(accel_desc_0);


    npu_instance.print_npu_info();

    npu_instance.list_kernels();

    npu_instance.interperate_bd(0);
    // npu_instance.interperate_bd(1); // They are the same
    
    vector w_0 = npu_instance.create_bo_vector<dtype_in>(vectorSize, 3, app_id_0);
    vector y_0 = npu_instance.create_bo_vector<dtype_out>(vectorSize, 4, app_id_0);



    for (int i = 0; i < vectorSize; i++){
        w_0[i] = utils::getRandInt(10, 100);
    }



    vector<dtype_out> y_ref(vectorSize);    
    for(size_t i = 0; i < vectorSize; i++){
        y_ref[i] = w_0[i] + 1;
    }


    w_0.sync_to_device();



    auto run_0 = npu_instance.create_run(w_0.bo(), y_0.bo(), app_id_0);
	
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call.");
    time_utils::time_with_unit npu_time = {0.0, "us"};
	
    for (int i = 0; i < 1; i++) {
        time_utils::time_point start = time_utils::now();
        run_0.start();
        run_0.wait();
	    time_utils::time_point stop = time_utils::now();
	    npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= 1000 * 2;
    MSG_BONDLINE(40);
    MSG_BOX_LINE(40, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(40);

    y_0.sync_from_device();    
    header_print("info", "Finished running kernel");

    bool pass = true;
    if (utils::compare_vectors(y_0, y_ref) > 0){
        pass = false;
    }

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }

    // utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N), 1000);
    return 0;
}
