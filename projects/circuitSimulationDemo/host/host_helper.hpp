// Auto-generated config header
#ifndef HOST_HELPPER_H
#define HOST_HELPPER_H

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


#include "typedef.hpp"
#include "npu_utils.hpp"

#include "circuitConfig.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"
#include <nlohmann/json.hpp> // Include the nlohmann/json header
using json = nlohmann::json;
namespace po = boost::program_options;
#include <vector>
#include <cmath>    // For std::fabs, std::max
#include <algorithm> // For std::max


bool are_results_close(
    buffer<float>& y_cpu,
    buffer<float>& y_npu,
    float rtol = 1e-5f,  // Relative tolerance (1e-5 = 0.001%)
    float atol = 1e-6f,   // Absolute tolerance (1e-6 = 0.0001%),
    uint32_t size_to_check = 0
) ;


// use for convert matrix from row-major to column major
void convertMatrix(
    const buffer<float> &in,
    buffer<float> &out,
    std::size_t in_offset,
    std::size_t &out_offset,
    std::int32_t rows,
    std::int32_t cols
);
buffer<float> transform_to_column_major_order(
    const buffer<float> &in,
    std::int32_t switch_size
) ;





// useful to  print out the column major matrix
void debug_print_reorder(
    const buffer<float>& row_buf,
    const buffer<float>& col_buf,
    std::size_t row_off,
    std::size_t col_off,
    int        rows,
    int        cols,
    float      eps = 1e-6f    // tolerance
) ;

void debug_inspect_all(
    const buffer<float>& row_buf,
    const buffer<float>& col_buf,
    int switch_size
);





std::vector<float> matvec_mul_col_major(
     float * A, // matrix A in column-major order
     float * x, // input vector x
    size_t m,                    // number of rows
    size_t n                     // number of columns
) ;
#endif