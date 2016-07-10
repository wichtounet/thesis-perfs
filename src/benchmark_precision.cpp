//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <random>

#include "etl/etl.hpp"

namespace {

namespace standard = etl::impl::standard;
namespace sse = etl::impl::sse;
namespace avx = etl::impl::avx;
namespace blas = etl::impl::blas;
namespace reduc = etl::impl::reduc;

template <typename I, typename K_T, typename C>
void blas_conv2_valid(const I& input, const K_T& kernel, C&& conv) {
    const std::size_t v1 = etl::dim<0>(input);
    const std::size_t v2 = etl::dim<1>(input);
    const std::size_t k1 = etl::dim<0>(kernel);
    const std::size_t k2 = etl::dim<1>(kernel);
    const std::size_t f1 = etl::dim<1>(conv);
    const std::size_t f2 = etl::dim<2>(conv);

    auto prepared_k = force_temporary(kernel);

    prepared_k.fflip_inplace();

    etl::dyn_matrix<etl::value_t<I>, 2> input_col(k1 * k2, (v1 - k1 + 1) * (v2 - k2 + 1));
    im2col_direct_tr(input_col, input, k1, k2);

    etl::reshape(conv, f1 * f2) = mul(etl::reshape(prepared_k, k1 * k2), input_col);
}

template<typename T>
void conv2_full_precision(){
    size_t n = 100;

    auto k = n / 4;
    auto c = n + k - 1;

    etl::dyn_matrix<T> input(n, n);
    etl::dyn_matrix<T> kernel(k, k);

    input = etl::normal_generator<T>(5.0, 1000.0);
    kernel = etl::normal_generator<T>(10.0, 2000.0);

    etl::dyn_matrix<T> ref(c,c);
    etl::dyn_matrix<T> ref_sse(c,c);
    etl::dyn_matrix<T> ref_avx(c,c);
    etl::dyn_matrix<T> ref_blas(c,c);

    standard::conv2_full(input, kernel, ref);
    sse::conv2_full(input.direct(), kernel.direct(), ref_sse.direct());
    avx::conv2_full(input.direct(), kernel.direct(), ref_avx.direct());
    blas::conv2_full(input.direct(), kernel.direct(), ref_blas.direct());

    std::cout << "Reference Average: " << etl::mean(ref) << std::endl;

    std::cout << "SSE Average: " << etl::mean(ref_sse) << std::endl;
    std::cout << "SSE Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_sse)) << std::endl;
    std::cout << "SSE Normalized Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_sse)) / std::abs(etl::mean(ref)) << std::endl;
    std::cout << "SSE Difference Average: " << etl::mean(ref_sse - ref) << std::endl;
    std::cout << "SSE Difference Standard Deviation: " << etl::stddev(ref_sse - ref) << std::endl;

    std::cout << "AVX Average: " << etl::mean(ref_avx) << std::endl;
    std::cout << "AVX Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_avx)) << std::endl;
    std::cout << "AVX Normalized Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_avx)) / std::abs(etl::mean(ref)) << std::endl;
    std::cout << "AVX Difference Average: " << etl::mean(ref_avx - ref) << std::endl;
    std::cout << "AVX Difference Standard Deviation: " << etl::stddev(ref_avx - ref) << std::endl;

    std::cout << "BLAS Average: " << etl::mean(ref_blas) << std::endl;
    std::cout << "BLAS Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_blas)) << std::endl;
    std::cout << "BLAS Normalized Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_blas)) / std::abs(etl::mean(ref)) << std::endl;
    std::cout << "BLAS Difference Average: " << etl::mean(ref_blas - ref) << std::endl;
    std::cout << "BLAS Difference Standard Deviation: " << etl::stddev(ref_blas - ref) << std::endl;
}

template<typename T>
void conv2_valid_precision(){
    size_t n = 100;

    auto k = n / 4;
    auto c = n - k + 1;

    etl::dyn_matrix<T> input(n, n);
    etl::dyn_matrix<T> kernel(k, k);

    input = etl::normal_generator<T>(5.0, 1000.0);
    kernel = etl::normal_generator<T>(10.0, 2000.0);

    etl::dyn_matrix<T> ref(c,c);
    etl::dyn_matrix<T> ref_sse(c,c);
    etl::dyn_matrix<T> ref_avx(c,c);
    etl::dyn_matrix<T> ref_blas(c,c);

    standard::conv2_valid(input, kernel, ref);
    sse::conv2_valid(input.direct(), kernel.direct(), ref_sse.direct());
    avx::conv2_valid(input.direct(), kernel.direct(), ref_avx.direct());
    blas_conv2_valid(input, kernel, ref_blas);

    std::cout << "Reference Average: " << etl::mean(ref) << std::endl;

    std::cout << "SSE Average: " << etl::mean(ref_sse) << std::endl;
    std::cout << "SSE Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_sse)) << std::endl;
    std::cout << "SSE Normalized Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_sse)) / std::abs(etl::mean(ref)) << std::endl;
    std::cout << "SSE Difference Average: " << etl::mean(ref_sse - ref) << std::endl;
    std::cout << "SSE Difference Standard Deviation: " << etl::stddev(ref_sse - ref) << std::endl;

    std::cout << "AVX Average: " << etl::mean(ref_avx) << std::endl;
    std::cout << "AVX Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_avx)) << std::endl;
    std::cout << "AVX Normalized Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_avx)) / std::abs(etl::mean(ref)) << std::endl;
    std::cout << "AVX Difference Average: " << etl::mean(ref_avx - ref) << std::endl;
    std::cout << "AVX Difference Standard Deviation: " << etl::stddev(ref_avx - ref) << std::endl;

    std::cout << "BLAS Average: " << etl::mean(ref_blas) << std::endl;
    std::cout << "BLAS Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_blas)) << std::endl;
    std::cout << "BLAS Normalized Average Difference: " << std::abs(etl::mean(ref) - etl::mean(ref_blas)) / std::abs(etl::mean(ref)) << std::endl;
    std::cout << "BLAS Difference Average: " << etl::mean(ref_blas - ref) << std::endl;
    std::cout << "BLAS Difference Standard Deviation: " << etl::stddev(ref_blas - ref) << std::endl;
}

} //end of anonymous namespace

int main(){
    std::cout << std::endl << "Single precision benchmark Full" << std::endl;
    conv2_full_precision<float>();

    std::cout << std::endl << "Double precision benchmark Full" << std::endl;
    conv2_full_precision<double>();

    std::cout << std::endl << "Single precision benchmark Valid" << std::endl;
    conv2_valid_precision<float>();

    std::cout << std::endl << "Double precision benchmark Valid" << std::endl;
    conv2_valid_precision<double>();

    return 0;
}
