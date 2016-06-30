//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_BENCHMARK "Thesis"
#include "benchmark.hpp"

using conv2_valid_policy = NARY_POLICY(
    VALUES_POLICY(12, 16, 16, 28, 50, 128, 128, 256),
    VALUES_POLICY(5, 5, 9, 9, 17, 17, 31, 31));

using conv2_full_policy = NARY_POLICY(
    VALUES_POLICY(12, 16, 16, 28, 50, 128, 128, 256),
    VALUES_POLICY(5, 5, 9, 9, 17, 17, 31, 31));

// TODO Integrate mmul reduction

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid", conv2_valid_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_valid(a, b)); }),
    CPM_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_valid(a, b)); }),
    CPM_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_valid(a, b)); }),
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_full", conv2_full_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_full(a, b)); }),
    CPM_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_full(a, b)); }),
    CPM_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_full(a, b)); }),
    CPM_SECTION_FUNCTOR("fft_mkl", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_2d_full(a, b)); })
)
