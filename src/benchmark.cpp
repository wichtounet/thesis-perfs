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

namespace {

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

} //end of anonymous namespace

namespace standard = etl::impl::standard;
namespace sse = etl::impl::sse;
namespace avx = etl::impl::avx;
namespace blas = etl::impl::blas;

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid", conv2_valid_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ standard::conv2_valid(a, b, r); }),
    CPM_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ sse::conv2_valid(a.direct(), b.direct(), r.direct()); }),
    CPM_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ avx::conv2_valid(a.direct(), b.direct(), r.direct()); }),
    CPM_SECTION_FUNCTOR("mmul", [](smat& a, smat& b, smat& r){ blas_conv2_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv2_valid", conv2_valid_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ standard::conv2_valid(a, b, r); }),
    CPM_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ sse::conv2_valid(a.direct(), b.direct(), r.direct()); }),
    CPM_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ avx::conv2_valid(a.direct(), b.direct(), r.direct()); }),
    CPM_SECTION_FUNCTOR("mmul", [](dmat& a, dmat& b, dmat& r){ blas_conv2_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_full", conv2_full_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ standard::conv2_full(a, b, r); }),
    CPM_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ sse::conv2_full(a.direct(), b.direct(), r.direct()); }),
    CPM_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ avx::conv2_full(a.direct(), b.direct(), r.direct()); }),
    CPM_SECTION_FUNCTOR("fft_mkl", [](smat& a, smat& b, smat& r){ blas::conv2_full(a.direct(), b.direct(), r.direct()); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv2_full", conv2_full_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ standard::conv2_full(a, b, r); }),
    CPM_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ sse::conv2_full(a.direct(), b.direct(), r.direct()); }),
    CPM_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ avx::conv2_full(a.direct(), b.direct(), r.direct()); }),
    CPM_SECTION_FUNCTOR("fft_mkl", [](dmat& a, dmat& b, dmat& r){ blas::conv2_full(a.direct(), b.direct(), r.direct()); })
)
