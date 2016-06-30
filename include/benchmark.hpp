//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iostream>
#include <chrono>
#include <random>

#include "etl/etl.hpp"

#define CPM_NO_RANDOMIZATION            //Randomly initialize only once
#define CPM_AUTO_STEPS                  //Enable steps estimation system
#define CPM_STEP_ESTIMATION_MIN 0.03   //Run during 0.03 seconds for estimating steps
#define CPM_RUNTIME_TARGET 1.0          //Run each test during 1.0 seconds

#include "cpm/cpm.hpp"

using dvec = etl::dyn_vector<double>;
using dmat = etl::dyn_matrix<double>;
using dmat2 = etl::dyn_matrix<double, 2>;
using dmat3 = etl::dyn_matrix<double, 3>;
using dmat4 = etl::dyn_matrix<double, 4>;
using dmat5 = etl::dyn_matrix<double, 5>;

using svec = etl::dyn_vector<float>;
using smat = etl::dyn_matrix<float>;
using smat3 = etl::dyn_matrix<float, 3>;
using smat4 = etl::dyn_matrix<float, 4>;
