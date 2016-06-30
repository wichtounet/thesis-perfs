default: release

.PHONY: default release debug all clean test

include make-utils/flags.mk
include make-utils/cpp-utils.mk

# Be stricter
CXX_FLAGS += -pedantic -Werror -Winvalid-pch -Wno-uninitialized

# Add includes
CXX_FLAGS += -Ietl/lib/include -Ietl/include/ -Icpm/include

CXX_FLAGS +=-DETL_VECTORIZE_FULL -DETL_MKL_MODE $(shell pkg-config --cflags mkl)
LD_FLAGS += -pthread $(shell pkg-config --libs mkl)

ifneq (,$(ETL_PARALLEL))
CXX_FLAGS += -DETL_PARALLEL
endif

# Compile source files
$(eval $(call auto_folder_compile,src))

# Create executables
$(eval $(call auto_add_executable,bench))
$(eval $(call add_executable_set,bench,bench))

release: release_bench
release_debug: release_debug_bench
debug: debug_bench

all: release release_debug debug

run: release
	./release/bin/bench

cppcheck:
	cppcheck -I include/ --platform=unix64 --suppress=missingIncludeSystem --enable=all --std=c++11 benchmark/*.cpp workbench/*.cpp include/etl/*.hpp

# Note: workbench / benchmark is no included on purpose because of poor macro alignment
format:
	find include test -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i -style=file

# Note: test are not included on purpose (we want to force to test some operators on matrix/vector)
modernize:
	find include benchmark workbench -name "*.hpp" -o -name "*.cpp" > etl_file_list
	clang-modernize -add-override -loop-convert -pass-by-value -use-auto -use-nullptr -p ${PWD} -include-from=etl_file_list
	rm etl_file_list

# clang-tidy with some false positive checks removed
tidy:
	clang-tidy -checks='*,-llvm-include-order,-clang-analyzer-alpha.core.PointerArithm,-clang-analyzer-alpha.deadcode.UnreachableCode,-clang-analyzer-alpha.core.IdenticalExpr,-google-readability-todo' -p ${PWD} test/src/*.cpp -header-filter='include/etl/*' | tee tidy_report_light
	echo "The full report from clang-tidy is availabe in tidy_report_light"

tidy_filter:
	/usr/bin/zgrep "warning:" tidy_report_light | sort | uniq | /usr/bin/zgrep -v "\.cpp"

# clang-tidy with all the checks
tidy_all:
	clang-tidy -checks='*' -p ${PWD} test/*.cpp -header-filter='include/etl/*' | tee tidy_report_all
	echo "The full report from clang-tidy is availabe in tidy_report_all"

tidy_all_filter:
	/usr/bin/zgrep "warning:" tidy_report_all | sort | uniq | /usr/bin/zgrep -v "\.cpp"

doc:
	doxygen Doxyfile

clean: base_clean
	rm -rf reports
	rm -rf latex/ html/

include make-utils/cpp-utils-finalize.mk
