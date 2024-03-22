cmake_minimum_required(VERSION 3.17)
project(gmic-py C CXX)
set(CMAKE_CXX_STANDARD 11)
enable_testing()

add_subdirectory("lib/gmic" EXCLUDE_FROM_ALL)

# Find the Python development files
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Add the module to compile
Python3_add_library(gmic-py MODULE "src/gmicpy.h" "src/gmicpy.cpp" WITH_SOABI)
set_target_properties(gmic-py PROPERTIES OUTPUT_NAME "_gmic")
target_link_libraries(gmic-py PRIVATE libgmicstatic)

option(GMIC_PY_NUMPY "Build with numpy support" ON)
if (GMIC_PY_NUMPY)
    target_compile_options(gmic-py PUBLIC -Dgmic_py_numpy)
endif ()

# Install the module
install(TARGETS gmic-py
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        LIBRARY DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME})

add_test(NAME gmic-py-tests
        COMMAND ${Python3_EXECUTABLE} -m pytest # Or just COMMAND pytest
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
set_tests_properties(gmic-py-tests
        PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
