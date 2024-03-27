cmake_minimum_required(VERSION 3.17)
project(gmic-py C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
enable_testing()

add_subdirectory("lib/gmic" EXCLUDE_FROM_ALL)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Detect the installed nanobind package and import it into CMake
execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Add the module to compile
nanobind_add_module(gmic-py "src/gmicpy.hpp" "src/gmicpy.cpp")
#Python_add_library(gmic-py MODULE "src/gmicpy.hpp" "src/gmicpy.cpp" WITH_SOABI)
set_target_properties(gmic-py PROPERTIES
        OUTPUT_NAME "_gmic"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gmic")
target_link_libraries(gmic-py PRIVATE libgmicstatic)
configure_file(pysrc/gmic/__init__.py "${CMAKE_CURRENT_BINARY_DIR}/gmic/__init__.py" COPYONLY)

get_target_property(PROJECT_CXX_STANDARD gmic-py CXX_STANDARD)

option(GMIC_PY_NUMPY "Build with numpy support" ON)
if (GMIC_PY_NUMPY)
    target_compile_options(gmic-py PUBLIC -Dgmic_py_numpy)
endif ()

# Install the module
install(TARGETS gmic-py
        EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME} ALL_COMPONENTS)

add_test(NAME gmic-py-tests
        COMMAND ${Python_EXECUTABLE} -m pytest # Or just COMMAND pytest
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
set_tests_properties(gmic-py-tests
        PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
