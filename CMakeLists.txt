cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 20)
project(gmic-py LANGUAGES C CXX)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Using python executable: ${Python_EXECUTABLE}")

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
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/gmic)

# Add the module to compile
if (NOT SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME gmic)
endif ()
nanobind_add_module(gmic-py "src/gmicpy.cpp")
target_link_libraries(gmic-py PRIVATE libgmicstatic)
set_target_properties(gmic-py PROPERTIES
        OUTPUT_NAME "__init__")
nanobind_add_stub(
        gmic-py-stub
        MODULE gmic
        DEPENDS gmic-py
        OUTPUT "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.pyi"
        PYTHON_PATH $<TARGET_FILE_DIR:gmic-py>
        MARKER_FILE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/py.typed"
)

# Install the module
if (DEFINED SKBUILD_PLATLIB_DIR)
    install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} DESTINATION ${SKBUILD_PLATLIB_DIR})
endif ()
