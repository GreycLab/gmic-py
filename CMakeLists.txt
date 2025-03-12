cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 20)
project(gmic-py LANGUAGES C CXX)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Using python executable: ${Python_EXECUTABLE}")

option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." FALSE)
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    endif ()
endif ()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

if (NOT DEFINED GMICPY_VERSION)
    execute_process(
            COMMAND "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/version_build.py"
            OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE GMICPY_VERSION
            COMMAND_ERROR_IS_FATAL ANY)
endif ()
if (DEFINED SKBUILD_PROJECT_VERSION_FULL)
    string(REGEX REPLACE [[^([^+]+)(\+.*)?$]] [[\1]] SKBUILD_VERSION_NOLOCAL ${SKBUILD_PROJECT_VERSION_FULL})
    if (NOT SKBUILD_VERSION_NOLOCAL STREQUAL GMICPY_VERSION)
        message(FATAL_ERROR "Versions provided by skbuild and version_build.py differ: "
                "${SKBUILD_VERSION_NOLOCAL} != ${GMICPY_VERSION}. Update version.txt with `./version_build.py -u`")
    endif ()
endif ()
message(STATUS "Building gmic-py version ${GMICPY_VERSION} (${CMAKE_BUILD_TYPE})")

set(GMIC_ROOT "${PROJECT_SOURCE_DIR}/lib/gmic")
set(CIMG_ROOT "${PROJECT_SOURCE_DIR}/lib/cimg")
if (NOT EXISTS "${GMIC_ROOT}/src/")
    message(FATAL_ERROR "./lib/gmic submodule appears to be uninitialized. Try running git submodule init; git submodule update")
endif ()

if (USE_SYSTEM_CIMG)
    message(VERBOSE "Using system CImg")
else ()
    if (NOT EXISTS "${CIMG_ROOT}/CImg.h")
        message(FATAL_ERROR "lib/cimg submodule appears to be uninitialized. Try running git submodule init; git submodule update")
    endif ()
    set(USE_SYSTEM_CIMG 1)
    message(VERBOSE "Using CImg from lib/cimg")
    cmake_path(APPEND CMAKE_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/lib/cimg/")
    include_directories("${PROJECT_SOURCE_DIR}/lib/cimg/")
endif ()
add_subdirectory("${GMIC_ROOT}" EXCLUDE_FROM_ALL)

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
nanobind_add_module(gmic-py "src/gmicpy.cpp" "src/gmicpy.h" "src/logging.h" "src/translate_args.h")
target_link_libraries(gmic-py PRIVATE libgmicstatic)
set_target_properties(gmic-py PROPERTIES
        OUTPUT_NAME "__init__")
target_compile_definitions(gmic-py PRIVATE "DEBUG=$<IF:$<CONFIG:Debug>,1,0>" "GMICPY_VERSION=${GMICPY_VERSION}")
nanobind_add_stub(
        gmic-py-stub
        MODULE gmic
        DEPENDS gmic-py
        OUTPUT "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.pyi"
        PYTHON_PATH $<TARGET_FILE_DIR:gmic-py>
        MARKER_FILE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/py.typed"
)
get_property(CIMG_COMPILE_DEFS
        TARGET gmic-py
        PROPERTY COMPILE_DEFINITIONS)

# Install the module
if (DEFINED SKBUILD_PLATLIB_DIR)
    install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} DESTINATION ${SKBUILD_PLATLIB_DIR})
endif ()
