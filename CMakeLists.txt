cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 20)
project(gmic-py C CXX)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
message("-- Using python executable: ${Python_EXECUTABLE}")
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
if (NOT SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME gmic)
endif ()
nanobind_add_module(gmic-py "src/gmicpy.cpp")
target_link_libraries(gmic-py PRIVATE libgmicstatic)
set_target_properties(gmic-py PROPERTIES
        OUTPUT_NAME "${SKBUILD_PROJECT_NAME}")

#add_custom_command(TARGET gmic-py POST_BUILD
#        COMMAND "${CMAKE_COMMAND}" -E create_symlink
#        "$<TARGET_FILE:gmic-py>"
#        "${CMAKE_CURRENT_SOURCE_DIR}/pysrc/$<TARGET_FILE_NAME:gmic-py>"
#        COMMENT "Copying .so file to pysrc/")

# Install the module
if (DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS gmic-py
            #        EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
    )
endif ()

add_test(NAME gmic-py-tests
        COMMAND ${Python_EXECUTABLE} -m pytest # Or just COMMAND pytest
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
set_tests_properties(gmic-py-tests
        PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/pysrc")
