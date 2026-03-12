cmake_minimum_required(VERSION 3.16)
project(
    tgn_python
    VERSION 0.1.0
    DESCRIPTION "Python bindings for tgn.cpp"
    LANGUAGES CXX
)


if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

nanobind_add_module(
    _core
    NB_STATIC
    nanobind/main.cpp
)

target_link_libraries(_core PRIVATE tgn)

install(TARGETS _core
        LIBRARY DESTINATION tguf
        COMPONENT python_bindings)
