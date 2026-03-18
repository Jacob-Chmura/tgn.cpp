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

nanobind_add_module(
    _tguf
    NB_STATIC
    bind.cpp
)

target_link_libraries(_tguf PRIVATE tgn)

nanobind_add_stub(
    _tguf_stub
    MODULE _tguf
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_tguf.pyi"
    PYTHON_PATH "$<TARGET_FILE_DIR:_tguf>"
    DEPENDS _tguf
)

# Re-export generated stubs for mkdocs integration
add_custom_command(
    TARGET _tguf_stub POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/_tguf.pyi"
        "${CMAKE_SOURCE_DIR}/python/tguf/_tguf.pyi"
)

# TODO(kuba): This is surely not the simplest way to do things
set(LOCAL_TORCH_LIB "${CMAKE_BINARY_DIR}/_deps/libtorch-src/lib")
get_target_property(TGN_BINARY_DIR tgn BINARY_DIR)

# $ORIGIN: Look in the current folder (for installed versions)
# $ORIGIN/..: Look in the parent folder (common in build trees)
# ${TGN_BINARY_DIR}: Look exactly where the C++ build put libtgn.so
# ${LOCAL_TORCH_LIB}: Look where FetchContent put LibTorch
set_target_properties(_tguf PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;$ORIGIN/..;${TGN_BINARY_DIR};${LOCAL_TORCH_LIB}"
)

install(TARGETS _tguf
        LIBRARY DESTINATION tguf
        COMPONENT python_bindings)
