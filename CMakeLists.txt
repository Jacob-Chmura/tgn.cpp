cmake_minimum_required(VERSION 3.16)
project(
    tgn
    VERSION 0.1.0
    DESCRIPTION "C++ Port of Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    LANGUAGES CXX
)

option(TGN_BUILD_TESTS "Build test binaries" OFF)
option(TGN_BUILD_EXAMPLES "Build example binaries" OFF)
option(TGN_BUILD_PYTHON "Build Python Bindings" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow                 # Catches local variables hiding class members
    -Wnon-virtual-dtor       # Ensures proper cleanup of polymorphic classes
    -Wold-style-cast         # Encourages static_cast over C-style casts
    -Woverloaded-virtual     # Warns if you accidentally hide virtual functions
    -Wnull-dereference       # Warns if a null dereference is detected
)

include(FetchContent)

if(APPLE) # Target Apple Silicon (M1/M2/M3)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.10.0.zip")
else() # Target Linux x86_64
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip")
endif()

FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP ON
)
FetchContent_MakeAvailable(libtorch)

set(CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(APPLE)
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()

find_package(OpenMP REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build: Enabling Address and UndefinedBehavior Sanitizers")
    add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release build: Enabling some Compiler Optimizations")
    add_compile_options(-O3 -march=native -ffast-math -pthread)
endif()

# TGUF Target
add_library(tguf
    SHARED
        src/tguf/store.cpp
        src/tguf/tguf.cpp
)
target_link_libraries(tguf
    PUBLIC
        "${TORCH_LIBRARIES}"
        OpenMP::OpenMP_CXX
)
target_include_directories(tguf
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(tguf PROPERTIES COMPILE_FLAGS "${TORCH_CXX_FLAGS}")

# TGUF-Models Target
add_library(tguf_models
    SHARED
        src/tguf_models/common/sampler.cpp
        src/tguf_models/common/scatter_ops.cpp
        src/tguf_models/tgn.cpp
)

target_link_libraries(tguf_models
    PUBLIC
        tguf
)

target_include_directories(tguf_models
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(tguf_models PROPERTIES COMPILE_FLAGS "${TORCH_CXX_FLAGS}")

if (TGN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (TGN_BUILD_PYTHON)
    add_subdirectory(python)
endif()


if (TGN_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # TGUF unit tests
    add_executable(test_tguf
        test/unit/tguf/test_store.cpp
        test/unit/tguf/test_tguf.cpp
    )
    target_link_libraries(test_tguf PRIVATE
        tguf
        "${TORCH_LIBRARIES}"
        GTest::gtest_main
    )
    target_include_directories(test_tguf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # TGUF-Model unit tests
    add_executable(test_tguf_models
        test/unit/tguf_models/test_recency_sampler.cpp
        test/unit/tguf_models/test_scatter_ops.cpp
        test/unit/tguf_models/test_tgn.cpp
    )
    target_link_libraries(test_tguf_models PRIVATE
        tguf_models
        "${TORCH_LIBRARIES}"
        GTest::gtest_main
    )
    target_include_directories(test_tguf_models PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    include(GoogleTest)
    gtest_discover_tests(test_tguf PROPERTIES LABELS "tguf")
    gtest_discover_tests(test_tguf_models PROPERTIES LABELS "models")
endif()
