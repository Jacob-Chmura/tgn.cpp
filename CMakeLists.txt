cmake_minimum_required(VERSION 3.16)
project(
    tgn
    VERSION 0.1.0
    DESCRIPTION "C++ Port of Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    LANGUAGES CXX
)

option(TGN_BUILD_TESTS "Build test binaries" OFF)
option(TGN_BUILD_EXAMPLES "Build example binaries" OFF)
option(TGN_BUILD_TOOLS "Build tool binaries" OFF)

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
FetchContent_Declare(
    libtorch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip
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

add_library(${PROJECT_NAME} SHARED
    src/tgn.cpp
    src/tguf.cpp
    src/sampler.cpp
    src/scatter_ops.cpp
    src/store.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    "${TORCH_LIBRARIES}"
    OpenMP::OpenMP_CXX
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${TORCH_CXX_FLAGS}")

if (TGN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (TGN_BUILD_TOOLS)
    add_subdirectory(tools/tguf_cli)
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

    # Unit tests
    add_executable(test_unit
        test/unit/test_recency_sampler.cpp
        test/unit/test_scatter_ops.cpp
        test/unit/test_store.cpp
        test/unit/test_tgn.cpp
        test/unit/test_tguf.cpp
    )
    target_link_libraries(test_unit PRIVATE
        ${PROJECT_NAME}
        "${TORCH_LIBRARIES}"
        GTest::gtest_main
    )
    target_include_directories(test_unit PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Integration tests
    add_executable(test_integration
        test/integration/test_csv_tguf_roundtrip.cpp
    )
    target_link_libraries(test_integration PRIVATE
        ${PROJECT_NAME}
        "${TORCH_LIBRARIES}"
        GTest::gtest_main
    )
    target_include_directories(test_integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    include(GoogleTest)
    gtest_discover_tests(test_unit PROPERTIES LABELS "unit")
    gtest_discover_tests(test_integration
        PROPERTIES LABELS "integration"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()
