cmake_minimum_required(VERSION 3.16)
project(
    tgn
    VERSION 0.1.0
    DESCRIPTION "C++ Port of Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    LANGUAGES CXX
)

option(TGN_BUILD_TESTS "Build test binaries" ON)

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

    # TODO(kuba): ensure these are broadly compatible
    #-O3 -march=native -ffast-math #-flto
)

# TODO(kuba): ensure we get a good version of libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build: Enabling Address and UndefinedBehavior Sanitizers")
    add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

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

    add_executable(test_tgn
        test/test_tgn.cpp
    )
    target_link_libraries(test_tgn PRIVATE
        ${PROJECT_NAME}
        "${TORCH_LIBRARIES}"
    #    GTest::gtest_main
    )
    #include(GoogleTest)
    #gtest_discover_tests(test_tgn)
endif()
