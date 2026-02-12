cmake_minimum_required(VERSION 3.16)
project(
    ggb
    VERSION 0.1.0
    DESCRIPTION "C++ Port of Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -pedantic)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_executable(test_tgn test/test_tgn.cpp)
target_link_libraries(test_tgn PRIVATE ${PROJECT_NAME} "${TORCH_LIBRARIES}")
