add_executable(tguf_cli main.cpp)

target_link_libraries(tguf_cli PRIVATE
    ${PROJECT_NAME}
)
target_include_directories(tguf_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
