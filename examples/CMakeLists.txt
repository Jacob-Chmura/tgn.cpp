add_executable(tgn_example
    main.cpp
)
target_link_libraries(tgn_example PRIVATE
    ${PROJECT_NAME}
)
target_include_directories(tgn_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
