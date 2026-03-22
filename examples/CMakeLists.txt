add_executable(tgn_link_pred
    link_pred.cpp
)
target_link_libraries(tgn_link_pred PRIVATE
    tguf_models
)
target_include_directories(tgn_link_pred PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(tgn_node_pred
    node_pred.cpp
)
target_link_libraries(tgn_node_pred PRIVATE
    tguf_models
)
target_include_directories(tgn_node_pred PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
