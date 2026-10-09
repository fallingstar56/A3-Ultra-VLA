# generated from rosidl_cmake/cmake/rosidl_cmake_aggregate_target-extras.cmake.in

# Create a convenience aggregate target ros2_plugin_proto::ros2_plugin_proto
# that links all generated interface targets, so downstream packages can use
# a single modern CMake target name instead of ${ros2_plugin_proto_TARGETS}.
if(ros2_plugin_proto_TARGETS AND NOT TARGET ros2_plugin_proto::ros2_plugin_proto)
  add_library(ros2_plugin_proto::ros2_plugin_proto INTERFACE IMPORTED)
  set_target_properties(ros2_plugin_proto::ros2_plugin_proto PROPERTIES
    INTERFACE_LINK_LIBRARIES "${ros2_plugin_proto_TARGETS}")
endif()
