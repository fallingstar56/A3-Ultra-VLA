# generated from rosidl_cmake/cmake/rosidl_cmake_aggregate_target-extras.cmake.in

# Create a convenience aggregate target joint_msgs::joint_msgs
# that links all generated interface targets, so downstream packages can use
# a single modern CMake target name instead of ${joint_msgs_TARGETS}.
if(joint_msgs_TARGETS AND NOT TARGET joint_msgs::joint_msgs)
  add_library(joint_msgs::joint_msgs INTERFACE IMPORTED)
  set_target_properties(joint_msgs::joint_msgs PROPERTIES
    INTERFACE_LINK_LIBRARIES "${joint_msgs_TARGETS}")
endif()
