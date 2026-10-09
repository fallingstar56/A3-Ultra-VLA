cmake_minimum_required(VERSION 3.10)
project(joint_msgs)

# Silence CMP0148 dev warnings if policy exists
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

set(msg_files
  "msg/JointState.msg"
  "msg/JointCommand.msg"
  "msg/State.msg"
  "msg/Command.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
