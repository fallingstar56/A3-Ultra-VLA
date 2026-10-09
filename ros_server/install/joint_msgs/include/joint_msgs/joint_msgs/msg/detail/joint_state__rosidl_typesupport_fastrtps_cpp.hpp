// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from joint_msgs:msg/JointState.idl
// generated code does not contain a copyright notice

#ifndef JOINT_MSGS__MSG__DETAIL__JOINT_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define JOINT_MSGS__MSG__DETAIL__JOINT_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include <cstddef>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "joint_msgs/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "joint_msgs/msg/detail/joint_state__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace joint_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_joint_msgs
cdr_serialize(
  const joint_msgs::msg::JointState & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_joint_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  joint_msgs::msg::JointState & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_joint_msgs
get_serialized_size(
  const joint_msgs::msg::JointState & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_joint_msgs
max_serialized_size_JointState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_joint_msgs
cdr_serialize_key(
  const joint_msgs::msg::JointState & ros_message,
  eprosima::fastcdr::Cdr &);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_joint_msgs
get_serialized_size_key(
  const joint_msgs::msg::JointState & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_joint_msgs
max_serialized_size_key_JointState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace joint_msgs

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_joint_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, joint_msgs, msg, JointState)();

#ifdef __cplusplus
}
#endif

#endif  // JOINT_MSGS__MSG__DETAIL__JOINT_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
