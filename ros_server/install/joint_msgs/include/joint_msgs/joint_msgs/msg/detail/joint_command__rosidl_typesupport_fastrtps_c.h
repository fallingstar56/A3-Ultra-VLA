// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from joint_msgs:msg/JointCommand.idl
// generated code does not contain a copyright notice
#ifndef JOINT_MSGS__MSG__DETAIL__JOINT_COMMAND__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define JOINT_MSGS__MSG__DETAIL__JOINT_COMMAND__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "joint_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "joint_msgs/msg/detail/joint_command__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_joint_msgs
bool cdr_serialize_joint_msgs__msg__JointCommand(
  const joint_msgs__msg__JointCommand * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_joint_msgs
bool cdr_deserialize_joint_msgs__msg__JointCommand(
  eprosima::fastcdr::Cdr &,
  joint_msgs__msg__JointCommand * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_joint_msgs
size_t get_serialized_size_joint_msgs__msg__JointCommand(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_joint_msgs
size_t max_serialized_size_joint_msgs__msg__JointCommand(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_joint_msgs
bool cdr_serialize_key_joint_msgs__msg__JointCommand(
  const joint_msgs__msg__JointCommand * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_joint_msgs
size_t get_serialized_size_key_joint_msgs__msg__JointCommand(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_joint_msgs
size_t max_serialized_size_key_joint_msgs__msg__JointCommand(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_joint_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, joint_msgs, msg, JointCommand)();

#ifdef __cplusplus
}
#endif

#endif  // JOINT_MSGS__MSG__DETAIL__JOINT_COMMAND__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
