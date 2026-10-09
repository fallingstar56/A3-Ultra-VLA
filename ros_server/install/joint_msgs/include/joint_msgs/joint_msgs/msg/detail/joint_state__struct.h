// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from joint_msgs:msg/JointState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/joint_state.h"


#ifndef JOINT_MSGS__MSG__DETAIL__JOINT_STATE__STRUCT_H_
#define JOINT_MSGS__MSG__DETAIL__JOINT_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'joints'
#include "joint_msgs/msg/detail/state__struct.h"

/// Struct defined in msg/JointState in the package joint_msgs.
typedef struct joint_msgs__msg__JointState
{
  std_msgs__msg__Header header;
  joint_msgs__msg__State__Sequence joints;
} joint_msgs__msg__JointState;

// Struct for a sequence of joint_msgs__msg__JointState.
typedef struct joint_msgs__msg__JointState__Sequence
{
  joint_msgs__msg__JointState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} joint_msgs__msg__JointState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // JOINT_MSGS__MSG__DETAIL__JOINT_STATE__STRUCT_H_
