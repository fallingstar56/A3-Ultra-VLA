// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from joint_msgs:msg/State.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/state.h"


#ifndef JOINT_MSGS__MSG__DETAIL__STATE__STRUCT_H_
#define JOINT_MSGS__MSG__DETAIL__STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'name'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/State in the package joint_msgs.
typedef struct joint_msgs__msg__State
{
  rosidl_runtime_c__String name;
  uint32_t sequence;
  double position;
  double velocity;
  double effort;
} joint_msgs__msg__State;

// Struct for a sequence of joint_msgs__msg__State.
typedef struct joint_msgs__msg__State__Sequence
{
  joint_msgs__msg__State * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} joint_msgs__msg__State__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // JOINT_MSGS__MSG__DETAIL__STATE__STRUCT_H_
