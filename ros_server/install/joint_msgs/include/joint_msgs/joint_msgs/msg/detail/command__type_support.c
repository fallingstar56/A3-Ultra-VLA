// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from joint_msgs:msg/Command.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "joint_msgs/msg/detail/command__rosidl_typesupport_introspection_c.h"
#include "joint_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "joint_msgs/msg/detail/command__functions.h"
#include "joint_msgs/msg/detail/command__struct.h"


// Include directives for member types
// Member `name`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  joint_msgs__msg__Command__init(message_memory);
}

void joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_fini_function(void * message_memory)
{
  joint_msgs__msg__Command__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_message_member_array[7] = {
  {
    "name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs__msg__Command, name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "sequence",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs__msg__Command, sequence),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs__msg__Command, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs__msg__Command, velocity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "effort",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs__msg__Command, effort),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "stiffness",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs__msg__Command, stiffness),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "damping",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs__msg__Command, damping),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_message_members = {
  "joint_msgs__msg",  // message namespace
  "Command",  // message name
  7,  // number of fields
  sizeof(joint_msgs__msg__Command),
  false,  // has_any_key_member_
  joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_message_member_array,  // message members
  joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_init_function,  // function to initialize message memory (memory has to be allocated)
  joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_message_type_support_handle = {
  0,
  &joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_message_members,
  get_message_typesupport_handle_function,
  &joint_msgs__msg__Command__get_type_hash,
  &joint_msgs__msg__Command__get_type_description,
  &joint_msgs__msg__Command__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_joint_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, joint_msgs, msg, Command)() {
  if (!joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_message_type_support_handle.typesupport_identifier) {
    joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &joint_msgs__msg__Command__rosidl_typesupport_introspection_c__Command_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
