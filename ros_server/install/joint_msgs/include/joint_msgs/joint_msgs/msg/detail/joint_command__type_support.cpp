// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from joint_msgs:msg/JointCommand.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "joint_msgs/msg/detail/joint_command__functions.h"
#include "joint_msgs/msg/detail/joint_command__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace joint_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void JointCommand_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) joint_msgs::msg::JointCommand(_init);
}

void JointCommand_fini_function(void * message_memory)
{
  auto typed_message = static_cast<joint_msgs::msg::JointCommand *>(message_memory);
  typed_message->~JointCommand();
}

size_t size_function__JointCommand__joints(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<joint_msgs::msg::Command> *>(untyped_member);
  return member->size();
}

const void * get_const_function__JointCommand__joints(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<joint_msgs::msg::Command> *>(untyped_member);
  return &member[index];
}

void * get_function__JointCommand__joints(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<joint_msgs::msg::Command> *>(untyped_member);
  return &member[index];
}

void fetch_function__JointCommand__joints(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const joint_msgs::msg::Command *>(
    get_const_function__JointCommand__joints(untyped_member, index));
  auto & value = *reinterpret_cast<joint_msgs::msg::Command *>(untyped_value);
  value = item;
}

void assign_function__JointCommand__joints(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<joint_msgs::msg::Command *>(
    get_function__JointCommand__joints(untyped_member, index));
  const auto & value = *reinterpret_cast<const joint_msgs::msg::Command *>(untyped_value);
  item = value;
}

void resize_function__JointCommand__joints(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<joint_msgs::msg::Command> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember JointCommand_message_member_array[2] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs::msg::JointCommand, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "joints",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<joint_msgs::msg::Command>(),  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs::msg::JointCommand, joints),  // bytes offset in struct
    nullptr,  // default value
    size_function__JointCommand__joints,  // size() function pointer
    get_const_function__JointCommand__joints,  // get_const(index) function pointer
    get_function__JointCommand__joints,  // get(index) function pointer
    fetch_function__JointCommand__joints,  // fetch(index, &value) function pointer
    assign_function__JointCommand__joints,  // assign(index, value) function pointer
    resize_function__JointCommand__joints  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers JointCommand_message_members = {
  "joint_msgs::msg",  // message namespace
  "JointCommand",  // message name
  2,  // number of fields
  sizeof(joint_msgs::msg::JointCommand),
  false,  // has_any_key_member_
  JointCommand_message_member_array,  // message members
  JointCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  JointCommand_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t JointCommand_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &JointCommand_message_members,
  get_message_typesupport_handle_function,
  &joint_msgs__msg__JointCommand__get_type_hash,
  &joint_msgs__msg__JointCommand__get_type_description,
  &joint_msgs__msg__JointCommand__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace joint_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<joint_msgs::msg::JointCommand>()
{
  return &::joint_msgs::msg::rosidl_typesupport_introspection_cpp::JointCommand_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, joint_msgs, msg, JointCommand)() {
  return &::joint_msgs::msg::rosidl_typesupport_introspection_cpp::JointCommand_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
