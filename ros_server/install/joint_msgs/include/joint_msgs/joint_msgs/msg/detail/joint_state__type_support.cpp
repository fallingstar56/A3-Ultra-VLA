// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from joint_msgs:msg/JointState.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "joint_msgs/msg/detail/joint_state__functions.h"
#include "joint_msgs/msg/detail/joint_state__struct.hpp"
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

void JointState_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) joint_msgs::msg::JointState(_init);
}

void JointState_fini_function(void * message_memory)
{
  auto typed_message = static_cast<joint_msgs::msg::JointState *>(message_memory);
  typed_message->~JointState();
}

size_t size_function__JointState__joints(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<joint_msgs::msg::State> *>(untyped_member);
  return member->size();
}

const void * get_const_function__JointState__joints(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<joint_msgs::msg::State> *>(untyped_member);
  return &member[index];
}

void * get_function__JointState__joints(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<joint_msgs::msg::State> *>(untyped_member);
  return &member[index];
}

void fetch_function__JointState__joints(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const joint_msgs::msg::State *>(
    get_const_function__JointState__joints(untyped_member, index));
  auto & value = *reinterpret_cast<joint_msgs::msg::State *>(untyped_value);
  value = item;
}

void assign_function__JointState__joints(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<joint_msgs::msg::State *>(
    get_function__JointState__joints(untyped_member, index));
  const auto & value = *reinterpret_cast<const joint_msgs::msg::State *>(untyped_value);
  item = value;
}

void resize_function__JointState__joints(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<joint_msgs::msg::State> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember JointState_message_member_array[2] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs::msg::JointState, header),  // bytes offset in struct
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
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<joint_msgs::msg::State>(),  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(joint_msgs::msg::JointState, joints),  // bytes offset in struct
    nullptr,  // default value
    size_function__JointState__joints,  // size() function pointer
    get_const_function__JointState__joints,  // get_const(index) function pointer
    get_function__JointState__joints,  // get(index) function pointer
    fetch_function__JointState__joints,  // fetch(index, &value) function pointer
    assign_function__JointState__joints,  // assign(index, value) function pointer
    resize_function__JointState__joints  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers JointState_message_members = {
  "joint_msgs::msg",  // message namespace
  "JointState",  // message name
  2,  // number of fields
  sizeof(joint_msgs::msg::JointState),
  false,  // has_any_key_member_
  JointState_message_member_array,  // message members
  JointState_init_function,  // function to initialize message memory (memory has to be allocated)
  JointState_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t JointState_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &JointState_message_members,
  get_message_typesupport_handle_function,
  &joint_msgs__msg__JointState__get_type_hash,
  &joint_msgs__msg__JointState__get_type_description,
  &joint_msgs__msg__JointState__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace joint_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<joint_msgs::msg::JointState>()
{
  return &::joint_msgs::msg::rosidl_typesupport_introspection_cpp::JointState_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, joint_msgs, msg, JointState)() {
  return &::joint_msgs::msg::rosidl_typesupport_introspection_cpp::JointState_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
