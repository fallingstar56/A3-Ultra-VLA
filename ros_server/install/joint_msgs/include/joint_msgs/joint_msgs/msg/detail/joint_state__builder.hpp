// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from joint_msgs:msg/JointState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/joint_state.hpp"


#ifndef JOINT_MSGS__MSG__DETAIL__JOINT_STATE__BUILDER_HPP_
#define JOINT_MSGS__MSG__DETAIL__JOINT_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "joint_msgs/msg/detail/joint_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace joint_msgs
{

namespace msg
{

namespace builder
{

class Init_JointState_joints
{
public:
  explicit Init_JointState_joints(::joint_msgs::msg::JointState & msg)
  : msg_(msg)
  {}
  ::joint_msgs::msg::JointState joints(::joint_msgs::msg::JointState::_joints_type arg)
  {
    msg_.joints = std::move(arg);
    return std::move(msg_);
  }

private:
  ::joint_msgs::msg::JointState msg_;
};

class Init_JointState_header
{
public:
  Init_JointState_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointState_joints header(::joint_msgs::msg::JointState::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_JointState_joints(msg_);
  }

private:
  ::joint_msgs::msg::JointState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::joint_msgs::msg::JointState>()
{
  return joint_msgs::msg::builder::Init_JointState_header();
}

}  // namespace joint_msgs

#endif  // JOINT_MSGS__MSG__DETAIL__JOINT_STATE__BUILDER_HPP_
