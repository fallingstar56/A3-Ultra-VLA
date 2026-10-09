// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from joint_msgs:msg/JointCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/joint_command.hpp"


#ifndef JOINT_MSGS__MSG__DETAIL__JOINT_COMMAND__BUILDER_HPP_
#define JOINT_MSGS__MSG__DETAIL__JOINT_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "joint_msgs/msg/detail/joint_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace joint_msgs
{

namespace msg
{

namespace builder
{

class Init_JointCommand_joints
{
public:
  explicit Init_JointCommand_joints(::joint_msgs::msg::JointCommand & msg)
  : msg_(msg)
  {}
  ::joint_msgs::msg::JointCommand joints(::joint_msgs::msg::JointCommand::_joints_type arg)
  {
    msg_.joints = std::move(arg);
    return std::move(msg_);
  }

private:
  ::joint_msgs::msg::JointCommand msg_;
};

class Init_JointCommand_header
{
public:
  Init_JointCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointCommand_joints header(::joint_msgs::msg::JointCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_JointCommand_joints(msg_);
  }

private:
  ::joint_msgs::msg::JointCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::joint_msgs::msg::JointCommand>()
{
  return joint_msgs::msg::builder::Init_JointCommand_header();
}

}  // namespace joint_msgs

#endif  // JOINT_MSGS__MSG__DETAIL__JOINT_COMMAND__BUILDER_HPP_
