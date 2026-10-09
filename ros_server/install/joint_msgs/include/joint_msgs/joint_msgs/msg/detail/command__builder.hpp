// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from joint_msgs:msg/Command.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/command.hpp"


#ifndef JOINT_MSGS__MSG__DETAIL__COMMAND__BUILDER_HPP_
#define JOINT_MSGS__MSG__DETAIL__COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "joint_msgs/msg/detail/command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace joint_msgs
{

namespace msg
{

namespace builder
{

class Init_Command_damping
{
public:
  explicit Init_Command_damping(::joint_msgs::msg::Command & msg)
  : msg_(msg)
  {}
  ::joint_msgs::msg::Command damping(::joint_msgs::msg::Command::_damping_type arg)
  {
    msg_.damping = std::move(arg);
    return std::move(msg_);
  }

private:
  ::joint_msgs::msg::Command msg_;
};

class Init_Command_stiffness
{
public:
  explicit Init_Command_stiffness(::joint_msgs::msg::Command & msg)
  : msg_(msg)
  {}
  Init_Command_damping stiffness(::joint_msgs::msg::Command::_stiffness_type arg)
  {
    msg_.stiffness = std::move(arg);
    return Init_Command_damping(msg_);
  }

private:
  ::joint_msgs::msg::Command msg_;
};

class Init_Command_effort
{
public:
  explicit Init_Command_effort(::joint_msgs::msg::Command & msg)
  : msg_(msg)
  {}
  Init_Command_stiffness effort(::joint_msgs::msg::Command::_effort_type arg)
  {
    msg_.effort = std::move(arg);
    return Init_Command_stiffness(msg_);
  }

private:
  ::joint_msgs::msg::Command msg_;
};

class Init_Command_velocity
{
public:
  explicit Init_Command_velocity(::joint_msgs::msg::Command & msg)
  : msg_(msg)
  {}
  Init_Command_effort velocity(::joint_msgs::msg::Command::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_Command_effort(msg_);
  }

private:
  ::joint_msgs::msg::Command msg_;
};

class Init_Command_position
{
public:
  explicit Init_Command_position(::joint_msgs::msg::Command & msg)
  : msg_(msg)
  {}
  Init_Command_velocity position(::joint_msgs::msg::Command::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_Command_velocity(msg_);
  }

private:
  ::joint_msgs::msg::Command msg_;
};

class Init_Command_sequence
{
public:
  explicit Init_Command_sequence(::joint_msgs::msg::Command & msg)
  : msg_(msg)
  {}
  Init_Command_position sequence(::joint_msgs::msg::Command::_sequence_type arg)
  {
    msg_.sequence = std::move(arg);
    return Init_Command_position(msg_);
  }

private:
  ::joint_msgs::msg::Command msg_;
};

class Init_Command_name
{
public:
  Init_Command_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Command_sequence name(::joint_msgs::msg::Command::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_Command_sequence(msg_);
  }

private:
  ::joint_msgs::msg::Command msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::joint_msgs::msg::Command>()
{
  return joint_msgs::msg::builder::Init_Command_name();
}

}  // namespace joint_msgs

#endif  // JOINT_MSGS__MSG__DETAIL__COMMAND__BUILDER_HPP_
