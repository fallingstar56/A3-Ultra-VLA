// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from joint_msgs:msg/State.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/state.hpp"


#ifndef JOINT_MSGS__MSG__DETAIL__STATE__BUILDER_HPP_
#define JOINT_MSGS__MSG__DETAIL__STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "joint_msgs/msg/detail/state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace joint_msgs
{

namespace msg
{

namespace builder
{

class Init_State_effort
{
public:
  explicit Init_State_effort(::joint_msgs::msg::State & msg)
  : msg_(msg)
  {}
  ::joint_msgs::msg::State effort(::joint_msgs::msg::State::_effort_type arg)
  {
    msg_.effort = std::move(arg);
    return std::move(msg_);
  }

private:
  ::joint_msgs::msg::State msg_;
};

class Init_State_velocity
{
public:
  explicit Init_State_velocity(::joint_msgs::msg::State & msg)
  : msg_(msg)
  {}
  Init_State_effort velocity(::joint_msgs::msg::State::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_State_effort(msg_);
  }

private:
  ::joint_msgs::msg::State msg_;
};

class Init_State_position
{
public:
  explicit Init_State_position(::joint_msgs::msg::State & msg)
  : msg_(msg)
  {}
  Init_State_velocity position(::joint_msgs::msg::State::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_State_velocity(msg_);
  }

private:
  ::joint_msgs::msg::State msg_;
};

class Init_State_sequence
{
public:
  explicit Init_State_sequence(::joint_msgs::msg::State & msg)
  : msg_(msg)
  {}
  Init_State_position sequence(::joint_msgs::msg::State::_sequence_type arg)
  {
    msg_.sequence = std::move(arg);
    return Init_State_position(msg_);
  }

private:
  ::joint_msgs::msg::State msg_;
};

class Init_State_name
{
public:
  Init_State_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_State_sequence name(::joint_msgs::msg::State::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_State_sequence(msg_);
  }

private:
  ::joint_msgs::msg::State msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::joint_msgs::msg::State>()
{
  return joint_msgs::msg::builder::Init_State_name();
}

}  // namespace joint_msgs

#endif  // JOINT_MSGS__MSG__DETAIL__STATE__BUILDER_HPP_
