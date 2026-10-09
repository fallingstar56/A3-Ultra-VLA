// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from joint_msgs:msg/Command.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/command.hpp"


#ifndef JOINT_MSGS__MSG__DETAIL__COMMAND__TRAITS_HPP_
#define JOINT_MSGS__MSG__DETAIL__COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "joint_msgs/msg/detail/command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace joint_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Command & msg,
  std::ostream & out)
{
  out << "{";
  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << ", ";
  }

  // member: sequence
  {
    out << "sequence: ";
    rosidl_generator_traits::value_to_yaml(msg.sequence, out);
    out << ", ";
  }

  // member: position
  {
    out << "position: ";
    rosidl_generator_traits::value_to_yaml(msg.position, out);
    out << ", ";
  }

  // member: velocity
  {
    out << "velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity, out);
    out << ", ";
  }

  // member: effort
  {
    out << "effort: ";
    rosidl_generator_traits::value_to_yaml(msg.effort, out);
    out << ", ";
  }

  // member: stiffness
  {
    out << "stiffness: ";
    rosidl_generator_traits::value_to_yaml(msg.stiffness, out);
    out << ", ";
  }

  // member: damping
  {
    out << "damping: ";
    rosidl_generator_traits::value_to_yaml(msg.damping, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Command & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }

  // member: sequence
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sequence: ";
    rosidl_generator_traits::value_to_yaml(msg.sequence, out);
    out << "\n";
  }

  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position: ";
    rosidl_generator_traits::value_to_yaml(msg.position, out);
    out << "\n";
  }

  // member: velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity, out);
    out << "\n";
  }

  // member: effort
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "effort: ";
    rosidl_generator_traits::value_to_yaml(msg.effort, out);
    out << "\n";
  }

  // member: stiffness
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stiffness: ";
    rosidl_generator_traits::value_to_yaml(msg.stiffness, out);
    out << "\n";
  }

  // member: damping
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "damping: ";
    rosidl_generator_traits::value_to_yaml(msg.damping, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Command & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace joint_msgs

namespace rosidl_generator_traits
{

[[deprecated("use joint_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const joint_msgs::msg::Command & msg,
  std::ostream & out, size_t indentation = 0)
{
  joint_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use joint_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const joint_msgs::msg::Command & msg)
{
  return joint_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<joint_msgs::msg::Command>()
{
  return "joint_msgs::msg::Command";
}

template<>
inline const char * name<joint_msgs::msg::Command>()
{
  return "joint_msgs/msg/Command";
}

template<>
struct has_fixed_size<joint_msgs::msg::Command>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<joint_msgs::msg::Command>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<joint_msgs::msg::Command>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // JOINT_MSGS__MSG__DETAIL__COMMAND__TRAITS_HPP_
