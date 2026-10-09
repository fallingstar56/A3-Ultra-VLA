// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from joint_msgs:msg/JointState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/joint_state.hpp"


#ifndef JOINT_MSGS__MSG__DETAIL__JOINT_STATE__TRAITS_HPP_
#define JOINT_MSGS__MSG__DETAIL__JOINT_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "joint_msgs/msg/detail/joint_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'joints'
#include "joint_msgs/msg/detail/state__traits.hpp"

namespace joint_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const JointState & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: joints
  {
    if (msg.joints.size() == 0) {
      out << "joints: []";
    } else {
      out << "joints: [";
      size_t pending_items = msg.joints.size();
      for (auto item : msg.joints) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const JointState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: joints
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joints.size() == 0) {
      out << "joints: []\n";
    } else {
      out << "joints:\n";
      for (auto item : msg.joints) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const JointState & msg, bool use_flow_style = false)
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
  const joint_msgs::msg::JointState & msg,
  std::ostream & out, size_t indentation = 0)
{
  joint_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use joint_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const joint_msgs::msg::JointState & msg)
{
  return joint_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<joint_msgs::msg::JointState>()
{
  return "joint_msgs::msg::JointState";
}

template<>
inline const char * name<joint_msgs::msg::JointState>()
{
  return "joint_msgs/msg/JointState";
}

template<>
struct has_fixed_size<joint_msgs::msg::JointState>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<joint_msgs::msg::JointState>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<joint_msgs::msg::JointState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // JOINT_MSGS__MSG__DETAIL__JOINT_STATE__TRAITS_HPP_
