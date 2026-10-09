// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from joint_msgs:msg/Command.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/command.hpp"


#ifndef JOINT_MSGS__MSG__DETAIL__COMMAND__STRUCT_HPP_
#define JOINT_MSGS__MSG__DETAIL__COMMAND__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__joint_msgs__msg__Command __attribute__((deprecated))
#else
# define DEPRECATED__joint_msgs__msg__Command __declspec(deprecated)
#endif

namespace joint_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Command_
{
  using Type = Command_<ContainerAllocator>;

  explicit Command_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->name = "";
      this->sequence = 0ul;
      this->position = 0.0;
      this->velocity = 0.0;
      this->effort = 0.0;
      this->stiffness = 0.0;
      this->damping = 0.0;
    }
  }

  explicit Command_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : name(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->name = "";
      this->sequence = 0ul;
      this->position = 0.0;
      this->velocity = 0.0;
      this->effort = 0.0;
      this->stiffness = 0.0;
      this->damping = 0.0;
    }
  }

  // field types and members
  using _name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _name_type name;
  using _sequence_type =
    uint32_t;
  _sequence_type sequence;
  using _position_type =
    double;
  _position_type position;
  using _velocity_type =
    double;
  _velocity_type velocity;
  using _effort_type =
    double;
  _effort_type effort;
  using _stiffness_type =
    double;
  _stiffness_type stiffness;
  using _damping_type =
    double;
  _damping_type damping;

  // setters for named parameter idiom
  Type & set__name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->name = _arg;
    return *this;
  }
  Type & set__sequence(
    const uint32_t & _arg)
  {
    this->sequence = _arg;
    return *this;
  }
  Type & set__position(
    const double & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__velocity(
    const double & _arg)
  {
    this->velocity = _arg;
    return *this;
  }
  Type & set__effort(
    const double & _arg)
  {
    this->effort = _arg;
    return *this;
  }
  Type & set__stiffness(
    const double & _arg)
  {
    this->stiffness = _arg;
    return *this;
  }
  Type & set__damping(
    const double & _arg)
  {
    this->damping = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    joint_msgs::msg::Command_<ContainerAllocator> *;
  using ConstRawPtr =
    const joint_msgs::msg::Command_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<joint_msgs::msg::Command_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<joint_msgs::msg::Command_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      joint_msgs::msg::Command_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<joint_msgs::msg::Command_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      joint_msgs::msg::Command_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<joint_msgs::msg::Command_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<joint_msgs::msg::Command_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<joint_msgs::msg::Command_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__joint_msgs__msg__Command
    std::shared_ptr<joint_msgs::msg::Command_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__joint_msgs__msg__Command
    std::shared_ptr<joint_msgs::msg::Command_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Command_ & other) const
  {
    if (this->name != other.name) {
      return false;
    }
    if (this->sequence != other.sequence) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->velocity != other.velocity) {
      return false;
    }
    if (this->effort != other.effort) {
      return false;
    }
    if (this->stiffness != other.stiffness) {
      return false;
    }
    if (this->damping != other.damping) {
      return false;
    }
    return true;
  }
  bool operator!=(const Command_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Command_

// alias to use template instance with default allocator
using Command =
  joint_msgs::msg::Command_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace joint_msgs

#endif  // JOINT_MSGS__MSG__DETAIL__COMMAND__STRUCT_HPP_
