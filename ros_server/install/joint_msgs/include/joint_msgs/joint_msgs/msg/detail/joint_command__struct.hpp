// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from joint_msgs:msg/JointCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "joint_msgs/msg/joint_command.hpp"


#ifndef JOINT_MSGS__MSG__DETAIL__JOINT_COMMAND__STRUCT_HPP_
#define JOINT_MSGS__MSG__DETAIL__JOINT_COMMAND__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'joints'
#include "joint_msgs/msg/detail/command__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__joint_msgs__msg__JointCommand __attribute__((deprecated))
#else
# define DEPRECATED__joint_msgs__msg__JointCommand __declspec(deprecated)
#endif

namespace joint_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct JointCommand_
{
  using Type = JointCommand_<ContainerAllocator>;

  explicit JointCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit JointCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _joints_type =
    std::vector<joint_msgs::msg::Command_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<joint_msgs::msg::Command_<ContainerAllocator>>>;
  _joints_type joints;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__joints(
    const std::vector<joint_msgs::msg::Command_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<joint_msgs::msg::Command_<ContainerAllocator>>> & _arg)
  {
    this->joints = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    joint_msgs::msg::JointCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const joint_msgs::msg::JointCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<joint_msgs::msg::JointCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<joint_msgs::msg::JointCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      joint_msgs::msg::JointCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<joint_msgs::msg::JointCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      joint_msgs::msg::JointCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<joint_msgs::msg::JointCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<joint_msgs::msg::JointCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<joint_msgs::msg::JointCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__joint_msgs__msg__JointCommand
    std::shared_ptr<joint_msgs::msg::JointCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__joint_msgs__msg__JointCommand
    std::shared_ptr<joint_msgs::msg::JointCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JointCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->joints != other.joints) {
      return false;
    }
    return true;
  }
  bool operator!=(const JointCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JointCommand_

// alias to use template instance with default allocator
using JointCommand =
  joint_msgs::msg::JointCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace joint_msgs

#endif  // JOINT_MSGS__MSG__DETAIL__JOINT_COMMAND__STRUCT_HPP_
