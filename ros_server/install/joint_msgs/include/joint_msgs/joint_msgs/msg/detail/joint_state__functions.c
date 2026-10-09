// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from joint_msgs:msg/JointState.idl
// generated code does not contain a copyright notice
#include "joint_msgs/msg/detail/joint_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `joints`
#include "joint_msgs/msg/detail/state__functions.h"

bool
joint_msgs__msg__JointState__init(joint_msgs__msg__JointState * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    joint_msgs__msg__JointState__fini(msg);
    return false;
  }
  // joints
  if (!joint_msgs__msg__State__Sequence__init(&msg->joints, 0)) {
    joint_msgs__msg__JointState__fini(msg);
    return false;
  }
  return true;
}

void
joint_msgs__msg__JointState__fini(joint_msgs__msg__JointState * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // joints
  joint_msgs__msg__State__Sequence__fini(&msg->joints);
}

bool
joint_msgs__msg__JointState__are_equal(const joint_msgs__msg__JointState * lhs, const joint_msgs__msg__JointState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // joints
  if (!joint_msgs__msg__State__Sequence__are_equal(
      &(lhs->joints), &(rhs->joints)))
  {
    return false;
  }
  return true;
}

bool
joint_msgs__msg__JointState__copy(
  const joint_msgs__msg__JointState * input,
  joint_msgs__msg__JointState * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // joints
  if (!joint_msgs__msg__State__Sequence__copy(
      &(input->joints), &(output->joints)))
  {
    return false;
  }
  return true;
}

joint_msgs__msg__JointState *
joint_msgs__msg__JointState__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  joint_msgs__msg__JointState * msg = (joint_msgs__msg__JointState *)allocator.allocate(sizeof(joint_msgs__msg__JointState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(joint_msgs__msg__JointState));
  bool success = joint_msgs__msg__JointState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
joint_msgs__msg__JointState__destroy(joint_msgs__msg__JointState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    joint_msgs__msg__JointState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
joint_msgs__msg__JointState__Sequence__init(joint_msgs__msg__JointState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  joint_msgs__msg__JointState * data = NULL;

  if (size) {
    if (size > SIZE_MAX / sizeof(joint_msgs__msg__JointState)) {
      return false;
    }
    data = (joint_msgs__msg__JointState *)allocator.zero_allocate(size, sizeof(joint_msgs__msg__JointState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = joint_msgs__msg__JointState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        joint_msgs__msg__JointState__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
joint_msgs__msg__JointState__Sequence__fini(joint_msgs__msg__JointState__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      joint_msgs__msg__JointState__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

joint_msgs__msg__JointState__Sequence *
joint_msgs__msg__JointState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  joint_msgs__msg__JointState__Sequence * array = (joint_msgs__msg__JointState__Sequence *)allocator.allocate(sizeof(joint_msgs__msg__JointState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = joint_msgs__msg__JointState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
joint_msgs__msg__JointState__Sequence__destroy(joint_msgs__msg__JointState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    joint_msgs__msg__JointState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
joint_msgs__msg__JointState__Sequence__are_equal(const joint_msgs__msg__JointState__Sequence * lhs, const joint_msgs__msg__JointState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!joint_msgs__msg__JointState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
joint_msgs__msg__JointState__Sequence__copy(
  const joint_msgs__msg__JointState__Sequence * input,
  joint_msgs__msg__JointState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    if (input->size > SIZE_MAX / sizeof(joint_msgs__msg__JointState)) {
      return false;
    }
    const size_t allocation_size =
      input->size * sizeof(joint_msgs__msg__JointState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    joint_msgs__msg__JointState * data =
      (joint_msgs__msg__JointState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!joint_msgs__msg__JointState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          joint_msgs__msg__JointState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!joint_msgs__msg__JointState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
