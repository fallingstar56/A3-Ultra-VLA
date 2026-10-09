// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from joint_msgs:msg/State.idl
// generated code does not contain a copyright notice
#include "joint_msgs/msg/detail/state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `name`
#include "rosidl_runtime_c/string_functions.h"

bool
joint_msgs__msg__State__init(joint_msgs__msg__State * msg)
{
  if (!msg) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__init(&msg->name)) {
    joint_msgs__msg__State__fini(msg);
    return false;
  }
  // sequence
  // position
  // velocity
  // effort
  return true;
}

void
joint_msgs__msg__State__fini(joint_msgs__msg__State * msg)
{
  if (!msg) {
    return;
  }
  // name
  rosidl_runtime_c__String__fini(&msg->name);
  // sequence
  // position
  // velocity
  // effort
}

bool
joint_msgs__msg__State__are_equal(const joint_msgs__msg__State * lhs, const joint_msgs__msg__State * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->name), &(rhs->name)))
  {
    return false;
  }
  // sequence
  if (lhs->sequence != rhs->sequence) {
    return false;
  }
  // position
  if (lhs->position != rhs->position) {
    return false;
  }
  // velocity
  if (lhs->velocity != rhs->velocity) {
    return false;
  }
  // effort
  if (lhs->effort != rhs->effort) {
    return false;
  }
  return true;
}

bool
joint_msgs__msg__State__copy(
  const joint_msgs__msg__State * input,
  joint_msgs__msg__State * output)
{
  if (!input || !output) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__copy(
      &(input->name), &(output->name)))
  {
    return false;
  }
  // sequence
  output->sequence = input->sequence;
  // position
  output->position = input->position;
  // velocity
  output->velocity = input->velocity;
  // effort
  output->effort = input->effort;
  return true;
}

joint_msgs__msg__State *
joint_msgs__msg__State__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  joint_msgs__msg__State * msg = (joint_msgs__msg__State *)allocator.allocate(sizeof(joint_msgs__msg__State), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(joint_msgs__msg__State));
  bool success = joint_msgs__msg__State__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
joint_msgs__msg__State__destroy(joint_msgs__msg__State * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    joint_msgs__msg__State__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
joint_msgs__msg__State__Sequence__init(joint_msgs__msg__State__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  joint_msgs__msg__State * data = NULL;

  if (size) {
    if (size > SIZE_MAX / sizeof(joint_msgs__msg__State)) {
      return false;
    }
    data = (joint_msgs__msg__State *)allocator.zero_allocate(size, sizeof(joint_msgs__msg__State), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = joint_msgs__msg__State__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        joint_msgs__msg__State__fini(&data[i - 1]);
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
joint_msgs__msg__State__Sequence__fini(joint_msgs__msg__State__Sequence * array)
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
      joint_msgs__msg__State__fini(&array->data[i]);
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

joint_msgs__msg__State__Sequence *
joint_msgs__msg__State__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  joint_msgs__msg__State__Sequence * array = (joint_msgs__msg__State__Sequence *)allocator.allocate(sizeof(joint_msgs__msg__State__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = joint_msgs__msg__State__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
joint_msgs__msg__State__Sequence__destroy(joint_msgs__msg__State__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    joint_msgs__msg__State__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
joint_msgs__msg__State__Sequence__are_equal(const joint_msgs__msg__State__Sequence * lhs, const joint_msgs__msg__State__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!joint_msgs__msg__State__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
joint_msgs__msg__State__Sequence__copy(
  const joint_msgs__msg__State__Sequence * input,
  joint_msgs__msg__State__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    if (input->size > SIZE_MAX / sizeof(joint_msgs__msg__State)) {
      return false;
    }
    const size_t allocation_size =
      input->size * sizeof(joint_msgs__msg__State);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    joint_msgs__msg__State * data =
      (joint_msgs__msg__State *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!joint_msgs__msg__State__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          joint_msgs__msg__State__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!joint_msgs__msg__State__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
