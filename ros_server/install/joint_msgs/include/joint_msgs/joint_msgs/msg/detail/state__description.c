// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from joint_msgs:msg/State.idl
// generated code does not contain a copyright notice

#include "joint_msgs/msg/detail/state__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_joint_msgs
const rosidl_type_hash_t *
joint_msgs__msg__State__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xd1, 0x27, 0x0e, 0xe8, 0x56, 0xf5, 0x85, 0x22,
      0xce, 0xaf, 0x92, 0x26, 0xd3, 0x0d, 0xf4, 0x37,
      0x83, 0xfe, 0xd2, 0xe4, 0xd8, 0x0c, 0x22, 0x9a,
      0xff, 0x3c, 0x24, 0xa0, 0x44, 0x0b, 0xff, 0x38,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char joint_msgs__msg__State__TYPE_NAME[] = "joint_msgs/msg/State";

// Define type names, field names, and default values
static char joint_msgs__msg__State__FIELD_NAME__name[] = "name";
static char joint_msgs__msg__State__FIELD_NAME__sequence[] = "sequence";
static char joint_msgs__msg__State__FIELD_NAME__position[] = "position";
static char joint_msgs__msg__State__FIELD_NAME__velocity[] = "velocity";
static char joint_msgs__msg__State__FIELD_NAME__effort[] = "effort";

static rosidl_runtime_c__type_description__Field joint_msgs__msg__State__FIELDS[] = {
  {
    {joint_msgs__msg__State__FIELD_NAME__name, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__State__FIELD_NAME__sequence, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__State__FIELD_NAME__position, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__State__FIELD_NAME__velocity, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__State__FIELD_NAME__effort, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
joint_msgs__msg__State__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {joint_msgs__msg__State__TYPE_NAME, 20, 20},
      {joint_msgs__msg__State__FIELDS, 5, 5},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "string name\n"
  "uint32 sequence\n"
  "float64 position\n"
  "float64 velocity\n"
  "float64 effort";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
joint_msgs__msg__State__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {joint_msgs__msg__State__TYPE_NAME, 20, 20},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 76, 76},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
joint_msgs__msg__State__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *joint_msgs__msg__State__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
