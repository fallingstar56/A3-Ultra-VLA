// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from joint_msgs:msg/Command.idl
// generated code does not contain a copyright notice

#include "joint_msgs/msg/detail/command__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_joint_msgs
const rosidl_type_hash_t *
joint_msgs__msg__Command__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x60, 0x1c, 0xba, 0x50, 0x0c, 0x36, 0xce, 0x51,
      0x2a, 0x15, 0xe3, 0xce, 0x42, 0x7d, 0x41, 0x23,
      0xdb, 0x0c, 0xd7, 0x9e, 0x15, 0x73, 0xdb, 0xaf,
      0x5d, 0xd2, 0x91, 0xed, 0x49, 0x2b, 0x21, 0x20,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char joint_msgs__msg__Command__TYPE_NAME[] = "joint_msgs/msg/Command";

// Define type names, field names, and default values
static char joint_msgs__msg__Command__FIELD_NAME__name[] = "name";
static char joint_msgs__msg__Command__FIELD_NAME__sequence[] = "sequence";
static char joint_msgs__msg__Command__FIELD_NAME__position[] = "position";
static char joint_msgs__msg__Command__FIELD_NAME__velocity[] = "velocity";
static char joint_msgs__msg__Command__FIELD_NAME__effort[] = "effort";
static char joint_msgs__msg__Command__FIELD_NAME__stiffness[] = "stiffness";
static char joint_msgs__msg__Command__FIELD_NAME__damping[] = "damping";

static rosidl_runtime_c__type_description__Field joint_msgs__msg__Command__FIELDS[] = {
  {
    {joint_msgs__msg__Command__FIELD_NAME__name, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__Command__FIELD_NAME__sequence, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__Command__FIELD_NAME__position, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__Command__FIELD_NAME__velocity, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__Command__FIELD_NAME__effort, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__Command__FIELD_NAME__stiffness, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__Command__FIELD_NAME__damping, 7, 7},
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
joint_msgs__msg__Command__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {joint_msgs__msg__Command__TYPE_NAME, 22, 22},
      {joint_msgs__msg__Command__FIELDS, 7, 7},
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
  "float64 effort\n"
  "float64 stiffness\n"
  "float64 damping";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
joint_msgs__msg__Command__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {joint_msgs__msg__Command__TYPE_NAME, 22, 22},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 110, 110},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
joint_msgs__msg__Command__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *joint_msgs__msg__Command__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
