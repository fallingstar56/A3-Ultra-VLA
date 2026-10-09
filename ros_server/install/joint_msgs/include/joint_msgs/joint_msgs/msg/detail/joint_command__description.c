// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from joint_msgs:msg/JointCommand.idl
// generated code does not contain a copyright notice

#include "joint_msgs/msg/detail/joint_command__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_joint_msgs
const rosidl_type_hash_t *
joint_msgs__msg__JointCommand__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xd0, 0x4f, 0xbb, 0x0c, 0x1f, 0xe4, 0x59, 0x67,
      0x67, 0xaa, 0xa9, 0x96, 0x5f, 0x2c, 0xad, 0x1a,
      0xe5, 0x50, 0xb2, 0x6a, 0x11, 0x0d, 0x87, 0xad,
      0x01, 0xd0, 0x08, 0xe2, 0x7c, 0x45, 0xec, 0xd3,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "std_msgs/msg/detail/header__functions.h"
#include "joint_msgs/msg/detail/command__functions.h"
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t joint_msgs__msg__Command__EXPECTED_HASH = {1, {
    0x60, 0x1c, 0xba, 0x50, 0x0c, 0x36, 0xce, 0x51,
    0x2a, 0x15, 0xe3, 0xce, 0x42, 0x7d, 0x41, 0x23,
    0xdb, 0x0c, 0xd7, 0x9e, 0x15, 0x73, 0xdb, 0xaf,
    0x5d, 0xd2, 0x91, 0xed, 0x49, 0x2b, 0x21, 0x20,
  }};
static const rosidl_type_hash_t std_msgs__msg__Header__EXPECTED_HASH = {1, {
    0xf4, 0x9f, 0xb3, 0xae, 0x2c, 0xf0, 0x70, 0xf7,
    0x93, 0x64, 0x5f, 0xf7, 0x49, 0x68, 0x3a, 0xc6,
    0xb0, 0x62, 0x03, 0xe4, 0x1c, 0x89, 0x1e, 0x17,
    0x70, 0x1b, 0x1c, 0xb5, 0x97, 0xce, 0x6a, 0x01,
  }};
#endif

static char joint_msgs__msg__JointCommand__TYPE_NAME[] = "joint_msgs/msg/JointCommand";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char joint_msgs__msg__Command__TYPE_NAME[] = "joint_msgs/msg/Command";
static char std_msgs__msg__Header__TYPE_NAME[] = "std_msgs/msg/Header";

// Define type names, field names, and default values
static char joint_msgs__msg__JointCommand__FIELD_NAME__header[] = "header";
static char joint_msgs__msg__JointCommand__FIELD_NAME__joints[] = "joints";

static rosidl_runtime_c__type_description__Field joint_msgs__msg__JointCommand__FIELDS[] = {
  {
    {joint_msgs__msg__JointCommand__FIELD_NAME__header, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    },
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__JointCommand__FIELD_NAME__joints, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_UNBOUNDED_SEQUENCE,
      0,
      0,
      {joint_msgs__msg__Command__TYPE_NAME, 22, 22},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription joint_msgs__msg__JointCommand__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {joint_msgs__msg__Command__TYPE_NAME, 22, 22},
    {NULL, 0, 0},
  },
  {
    {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
joint_msgs__msg__JointCommand__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {joint_msgs__msg__JointCommand__TYPE_NAME, 27, 27},
      {joint_msgs__msg__JointCommand__FIELDS, 2, 2},
    },
    {joint_msgs__msg__JointCommand__REFERENCED_TYPE_DESCRIPTIONS, 3, 3},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&joint_msgs__msg__Command__EXPECTED_HASH, joint_msgs__msg__Command__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = joint_msgs__msg__Command__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&std_msgs__msg__Header__EXPECTED_HASH, std_msgs__msg__Header__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = std_msgs__msg__Header__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "std_msgs/Header header\n"
  "\n"
  "Command[] joints";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
joint_msgs__msg__JointCommand__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {joint_msgs__msg__JointCommand__TYPE_NAME, 27, 27},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 41, 41},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
joint_msgs__msg__JointCommand__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[4];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 4, 4};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *joint_msgs__msg__JointCommand__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *joint_msgs__msg__Command__get_individual_type_description_source(NULL);
    sources[3] = *std_msgs__msg__Header__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
