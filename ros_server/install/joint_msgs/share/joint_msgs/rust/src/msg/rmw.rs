#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "joint_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__joint_msgs__msg__JointState() -> *const std::ffi::c_void;
}

#[link(name = "joint_msgs__rosidl_generator_c")]
extern "C" {
    fn joint_msgs__msg__JointState__init(msg: *mut JointState) -> bool;
    fn joint_msgs__msg__JointState__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<JointState>, size: usize) -> bool;
    fn joint_msgs__msg__JointState__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<JointState>);
    fn joint_msgs__msg__JointState__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<JointState>, out_seq: *mut rosidl_runtime_rs::Sequence<JointState>) -> bool;
}

// Corresponds to joint_msgs__msg__JointState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub joints: rosidl_runtime_rs::Sequence<super::super::msg::rmw::State>,

}



impl Default for JointState {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !joint_msgs__msg__JointState__init(&mut msg as *mut _) {
        panic!("Call to joint_msgs__msg__JointState__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for JointState {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__JointState__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__JointState__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__JointState__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for JointState {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for JointState where Self: Sized {
  const TYPE_NAME: &'static str = "joint_msgs/msg/JointState";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__joint_msgs__msg__JointState() }
  }
}


#[link(name = "joint_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__joint_msgs__msg__JointCommand() -> *const std::ffi::c_void;
}

#[link(name = "joint_msgs__rosidl_generator_c")]
extern "C" {
    fn joint_msgs__msg__JointCommand__init(msg: *mut JointCommand) -> bool;
    fn joint_msgs__msg__JointCommand__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<JointCommand>, size: usize) -> bool;
    fn joint_msgs__msg__JointCommand__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<JointCommand>);
    fn joint_msgs__msg__JointCommand__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<JointCommand>, out_seq: *mut rosidl_runtime_rs::Sequence<JointCommand>) -> bool;
}

// Corresponds to joint_msgs__msg__JointCommand
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointCommand {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub joints: rosidl_runtime_rs::Sequence<super::super::msg::rmw::Command>,

}



impl Default for JointCommand {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !joint_msgs__msg__JointCommand__init(&mut msg as *mut _) {
        panic!("Call to joint_msgs__msg__JointCommand__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for JointCommand {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__JointCommand__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__JointCommand__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__JointCommand__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for JointCommand {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for JointCommand where Self: Sized {
  const TYPE_NAME: &'static str = "joint_msgs/msg/JointCommand";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__joint_msgs__msg__JointCommand() }
  }
}


#[link(name = "joint_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__joint_msgs__msg__State() -> *const std::ffi::c_void;
}

#[link(name = "joint_msgs__rosidl_generator_c")]
extern "C" {
    fn joint_msgs__msg__State__init(msg: *mut State) -> bool;
    fn joint_msgs__msg__State__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<State>, size: usize) -> bool;
    fn joint_msgs__msg__State__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<State>);
    fn joint_msgs__msg__State__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<State>, out_seq: *mut rosidl_runtime_rs::Sequence<State>) -> bool;
}

// Corresponds to joint_msgs__msg__State
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct State {

    // This member is not documented.
    #[allow(missing_docs)]
    pub name: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub sequence: u32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub position: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub velocity: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub effort: f64,

}



impl Default for State {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !joint_msgs__msg__State__init(&mut msg as *mut _) {
        panic!("Call to joint_msgs__msg__State__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for State {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__State__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__State__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__State__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for State {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for State where Self: Sized {
  const TYPE_NAME: &'static str = "joint_msgs/msg/State";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__joint_msgs__msg__State() }
  }
}


#[link(name = "joint_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__joint_msgs__msg__Command() -> *const std::ffi::c_void;
}

#[link(name = "joint_msgs__rosidl_generator_c")]
extern "C" {
    fn joint_msgs__msg__Command__init(msg: *mut Command) -> bool;
    fn joint_msgs__msg__Command__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Command>, size: usize) -> bool;
    fn joint_msgs__msg__Command__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Command>);
    fn joint_msgs__msg__Command__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Command>, out_seq: *mut rosidl_runtime_rs::Sequence<Command>) -> bool;
}

// Corresponds to joint_msgs__msg__Command
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Command {

    // This member is not documented.
    #[allow(missing_docs)]
    pub name: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub sequence: u32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub position: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub velocity: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub effort: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub stiffness: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub damping: f64,

}



impl Default for Command {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !joint_msgs__msg__Command__init(&mut msg as *mut _) {
        panic!("Call to joint_msgs__msg__Command__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Command {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__Command__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__Command__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { joint_msgs__msg__Command__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Command {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Command where Self: Sized {
  const TYPE_NAME: &'static str = "joint_msgs/msg/Command";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__joint_msgs__msg__Command() }
  }
}


