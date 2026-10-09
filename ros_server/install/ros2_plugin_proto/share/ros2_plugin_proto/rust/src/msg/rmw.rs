#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "ros2_plugin_proto__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__ros2_plugin_proto__msg__RosMsgWrapper() -> *const std::ffi::c_void;
}

#[link(name = "ros2_plugin_proto__rosidl_generator_c")]
extern "C" {
    fn ros2_plugin_proto__msg__RosMsgWrapper__init(msg: *mut RosMsgWrapper) -> bool;
    fn ros2_plugin_proto__msg__RosMsgWrapper__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<RosMsgWrapper>, size: usize) -> bool;
    fn ros2_plugin_proto__msg__RosMsgWrapper__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<RosMsgWrapper>);
    fn ros2_plugin_proto__msg__RosMsgWrapper__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<RosMsgWrapper>, out_seq: *mut rosidl_runtime_rs::Sequence<RosMsgWrapper>) -> bool;
}

// Corresponds to ros2_plugin_proto__msg__RosMsgWrapper
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RosMsgWrapper {

    // This member is not documented.
    #[allow(missing_docs)]
    pub serialization_type: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub context: rosidl_runtime_rs::Sequence<rosidl_runtime_rs::String>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub data: rosidl_runtime_rs::Sequence<u8>,

}



impl Default for RosMsgWrapper {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !ros2_plugin_proto__msg__RosMsgWrapper__init(&mut msg as *mut _) {
        panic!("Call to ros2_plugin_proto__msg__RosMsgWrapper__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for RosMsgWrapper {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { ros2_plugin_proto__msg__RosMsgWrapper__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { ros2_plugin_proto__msg__RosMsgWrapper__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { ros2_plugin_proto__msg__RosMsgWrapper__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for RosMsgWrapper {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for RosMsgWrapper where Self: Sized {
  const TYPE_NAME: &'static str = "ros2_plugin_proto/msg/RosMsgWrapper";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__ros2_plugin_proto__msg__RosMsgWrapper() }
  }
}


