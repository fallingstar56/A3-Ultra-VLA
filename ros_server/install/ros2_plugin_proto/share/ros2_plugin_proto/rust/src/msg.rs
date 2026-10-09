#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to ros2_plugin_proto__msg__RosMsgWrapper

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RosMsgWrapper {

    // This member is not documented.
    #[allow(missing_docs)]
    pub serialization_type: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub context: Vec<std::string::String>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub data: Vec<u8>,

}



impl Default for RosMsgWrapper {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::RosMsgWrapper::default())
  }
}

impl rosidl_runtime_rs::Message for RosMsgWrapper {
  type RmwMsg = super::msg::rmw::RosMsgWrapper;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        serialization_type: msg.serialization_type.as_str().into(),
        context: msg.context
          .into_iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        data: msg.data.into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        serialization_type: msg.serialization_type.as_str().into(),
        context: msg.context
          .iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        data: msg.data.as_slice().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      serialization_type: msg.serialization_type.to_string(),
      context: msg.context
          .into_iter()
          .map(|elem| elem.to_string())
          .collect(),
      data: msg.data
          .into_iter()
          .collect(),
    }
  }
}


