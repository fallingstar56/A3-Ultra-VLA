#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to joint_msgs__msg__JointState

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub joints: Vec<super::msg::State>,

}



impl Default for JointState {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::JointState::default())
  }
}

impl rosidl_runtime_rs::Message for JointState {
  type RmwMsg = super::msg::rmw::JointState;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        joints: msg.joints
          .into_iter()
          .map(|elem| super::msg::State::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        joints: msg.joints
          .iter()
          .map(|elem| super::msg::State::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      joints: msg.joints
          .into_iter()
          .map(super::msg::State::from_rmw_message)
          .collect(),
    }
  }
}


// Corresponds to joint_msgs__msg__JointCommand

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointCommand {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub joints: Vec<super::msg::Command>,

}



impl Default for JointCommand {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::JointCommand::default())
  }
}

impl rosidl_runtime_rs::Message for JointCommand {
  type RmwMsg = super::msg::rmw::JointCommand;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        joints: msg.joints
          .into_iter()
          .map(|elem| super::msg::Command::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        joints: msg.joints
          .iter()
          .map(|elem| super::msg::Command::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      joints: msg.joints
          .into_iter()
          .map(super::msg::Command::from_rmw_message)
          .collect(),
    }
  }
}


// Corresponds to joint_msgs__msg__State

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct State {

    // This member is not documented.
    #[allow(missing_docs)]
    pub name: std::string::String,


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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::State::default())
  }
}

impl rosidl_runtime_rs::Message for State {
  type RmwMsg = super::msg::rmw::State;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        name: msg.name.as_str().into(),
        sequence: msg.sequence,
        position: msg.position,
        velocity: msg.velocity,
        effort: msg.effort,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        name: msg.name.as_str().into(),
      sequence: msg.sequence,
      position: msg.position,
      velocity: msg.velocity,
      effort: msg.effort,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      name: msg.name.to_string(),
      sequence: msg.sequence,
      position: msg.position,
      velocity: msg.velocity,
      effort: msg.effort,
    }
  }
}


// Corresponds to joint_msgs__msg__Command

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Command {

    // This member is not documented.
    #[allow(missing_docs)]
    pub name: std::string::String,


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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Command::default())
  }
}

impl rosidl_runtime_rs::Message for Command {
  type RmwMsg = super::msg::rmw::Command;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        name: msg.name.as_str().into(),
        sequence: msg.sequence,
        position: msg.position,
        velocity: msg.velocity,
        effort: msg.effort,
        stiffness: msg.stiffness,
        damping: msg.damping,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        name: msg.name.as_str().into(),
      sequence: msg.sequence,
      position: msg.position,
      velocity: msg.velocity,
      effort: msg.effort,
      stiffness: msg.stiffness,
      damping: msg.damping,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      name: msg.name.to_string(),
      sequence: msg.sequence,
      position: msg.position,
      velocity: msg.velocity,
      effort: msg.effort,
      stiffness: msg.stiffness,
      damping: msg.damping,
    }
  }
}


