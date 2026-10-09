"""O10 灵巧手主动关节弧度 <-> 执行器输入 (0-4096) 的转换。

跟 a2_server.replay.OmnihandCtrl 等价 (a2/a3 各持一份, 部署独立, 避免跨包依赖)。
仅用于 hand_kind=hand 路径; gripper 路径数据本身已经是 actuator, 不走这个转换。
"""


class OmnihandCtrl:
    """主动关节弧度 <-> 执行器输入 (0-4096) 的转换。"""

    def __init__(self, hand_type: bool = False):
        self.hand_type = hand_type
        self.active_joint_count = 10
        self.active_joint_max = [1.12, 0.05, 0.8416, 0, 1.48, 1.48, 0.17, 1.48, 0.19, 1.48]
        self.motor_max = [1.12, 0.05, 1.33, 0, 1.43, 1.43, 0.17, 1.43, 0.19, 1.43]
        self.active_joint_min = [-0.03, -1.64, 0.0, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.motor_min = [-0.03, -1.64, 0.0, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_pos_direction = [-1, -1, -1, -1, 1, 1, -1, 1, -1, 1]

        if not hand_type:
            self.actuator_max = [0, 0, 0, 0, 0, 0, 4096, 0, 0, 0]
            self.actuator_min = [4096, 4096, 4096, 4096, 4096, 4096, 0, 4096, 4096, 4096]
        else:
            left_active_joint_max = self.active_joint_max.copy()
            left_active_joint_min = self.active_joint_min.copy()
            left_motor_max = self.motor_max.copy()
            left_motor_min = self.motor_min.copy()
            for i in range(10):
                if self.left_pos_direction[i] == -1:
                    left_active_joint_max[i] = -self.active_joint_min[i]
                    left_active_joint_min[i] = -self.active_joint_max[i]
                    left_motor_max[i] = -self.motor_min[i]
                    left_motor_min[i] = -self.motor_max[i]
            self.active_joint_min = left_active_joint_min
            self.active_joint_max = left_active_joint_max
            self.motor_max = left_motor_max
            self.motor_min = left_motor_min
            self.actuator_max = [4095, 0, 4095, 0, 0, 0, 4095, 0, 0, 0]
            self.actuator_min = [0, 4095, 0, 4095, 4095, 4095, 0, 4095, 4095, 4095]

        self.finger_mcp2motor_poly = [0.00944480234881967, 0.455882677008572, 0.683758090072141,
                                      -0.916673507519311, 0.459387725400186]
        self.finger_motor2mcp_poly = [-0.000257594494466942, 1.57144033291557, 0.217395210463076,
                                      -0.768328304426314, 0.248168989312469]
        self.right_thumb_mcp2motor_poly = [0.00126371020922368, 0.919140692758276, 0.550958572722048,
                                           -0.785384985903032, 1.25635285116862]
        self.left_thumb_mcp2motor_poly = [-0.00126371020922368, 0.919140692758276, -0.550958572722048,
                                          -0.785384985903032, -1.25635285116862]
        self.right_thumb_motor2mcp_poly = [-0.000677604838762652, 1.05175893483608, -0.280133575638901,
                                           -0.115384415912668, 0.0676128925382166]
        self.left_thumb_motor2mcp_poly = [0.000677604838762652, 1.05175893483608, 0.280133575638901,
                                          -0.115384415912668, -0.0676128925382166]
        if hand_type:
            self.thumb_mcp2dip_poly = [0.0, 1.846, 0.853, 0.280]
        else:
            self.thumb_mcp2dip_poly = [0.0, 1.846, -0.853, 0.280]

    def _calculate_power(self, x, coeffs):
        result, power = 0.0, 1.0
        for c in coeffs:
            result += c * power
            power *= x
        return result

    def _clamp_joint_pos(self, pos):
        for i in range(self.active_joint_count):
            pos[i] = max(self.active_joint_min[i], min(self.active_joint_max[i], pos[i]))
        return pos

    def active_joint_pos_to_actuator_input(self, active_joint_pos):
        assert len(active_joint_pos) == self.active_joint_count
        hand_pos = list(active_joint_pos)
        self._clamp_joint_pos(hand_pos)
        if not self.hand_type:
            hand_pos[2] = self._calculate_power(hand_pos[2], self.right_thumb_mcp2motor_poly)
        else:
            hand_pos[2] = self._calculate_power(hand_pos[2], self.left_thumb_mcp2motor_poly)
        for idx in [4, 5, 7, 9]:
            hand_pos[idx] = self._calculate_power(hand_pos[idx], self.finger_mcp2motor_poly)
        actuator_input = []
        for i in range(self.active_joint_count):
            denom = self.motor_max[i] - self.motor_min[i]
            if abs(denom) < 1e-9:
                mapped = self.actuator_min[i]
            else:
                mapped = (hand_pos[i] - self.motor_min[i]) * (self.actuator_max[i] - self.actuator_min[i]) / denom + self.actuator_min[i]
            actuator_input.append(int(mapped))
        return actuator_input

    def actuator_input_to_active_joint_pos(self, actuator_input):
        assert len(actuator_input) == self.active_joint_count
        motor_pos = []
        for i in range(self.active_joint_count):
            denom = self.actuator_max[i] - self.actuator_min[i]
            if abs(denom) < 1e-9:
                pos = self.motor_min[i]
            else:
                pos = (actuator_input[i] - self.actuator_min[i]) * (self.motor_max[i] - self.motor_min[i]) / denom + self.motor_min[i]
            motor_pos.append(pos)
        if not self.hand_type:
            motor_pos[2] = self._calculate_power(motor_pos[2], self.right_thumb_motor2mcp_poly)
        else:
            motor_pos[2] = self._calculate_power(motor_pos[2], self.left_thumb_motor2mcp_poly)
        for idx in [4, 5, 7, 9]:
            motor_pos[idx] = self._calculate_power(motor_pos[idx], self.finger_motor2mcp_poly)
        self._clamp_joint_pos(motor_pos)
        return motor_pos
