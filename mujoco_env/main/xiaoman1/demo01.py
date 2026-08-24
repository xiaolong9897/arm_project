import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

class JointStatePrinter(Node):

    def __init__(self):
        super().__init__('joint_state_printer')

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states_sim',
            self.callback,
            10
        )

    def callback(self, msg: JointState):
        self.get_logger().info('Received one JointState:')
        print(type(msg))
        print('name      :', msg.name)
        print('position  :', msg.position)
        print('velocity  :', msg.velocity)
        print('effort    :', msg.effort)

        print(list(msg.position))
        for i in range(len(msg.name)):
            # print(list(msg.name)[i],"  ",round(list(msg.position)[i]*(180.000/3.1415926),3),
            #       round(list(msg.velocity)[i]*(180.000/3.1415926),3),round(list(msg.effort)[i]*(180.000/3.1415926),3))
            print(list(msg.name)[i],"  ",round(list(msg.position)[i]*(180.000/3.1415926),3))

        # 从消息中按名字提取前 6 个关节角，避免仅依赖位置顺序。
        joint_names = [f'joint{i}' for i in range(1, 7)]
        name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
        q = [float(name_to_pos[name]) for name in joint_names]

        # 每个关节的固定安装位姿，来自 URDF 的 origin(xyz, rpy)。
        fixed_joint_xyz_rpy = [
            ([0.0, 0.0, 0.084], [0.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.068718], [1.5708, 0.0, -1.5708]),
            ([0.0, 0.30025, 0.0], [-3.1416, 0.0, 0.0]),
            ([0.0, -0.15558, 3.5e-05], [1.5708, -1.5708, 0.0]),
            ([-3.5e-05, 1.5362e-05, 0.064223], [1.5708, 0.0, 0.0]),
            ([-0.00047523, 0.095552, 0.0], [-1.5709, 0.0, 0.0]),
        ]

        # 关节旋转轴（相对于各自局部坐标系）
        joint_axes_local = [
            [0, 0,  1],  # joint1: Z轴
            [0, 0, -1],  # joint2: -Z轴
            [0, 0,  1],  # joint3: Z轴
            [0, 0,  1],  # joint4: Z轴
            [1, 0,  0],  # joint5: X轴
            [0, 0,  1],  # joint6: Z轴
        ]
        # 单位矩阵
        I = np.eye(4)
        
        # 定义旋转矩阵函数
        def rot_x(theta):
            return np.array([
                [1, 0            , 0             , 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta) , 0],
                [0, 0            , 0             , 1]
            ])


        def rot_y(theta):
            return np.array([
                [np.cos(theta) , 0, np.sin(theta), 0],
                [0             , 1, 0            , 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0             , 0, 0            , 1]
            ])
        
        def rot_z(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta) , 0, 0],
                [0            , 0             , 1, 0],
                [0            , 0             , 0, 1]
            ])        
        # 定义平移矩阵函数
        def trans(x, y, z):
            return np.array([
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1]
            ])

        def rpy_to_transform(roll, pitch, yaw):
            return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
        # “绕固定笛卡尔轴旋转”
        def axis_angle_transform(axis, theta):
            if axis[0] == 1:
                return rot_x(theta)
            if axis[0] == -1:
                return rot_x(-theta)
            if axis[1] == 1:
                return rot_y(theta)
            if axis[1] == -1:
                return rot_y(-theta)
            if axis[2] == 1:
                return rot_z(theta)
            if axis[2] == -1:
                return rot_z(-theta)
            return I

        # 正运动学计算
        T = I  # 初始变换矩阵
        for i in range(6):
            # 固定安装位姿 = 固定平移 + 固定旋转
            xyz, rpy = fixed_joint_xyz_rpy[i]
            T_fixed_i = trans(*xyz) @ rpy_to_transform(*rpy)
            T = T @ T_fixed_i

            # 关节转动
            T_rot_i = axis_angle_transform(joint_axes_local[i], q[i])
            T = T @ T_rot_i

        # 末端工具点相对 Link6 还有固定偏移。
        T_tool = trans(0.0, 0.0, 0.13/2.0)
        T = T @ T_tool

        # 提取末端执行器位置
        ee_xyz = T[:3, 3]
        print('End Effector Position:', *[round(i*100,2) for i in ee_xyz])





        # 打印完一条就退出
        raise SystemExit


def main():
    rclpy.init()
    node = JointStatePrinter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
