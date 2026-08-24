import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class SnapshotPrinter(Node):

    def __init__(self):
        super().__init__('task_scene_snapshot_printer')

        self.subscription = self.create_subscription(
            String,
            '/task_scene_snapshot',
            self.callback,
            10
        )

    def callback(self, msg: String):
        self.get_logger().info('Received one snapshot:')
        print(type(msg))
        try:
            data = json.loads(msg.data)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(msg.data)
        # 打印完一条就退出
        raise SystemExit

def main():
    rclpy.init()
    node = SnapshotPrinter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
