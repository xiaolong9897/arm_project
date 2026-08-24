import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class SnapshotPrinter(Node):
    def __init__(self):
        super().__init__('snapshot_printer')
        self.sub = self.create_subscription(
            String, '/task_scene_snapshot', self.cb, 10
        )

    def cb(self, msg): 
        data = json.loads(msg.data)
        print(json.dumps(data, indent=2, ensure_ascii=False))

        data_list=data["movable_objects"];
        for i in data_list:
            print(i["body_name"],end='\t')
            t_list=i["position_xyz"]
            # print(t_list)
            
            for t in t_list:
                print(round(t,3),end=' ')
            print("")

        print("-----------------")
        raise SystemExit  # 打印一条就退出

rclpy.init()
rclpy.spin(SnapshotPrinter())
rclpy.shutdown()