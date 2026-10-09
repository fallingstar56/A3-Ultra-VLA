from setuptools import setup, find_packages

package_name = "a3_server"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "requests", "numpy", "opencv-python", "flask", "protobuf",
                       "pandas", "h5py"],
    zip_safe=True,
    maintainer="agibot",
    maintainer_email="dev@agibot.com",
    description="A3 机器人 ROS2 底层服务节点（无相机；含 TA whole_body_command 订阅）",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "server = a3_server.server_node:main",
            "replay = a3_server.replay:main",
            "reset_pose = a3_server.reset_pose:main",
        ],
    },
)
