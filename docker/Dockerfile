FROM osrf/ros:humble-desktop

  ENV DEBIAN_FRONTEND=noninteractive
  ENV LANG=C.UTF-8
  ENV LC_ALL=C.UTF-8

  SHELL ["/bin/bash", "-c"]

  RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      curl \
      wget \
      vim \
      python3-pip \
      python3-colcon-common-extensions \
      ros-dev-tools \
      iputils-ping \
      net-tools \
      usbutils \
      && rm -rf /var/lib/apt/lists/*

  RUN mkdir -p /root/ros2_ws/src
  WORKDIR /root/ros2_ws

  COPY ./ros_entrypoint.sh /ros_entrypoint.sh
  RUN chmod +x /ros_entrypoint.sh

  ENTRYPOINT ["/ros_entrypoint.sh"]
  CMD ["bash"]

