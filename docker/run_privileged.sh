#!/bin/bash
set -e

xhost +local:root >/dev/null 2>&1 || true
docker compose up -d --build
docker exec -it ros2_humble_privileged bash

