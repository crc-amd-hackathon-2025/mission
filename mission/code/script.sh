#!/usr/bin/bash

rm -rf /home/clementverrier/.cache/huggingface/lerobot/crc-amd-hackathon-2025/eval_pi05-grab-cam-2

lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
  --robot.id=SO101_follower \
  --display_data=true \
  --dataset.repo_id=crc-amd-hackathon-2025/eval_pi05-grab-cam-2 \
  --dataset.single_task="Grab the camera" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=SO101_leader \
  --policy.path=crc-amd-hackathon-2025/pi05-grab-cam-2 \
  --dataset.push_to_hub=false \
  --policy.compile_model=false