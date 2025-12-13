#!/usr/bin/env python3
"""
Keyboard control for SO101 robot + independent YOLO streaming display

Changes vs your SO100 version:
- Uses SO101Follower + SO101FollowerConfig
- Uses id=... + use_degrees=True so observations/actions are in degrees (not servo ticks)
- Removes manual apply_joint_calibration on observations (avoid double calibration)
- Adds a stop_event so video window quit also stops control loop cleanly

YOLO stream displays object detection but does NOT control the robot
Video stream and robot control are independent, synchronized only for quitting
"""

import time
import logging
import traceback
import math
import cv2
import threading
from ultralytics import YOLOE

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Optional fine trim (usually keep identity when use_degrees=True)
# Format: [joint_name, add_offset_degrees, scale_factor]
# Applied ONLY if APPLY_FINE_TRIM=True
APPLY_FINE_TRIM = False
JOINT_FINE_TRIM = [
    ["shoulder_pan", 0.0, 1.0],
    ["shoulder_lift", 0.0, 1.0],
    ["elbow_flex", 0.0, 1.0],
    ["wrist_flex", 0.0, 1.0],
    ["wrist_roll", 0.0, 1.0],
    ["gripper", 0.0, 1.0],
]

def apply_fine_trim(joint_name: str, deg_value: float) -> float:
    if not APPLY_FINE_TRIM:
        return float(deg_value)
    for name, offset, scale in JOINT_FINE_TRIM:
        if name == joint_name:
            return (float(deg_value) + float(offset)) * float(scale)
    return float(deg_value)

def inverse_kinematics_2d(x, y, l1=0.1159, l2=0.1350):
    """
    2-link planar IK used by your original script.
    WARNING: this is a simplified model and the final degree mapping depends on conventions.
    It can work as a starting point, but expect tuning.

    Returns: (joint2_deg, joint3_deg) as degrees (your original convention)
    """
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r = math.sqrt(x * x + y * y)
    r_max = l1 + l2
    if r > r_max:
        s = r_max / r
        x *= s
        y *= s
        r = r_max

    r_min = abs(l1 - l2)
    if 0 < r < r_min:
        s = r_min / r
        x *= s
        y *= s
        r = r_min

    cos_theta2 = -(r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    theta2 = math.pi - math.acos(cos_theta2)

    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset

    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))

    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    # Your original repere conversion
    joint2_deg = 90 - joint2_deg
    joint3_deg = joint3_deg - 90

    return joint2_deg, joint3_deg

def move_to_zero_position(robot, stop_event: threading.Event, duration=3.0, kp=0.5, control_freq=50):
    print("Using P control to slowly move robot to zero position...")

    zero_positions = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    }

    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    print(f"Move to zero in {duration}s at {control_freq}Hz, kp={kp}")

    for step in range(total_steps):
        if stop_event.is_set():
            print("Stop requested, abort move_to_zero.")
            return

        obs = robot.get_observation()
        current_positions = {}
        for k, v in obs.items():
            if k.endswith(".pos"):
                name = k.removesuffix(".pos")
                current_positions[name] = apply_fine_trim(name, float(v))

        action = {}
        for j, target in zero_positions.items():
            if j in current_positions:
                cur = current_positions[j]
                err = target - cur
                new_pos = cur + kp * err
                action[f"{j}.pos"] = float(new_pos)

        if action:
            robot.send_action(action)

        if step % max(1, (control_freq // 2)) == 0:
            progress = (step / total_steps) * 100
            print(f"Moving to zero: {progress:.1f}%")

        time.sleep(step_time)

    print("Robot moved to zero position.")

def return_to_start_position(robot, stop_event: threading.Event, start_positions, kp=0.2, control_freq=50):
    print("Returning to start position...")

    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)

    for _ in range(max_steps):
        if stop_event.is_set():
            print("Stop requested, abort return_to_start.")
            return

        obs = robot.get_observation()
        current_positions = {}
        for k, v in obs.items():
            if k.endswith(".pos"):
                name = k.removesuffix(".pos")
                current_positions[name] = float(v)

        action = {}
        total_err = 0.0

        for j, target in start_positions.items():
            if j in current_positions:
                cur = current_positions[j]
                err = float(target) - float(cur)
                total_err += abs(err)
                action[f"{j}.pos"] = float(cur + kp * err)

        if action:
            robot.send_action(action)

        if total_err < 2.0:
            print("Returned to start position.")
            return

        time.sleep(control_period)

    print("Return to start completed (timeout).")

def video_stream_loop(model, cap, stop_event: threading.Event):
    print("Starting YOLO video stream...")

    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            results = model(frame)
            if not results or not hasattr(results[0], "boxes") or results[0].boxes is None or len(results[0].boxes) == 0:
                annotated = frame
            else:
                annotated = results[0].plot()

            cv2.imshow("YOLO Live Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                stop_event.set()
                break

        except Exception as e:
            print(f"Video stream error: {e}")
            stop_event.set()
            break

    print("Video stream ended.")
    cv2.destroyAllWindows()

def p_control_loop(robot, keyboard, stop_event: threading.Event, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50):
    control_period = 1.0 / control_freq
    pitch = 0.0
    pitch_step = 1.0

    joint_controls = {
        "q": ("shoulder_pan", -1.0),
        "a": ("shoulder_pan", 1.0),
        "t": ("wrist_roll", -1.0),
        "g": ("wrist_roll", 1.0),
        "y": ("gripper", -1.0),
        "h": ("gripper", 1.0),
    }

    xy_controls = {
        "w": ("x", -0.004),
        "s": ("x", 0.004),
        "e": ("y", -0.004),
        "d": ("y", 0.004),
    }

    step_counter = 0
    print(f"Starting P control loop at {control_freq}Hz, kp={kp}")

    while not stop_event.is_set():
        try:
            keyboard_action = keyboard.get_action()

            if keyboard_action:
                for key in keyboard_action.keys():
                    if key == "x":
                        print("Exit command detected, returning to start position...")
                        stop_event.set()
                        break

                    if key == "r":
                        pitch += pitch_step
                        print(f"Pitch: {pitch:.3f}")
                    elif key == "f":
                        pitch -= pitch_step
                        print(f"Pitch: {pitch:.3f}")

                    if key in joint_controls:
                        joint_name, delta = joint_controls[key]
                        cur = float(target_positions.get(joint_name, 0.0))
                        new = cur + delta
                        target_positions[joint_name] = new
                        print(f"Target {joint_name}: {cur:.2f} -> {new:.2f}")

                    elif key in xy_controls:
                        coord, delta = xy_controls[key]
                        if coord == "x":
                            current_x += delta
                        else:
                            current_y += delta

                        j2, j3 = inverse_kinematics_2d(current_x, current_y)
                        target_positions["shoulder_lift"] = float(j2)
                        target_positions["elbow_flex"] = float(j3)
                        print(f"EE x={current_x:.4f} y={current_y:.4f} -> shoulder_lift={j2:.2f} elbow_flex={j3:.2f}")

            # Pitch affects wrist_flex
            if "shoulder_lift" in target_positions and "elbow_flex" in target_positions:
                target_positions["wrist_flex"] = -float(target_positions["shoulder_lift"]) - float(target_positions["elbow_flex"]) + float(pitch)

                step_counter += 1
                if step_counter % 100 == 0:
                    print(f"Pitch={pitch:.2f}, wrist_flex target={target_positions['wrist_flex']:.2f}")

            obs = robot.get_observation()
            current_positions = {}
            for k, v in obs.items():
                if k.endswith(".pos"):
                    name = k.removesuffix(".pos")
                    current_positions[name] = apply_fine_trim(name, float(v))

            action = {}
            for j, target in target_positions.items():
                if j in current_positions:
                    cur = current_positions[j]
                    err = float(target) - float(cur)
                    action[f"{j}.pos"] = float(cur + kp * err)

            if action:
                robot.send_action(action)

            time.sleep(control_period)

        except KeyboardInterrupt:
            print("KeyboardInterrupt, stopping...")
            stop_event.set()
            break
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            stop_event.set()
            break

    # Return to start when loop ends
    try:
        return_to_start_position(robot, stop_event, start_positions, kp=0.2, control_freq=control_freq)
    except Exception:
        pass

def list_cameras(max_index=6):
    available = []
    for idx in range(max_index):
        cap_test = cv2.VideoCapture(idx)
        if cap_test.isOpened():
            available.append(idx)
            cap_test.release()
    return available

def main():
    print("LeRobot SO101 Keyboard Control + Independent YOLO Display")
    print("=" * 60)

    robot = None
    keyboard = None
    cap = None
    stop_event = threading.Event()

    try:
        # Robust imports depending on lerobot version
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except Exception:
            from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

        port = input("SO101 robot USB port (default /dev/ttyACM1): ").strip() or "/dev/ttyACM1"
        robot_id = input("Robot id (default SO101_follower): ").strip() or "SO101_follower"

        print(f"Connecting: port={port}, id={robot_id}, use_degrees=True")
        robot_config = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
        robot = SO101Follower(robot_config)

        keyboard = KeyboardTeleop(KeyboardTeleopConfig())

        robot.connect()
        keyboard.connect()

        print("Devices connected successfully.")

        # Calibration choice
        while True:
            calibrate_choice = input("Recalibrate now? (y/n): ").strip().lower()
            if calibrate_choice in ["y", "yes"]:
                print("Starting calibration...")
                robot.calibrate()
                print("Calibration completed.")
                break
            if calibrate_choice in ["n", "no"]:
                print("Using existing calibration file.")
                break
            print("Please enter y or n.")

        # Read start positions (degrees)
        print("Reading start joint angles (degrees)...")
        start_obs = robot.get_observation()
        start_positions = {}
        for k, v in start_obs.items():
            if k.endswith(".pos"):
                name = k.removesuffix(".pos")
                start_positions[name] = float(v)

        print("Start joint angles:")
        for j, p in start_positions.items():
            print(f"  {j}: {p:.2f} deg")

        move_to_zero_position(robot, stop_event, duration=3.0, kp=0.5, control_freq=50)

        # Targets in degrees
        target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

        # EE init (your original values)
        current_x, current_y = 0.1629, 0.1131
        print(f"Init end effector position: x={current_x:.4f}, y={current_y:.4f}")

        # YOLO init
        model = YOLOE("yoloe-11l-seg.pt")

        print("\n" + "=" * 60)
        print("YOLO Detection Target Setup")
        print("=" * 60)
        target_input = input("Objects to detect (comma-separated, default bottle): ").strip()
        if not target_input:
            target_objects = ["bottle"]
        else:
            target_objects = [obj.strip() for obj in target_input.split(",") if obj.strip()]

        print(f"Detection targets: {target_objects}")
        model.set_classes(target_objects, model.get_text_pe(target_objects))

        cameras = list_cameras()
        if not cameras:
            print("No cameras found.")
            return

        print(f"Available cameras: {cameras}")
        selected = int(input(f"Select camera index from {cameras}: ").strip())
        cap = cv2.VideoCapture(selected)
        if not cap.isOpened():
            print("Camera open failed.")
            return

        print("Control instructions:")
        print("- Q/A: Joint1 shoulder_pan decrease/increase")
        print("- W/S: EE x move (updates joint2+3)")
        print("- E/D: EE y move (updates joint2+3)")
        print("- R/F: Pitch adjust (affects wrist_flex)")
        print("- T/G: Joint5 wrist_roll decrease/increase")
        print("- Y/H: Joint6 gripper close/open")
        print("- X: Exit (returns to start)")
        print("")
        print("YOLO window: Q or ESC to quit (also stops control loop)")
        print("=" * 60)

        video_thread = threading.Thread(target=video_stream_loop, args=(model, cap, stop_event), daemon=True)
        video_thread.start()

        p_control_loop(robot, keyboard, stop_event, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50)

    except Exception as e:
        print(f"Program failed: {e}")
        traceback.print_exc()
        print("Checklist:")
        print("1) Robot connected and powered")
        print("2) Correct USB port")
        print("3) Permissions on /dev/ttyACM*")
        print("4) lerobot version contains SO101 follower")
    finally:
        try:
            stop_event.set()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if keyboard is not None:
                keyboard.disconnect()
        except Exception:
            pass
        try:
            if robot is not None:
                robot.disconnect()
        except Exception:
            pass
        print("Program ended.")

if __name__ == "__main__":
    main()
