#!/usr/bin/env python3
"""
SO101: Keyboard EE control + YOLO display, with optional YOLO follow (toggle with 'p').

Requested behavior:
- Default: YOLO only displays detections, no robot control from vision.
- Press 'p' to toggle YOLO follow ON/OFF.
- Console prints only on events:
  - key press that changes something
  - follow actually applies a non-zero movement
- Default inputs:
    port = /dev/ttyACM1
    robot_id = SO101_follower
"""

import time
import math
import cv2
import threading
import traceback
import logging
from ultralytics import YOLOE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("so101_yolo_follow")

# =========================
# Tuning knobs (safe-ish defaults)
# =========================

CONTROL_FREQ = 50
KP = 0.5

# Keyboard EE step (meters)
EE_STEP = 0.004

# Pitch adjustment (degrees per keypress)
PITCH_STEP = 1.0

# Follow gains:
# Use normalized offsets: dx_norm, dy_norm in [-1, 1]
K_PAN_RAD_PER_NORM = -0.35   # rad per normalized horizontal offset
K_Y_M_PER_NORM = 0.02       # meters per normalized vertical offset

# Follow deadzones on normalized offsets
PAN_DEADZONE_NORM = 0.10
Y_DEADZONE_NORM = 0.06

# Clamp follow deltas per frame
PAN_MAX_DEG_PER_FRAME = 3.0
Y_MAX_M_PER_FRAME = 0.005

# If you want a bit of extra manual trim (usually keep OFF with use_degrees=True)
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

# =========================
# Simple planar IK (your original convention)
# =========================
def inverse_kinematics_2d(x, y, l1=0.1159, l2=0.1350):
    """
    2-link planar IK used in your SO100 script.
    Returns (joint2_deg, joint3_deg) in your original degree convention.
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

    # These clamps are from your original; they may not match SO101 URDF exactly
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))

    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    joint2_deg = 90 - joint2_deg
    joint3_deg = joint3_deg - 90

    return joint2_deg, joint3_deg

# =========================
# Motion helpers
# =========================
def move_to_zero_position(robot, stop_event: threading.Event, duration=3.0, kp=0.5, control_freq=50):
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

    for _ in range(total_steps):
        if stop_event.is_set():
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
                action[f"{j}.pos"] = float(cur + kp * err)

        if action:
            robot.send_action(action)

        time.sleep(step_time)

def return_to_start_position(robot, stop_event: threading.Event, start_positions, kp=0.2, control_freq=50):
    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)

    for _ in range(max_steps):
        if stop_event.is_set():
            # still try to go home a bit, but don't block forever
            pass

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
            return

        time.sleep(control_period)

# =========================
# Shared state
# =========================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.follow_enabled = False
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# =========================
# YOLO display + optional follow
# =========================
def yolo_loop(model, cap, robot, stop_event: threading.Event, shared: SharedState):
    """
    Always displays detections.
    If follow_enabled is True, it updates target_positions based on bbox center offset.
    """
    last_follow_print_ts = 0.0

    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            results = model(frame)
            annotated = frame
            best_box = None

            if results and hasattr(results[0], "boxes") and results[0].boxes is not None and len(results[0].boxes) > 0:
                annotated = results[0].plot()

                # pick largest box (stable heuristic)
                boxes_xyxy = results[0].boxes.xyxy
                areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
                idx = int(areas.argmax().item())
                best_box = boxes_xyxy[idx].tolist()

            cv2.imshow("YOLO Live Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                stop_event.set()
                break

            # Follow logic
            with shared.lock:
                follow_on = shared.follow_enabled

            if follow_on and best_box is not None:
                x1, y1, x2, y2 = best_box
                h, w = frame.shape[:2]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                dx_norm = (cx - (w * 0.5)) / (w * 0.5)
                dy_norm = (cy - (h * 0.5)) / (h * 0.5)

                # Deadzones
                if abs(dx_norm) < PAN_DEADZONE_NORM:
                    dx_norm = 0.0
                if abs(dy_norm) < Y_DEADZONE_NORM:
                    dy_norm = 0.0

                # Compute deltas
                delta_pan_deg = math.degrees(K_PAN_RAD_PER_NORM * dx_norm)
                delta_pan_deg = clamp(delta_pan_deg, -PAN_MAX_DEG_PER_FRAME, PAN_MAX_DEG_PER_FRAME)

                delta_y = K_Y_M_PER_NORM * dy_norm
                delta_y = clamp(delta_y, -Y_MAX_M_PER_FRAME, Y_MAX_M_PER_FRAME)

                did_move = (abs(delta_pan_deg) > 1e-6) or (abs(delta_y) > 1e-9)

                if did_move:
                    with shared.lock:
                        # Update shoulder_pan target
                        shared.target_positions["shoulder_pan"] = float(shared.target_positions["shoulder_pan"] + delta_pan_deg)

                        # Update EE y and recompute joint2/3
                        shared.current_y = float(shared.current_y + delta_y)
                        j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                        shared.target_positions["shoulder_lift"] = float(j2)
                        shared.target_positions["elbow_flex"] = float(j3)

                    # Print only if something actually moved, and not too spammy
                    now = time.time()
                    if now - last_follow_print_ts > 0.08:
                        last_follow_print_ts = now
                        print(f"[FOLLOW] pan {delta_pan_deg:+.2f} deg, y {delta_y:+.4f} m, dx={dx_norm:+.2f}, dy={dy_norm:+.2f}")

        except Exception as e:
            print(f"YOLO loop error: {e}")
            traceback.print_exc()
            stop_event.set()
            break

    cv2.destroyAllWindows()

# =========================
# Keyboard + P-control loop
# =========================
def p_control_loop(robot, keyboard, stop_event: threading.Event, shared: SharedState, start_positions):
    control_period = 1.0 / CONTROL_FREQ
    step_counter = 0

    joint_controls = {
        "q": ("shoulder_pan", -1.0),
        "a": ("shoulder_pan", 1.0),
        "t": ("wrist_roll", -1.0),
        "g": ("wrist_roll", 1.0),
        "y": ("gripper", -1.0),
        "h": ("gripper", 1.0),
    }

    xy_controls = {
        "w": ("x", -EE_STEP),
        "s": ("x", EE_STEP),
        "e": ("y", -EE_STEP),
        "d": ("y", EE_STEP),
    }

    while not stop_event.is_set():
        try:
            keyboard_action = keyboard.get_action()

            # Handle key events (prints only on actual action)
            if keyboard_action:
                for key in keyboard_action.keys():
                    if key == "x":
                        stop_event.set()
                        print("[KEY] exit requested")
                        break

                    if key == "p":
                        with shared.lock:
                            shared.follow_enabled = not shared.follow_enabled
                            state = shared.follow_enabled
                        print(f"[KEY] follow {'ON' if state else 'OFF'}")

                    if key == "r":
                        with shared.lock:
                            shared.pitch += PITCH_STEP
                            p = shared.pitch
                        print(f"[KEY] pitch {p:+.1f} deg")

                    if key == "f":
                        with shared.lock:
                            shared.pitch -= PITCH_STEP
                            p = shared.pitch
                        print(f"[KEY] pitch {p:+.1f} deg")

                    if key in joint_controls:
                        jname, delta = joint_controls[key]
                        with shared.lock:
                            cur = float(shared.target_positions.get(jname, 0.0))
                            new = cur + float(delta)
                            shared.target_positions[jname] = new
                        print(f"[KEY] {jname} target {cur:.2f} -> {new:.2f} deg")

                    if key in xy_controls:
                        coord, delta = xy_controls[key]
                        with shared.lock:
                            if coord == "x":
                                shared.current_x += float(delta)
                            else:
                                shared.current_y += float(delta)
                            j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                            shared.target_positions["shoulder_lift"] = float(j2)
                            shared.target_positions["elbow_flex"] = float(j3)
                            cx, cy = shared.current_x, shared.current_y
                        print(f"[KEY] EE x={cx:.4f} y={cy:.4f} -> shoulder_lift={j2:.2f} elbow_flex={j3:.2f}")

            # Update wrist_flex using pitch (quiet unless it changes via keypress)
            with shared.lock:
                pitch = float(shared.pitch)
                if "shoulder_lift" in shared.target_positions and "elbow_flex" in shared.target_positions:
                    shared.target_positions["wrist_flex"] = (
                        -float(shared.target_positions["shoulder_lift"]) - float(shared.target_positions["elbow_flex"]) + pitch
                    )
                targets_snapshot = dict(shared.target_positions)

            # Read robot positions (degrees) and apply P-control
            obs = robot.get_observation()
            current_positions = {}
            for k, v in obs.items():
                if k.endswith(".pos"):
                    name = k.removesuffix(".pos")
                    current_positions[name] = apply_fine_trim(name, float(v))

            action = {}
            for j, tgt in targets_snapshot.items():
                if j in current_positions:
                    cur = current_positions[j]
                    err = float(tgt) - float(cur)
                    action[f"{j}.pos"] = float(cur + KP * err)

            if action:
                robot.send_action(action)

            step_counter += 1
            time.sleep(control_period)

        except KeyboardInterrupt:
            stop_event.set()
            print("[KEY] KeyboardInterrupt")
            break
        except Exception as e:
            stop_event.set()
            print(f"P control loop error: {e}")
            traceback.print_exc()
            break

    # Go home
    try:
        return_to_start_position(robot, stop_event, start_positions, kp=0.2, control_freq=CONTROL_FREQ)
    except Exception:
        pass

def list_cameras(max_index=8):
    available = []
    for idx in range(max_index):
        cap_test = cv2.VideoCapture(idx)
        if cap_test.isOpened():
            available.append(idx)
            cap_test.release()
    return available

# =========================
# Main
# =========================
def main():
    print("SO101 YOLO follow toggle (p) + keyboard control")
    print("=" * 60)

    robot = None
    keyboard = None
    cap = None
    stop_event = threading.Event()
    shared = SharedState()

    try:
        # Imports depending on your lerobot version
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except Exception:
            from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

        port = input("SO101 robot USB port (default /dev/ttyACM1): ").strip() or "/dev/ttyACM1"
        robot_id = input("Robot id (default SO101_follower): ").strip() or "SO101_follower"

        # Use degrees so you don't see servo ticks
        robot_config = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
        robot = SO101Follower(robot_config)

        keyboard = KeyboardTeleop(KeyboardTeleopConfig())

        robot.connect()
        keyboard.connect()

        # Calibrate choice
        while True:
            calibrate_choice = input("Recalibrate now? (y/n): ").strip().lower()
            if calibrate_choice in ["y", "yes"]:
                robot.calibrate()
                print("[INFO] calibration done")
                break
            if calibrate_choice in ["n", "no"]:
                break
            print("Please enter y or n")

        # Start joint angles (degrees)
        start_obs = robot.get_observation()
        start_positions = {}
        for k, v in start_obs.items():
            if k.endswith(".pos"):
                name = k.removesuffix(".pos")
                start_positions[name] = float(v)

        move_to_zero_position(robot, stop_event, duration=3.0, kp=0.5, control_freq=CONTROL_FREQ)

        # YOLO init
        model = YOLOE("yoloe-11l-seg.pt")

        target_input = input("Objects to detect (comma-separated, default bottle): ").strip()
        if not target_input:
            target_objects = ["bottle"]
        else:
            target_objects = [obj.strip() for obj in target_input.split(",") if obj.strip()]
        model.set_classes(target_objects, model.get_text_pe(target_objects))

        cameras = list_cameras()
        if not cameras:
            print("No cameras found!")
            return
        print(f"Available cameras: {cameras}")
        selected = int(input(f"Select camera index from {cameras}: ").strip())
        cap = cv2.VideoCapture(selected)
        if not cap.isOpened():
            print("Camera not found!")
            return

        print("=" * 60)
        print("Keys:")
        print("  q/a: shoulder_pan -/+")
        print("  w/s: EE x -/+ (updates shoulder_lift+elbow_flex)")
        print("  e/d: EE y -/+ (updates shoulder_lift+elbow_flex)")
        print("  r/f: pitch +/-(wrist_flex)")
        print("  t/g: wrist_roll -/+")
        print("  y/h: gripper close/open")
        print("  p: toggle YOLO follow ON/OFF (default OFF)")
        print("  x: exit (returns to start)")
        print("")
        print("YOLO window:")
        print("  q or ESC: quit (also stops control)")
        print("=" * 60)

        # Start YOLO thread
        video_thread = threading.Thread(target=yolo_loop, args=(model, cap, robot, stop_event, shared), daemon=True)
        video_thread.start()

        # Control loop in main thread
        p_control_loop(robot, keyboard, stop_event, shared, start_positions)

    except Exception as e:
        print(f"Program failed: {e}")
        traceback.print_exc()
        print("Checklist:")
        print("  1) Robot connected and powered")
        print("  2) Correct USB port (/dev/ttyACM*)")
        print("  3) Permissions on the USB device")
        print("  4) lerobot version contains SO101 follower")
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
