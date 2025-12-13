#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO101: Keyboard + YOLO11x (COCO) display, optional follow with calibration and persistent settings.

Keys:
- ESC: quit (both in YOLO window and keyboard teleop)
- p: toggle follow ON/OFF (default OFF)
- c: auto-calibrate follow (hold the target visible, ideally near center)
- k: edit CONTROL_FREQ, KP, EE_STEP, PITCH_STEP
- Digits 0..9: select memory slot
- o: store current pose to selected slot
- i: go to selected slot

Manual joint controls:
- q/a: shoulder_pan -/+
- t/g: wrist_roll -/+
- y/h: gripper close/open
- r/f: pitch +/-

EE controls:
- Left/Right arrow: x -/+
- Up/Down arrow: y -/+

Notes:
- Uses LeRobot SO101 follower in degrees (use_degrees=True).
- YOLO is COCO with yolo11x.pt.
"""

import os
import json
import time
import math
import cv2
import threading
import traceback
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List

# -----------------------------
# Silence Ultralytics console logs
# -----------------------------
os.environ.setdefault("YOLO_VERBOSE", "False")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("ultralytics.utils").setLevel(logging.ERROR)

from ultralytics import YOLO  # noqa: E402


# -----------------------------
# Persistent settings file
# -----------------------------
SETTINGS_PATH = "so101_yolo_follow_settings.json"
SETTINGS_VERSION = 1


# -----------------------------
# COCO aliases (user-friendly)
# -----------------------------
COCO_ALIAS = {
    "human": "person",
    "man": "person",
    "woman": "person",
    "cellphone": "cell phone",
}


# -----------------------------
# Planar IK (same as your original)
# -----------------------------
def inverse_kinematics_2d(x, y, l1=0.1159, l2=0.1350):
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

    joint2_deg = 90 - joint2_deg
    joint3_deg = joint3_deg - 90

    return joint2_deg, joint3_deg


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def now_ts() -> float:
    return time.time()


# -----------------------------
# Settings schema
# -----------------------------
@dataclass
class FollowGains:
    pan_deg_per_norm: float = -2.0   # delta_pan_deg = gain * dx_norm
    y_m_per_norm: float = -0.01      # delta_y_m = gain * dy_norm
    calibrated: bool = False

@dataclass
class ControlParams:
    control_freq: int = 20
    kp: float = 0.3
    ee_step: float = 0.004
    pitch_step: float = 1.0

@dataclass
class YoloParams:
    model_path: str = "yolo11x.pt"
    imgsz: int = 960
    conf: float = 0.15
    iou: float = 0.7
    target_objects: List[str] = None
    camera_index: Optional[int] = None

    def __post_init__(self):
        if self.target_objects is None:
            self.target_objects = ["cup"]

@dataclass
class FollowParams:
    deadzone_x: float = 0.06
    deadzone_y: float = 0.05
    pan_max_deg_per_frame: float = 3.0
    y_max_m_per_frame: float = 0.005
    calibration_fraction: float = 0.5      # Réduit pour gains plus conservateurs
    calibration_pan_step_deg: float = 3.0  # Augmenté pour meilleure détection de changement
    calibration_y_step_m: float = 0.005    # Augmenté pour meilleure détection
    calibration_settle_frames: int = 12    # Augmenté pour plus de stabilité

@dataclass
class SavedPose:
    joints_deg: Dict[str, float]
    current_x: float
    current_y: float
    pitch: float
    timestamp: float

@dataclass
class AppSettings:
    version: int = SETTINGS_VERSION
    control: ControlParams = ControlParams()
    yolo: YoloParams = YoloParams()
    follow: FollowParams = FollowParams()
    gains: FollowGains = FollowGains()
    poses: Dict[str, SavedPose] = None

    def __post_init__(self):
        if self.poses is None:
            self.poses = {}


def _dict_to_dataclass(settings_dict: Dict[str, Any]) -> AppSettings:
    # Robust-ish loader with defaults
    s = AppSettings()

    try:
        if int(settings_dict.get("version", SETTINGS_VERSION)) != SETTINGS_VERSION:
            # If version mismatch, just try best-effort merge
            pass
    except Exception:
        pass

    # control
    c = settings_dict.get("control", {})
    s.control.control_freq = int(c.get("control_freq", s.control.control_freq))
    s.control.kp = float(c.get("kp", s.control.kp))
    s.control.ee_step = float(c.get("ee_step", s.control.ee_step))
    s.control.pitch_step = float(c.get("pitch_step", s.control.pitch_step))

    # yolo
    y = settings_dict.get("yolo", {})
    s.yolo.model_path = str(y.get("model_path", s.yolo.model_path))
    s.yolo.imgsz = int(y.get("imgsz", s.yolo.imgsz))
    s.yolo.conf = float(y.get("conf", s.yolo.conf))
    s.yolo.iou = float(y.get("iou", s.yolo.iou))
    s.yolo.target_objects = list(y.get("target_objects", s.yolo.target_objects or ["cup"]))
    s.yolo.camera_index = y.get("camera_index", s.yolo.camera_index)
    if s.yolo.camera_index is not None:
        try:
            s.yolo.camera_index = int(s.yolo.camera_index)
        except Exception:
            s.yolo.camera_index = None

    # follow params
    fp = settings_dict.get("follow", {})
    s.follow.deadzone_x = float(fp.get("deadzone_x", s.follow.deadzone_x))
    s.follow.deadzone_y = float(fp.get("deadzone_y", s.follow.deadzone_y))
    s.follow.pan_max_deg_per_frame = float(fp.get("pan_max_deg_per_frame", s.follow.pan_max_deg_per_frame))
    s.follow.y_max_m_per_frame = float(fp.get("y_max_m_per_frame", s.follow.y_max_m_per_frame))
    s.follow.calibration_fraction = float(fp.get("calibration_fraction", s.follow.calibration_fraction))
    s.follow.calibration_pan_step_deg = float(fp.get("calibration_pan_step_deg", s.follow.calibration_pan_step_deg))
    s.follow.calibration_y_step_m = float(fp.get("calibration_y_step_m", s.follow.calibration_y_step_m))
    s.follow.calibration_settle_frames = int(fp.get("calibration_settle_frames", s.follow.calibration_settle_frames))

    # gains
    g = settings_dict.get("gains", {})
    s.gains.pan_deg_per_norm = float(g.get("pan_deg_per_norm", s.gains.pan_deg_per_norm))
    s.gains.y_m_per_norm = float(g.get("y_m_per_norm", s.gains.y_m_per_norm))
    s.gains.calibrated = bool(g.get("calibrated", s.gains.calibrated))

    # poses
    poses = settings_dict.get("poses", {}) or {}
    s.poses = {}
    for k, v in poses.items():
        try:
            s.poses[str(k)] = SavedPose(
                joints_deg=dict(v.get("joints_deg", {})),
                current_x=float(v.get("current_x", 0.1629)),
                current_y=float(v.get("current_y", 0.1131)),
                pitch=float(v.get("pitch", 0.0)),
                timestamp=float(v.get("timestamp", now_ts())),
            )
        except Exception:
            continue

    return s


def load_settings(path: str) -> AppSettings:
    if not os.path.exists(path):
        return AppSettings()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _dict_to_dataclass(data if isinstance(data, dict) else {})
    except Exception:
        return AppSettings()


def save_settings(path: str, settings: AppSettings) -> None:
    try:
        # Convert dataclasses safely
        out = {
            "version": SETTINGS_VERSION,
            "control": asdict(settings.control),
            "yolo": asdict(settings.yolo),
            "follow": asdict(settings.follow),
            "gains": asdict(settings.gains),
            "poses": {k: asdict(v) for k, v in settings.poses.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    except Exception:
        # Stay silent to respect "prints only on events"
        pass


# -----------------------------
# Shared runtime state
# -----------------------------
class SharedState:
    def __init__(self, settings: AppSettings):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.settings = settings

        # follow toggle (default OFF)
        self.follow_enabled = False

        # EE state (2D)
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0

        # target joint positions (degrees)
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

        # latest detection for follow/calibration
        self.last_frame_shape: Optional[Tuple[int, int]] = None
        self.last_best_box: Optional[Tuple[float, float, float, float]] = None
        self.last_best_conf: Optional[float] = None

        # pose slot selection
        self.selected_slot: str = "1"

    def sync_from_saved_pose(self, pose: SavedPose):
        self.current_x = float(pose.current_x)
        self.current_y = float(pose.current_y)
        self.pitch = float(pose.pitch)
        # copy targets for main joints; wrist_flex will be recomputed from pitch
        for k, v in pose.joints_deg.items():
            if k in self.target_positions:
                self.target_positions[k] = float(v)

    def snapshot_pose(self, joints_deg: Dict[str, float]) -> SavedPose:
        return SavedPose(
            joints_deg=dict(joints_deg),
            current_x=float(self.current_x),
            current_y=float(self.current_y),
            pitch=float(self.pitch),
            timestamp=now_ts(),
        )


# -----------------------------
# YOLO helpers
# -----------------------------
def normalize_target_names(names: List[str]) -> List[str]:
    out = []
    for n in names:
        t = str(n).strip().lower()
        if not t:
            continue
        out.append(COCO_ALIAS.get(t, t))
    return out


def get_class_ids_from_names(model: YOLO, names: List[str]) -> List[int]:
    # model.names: dict id->name
    name_to_id = {str(v).lower(): int(k) for k, v in model.names.items()}
    ids = []
    for n in names:
        if n in name_to_id:
            ids.append(name_to_id[n])
    return ids


def pick_best_box(result0) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[float]]:
    boxes = result0.boxes
    if boxes is None or len(boxes) == 0:
        return None, None
    confs = boxes.conf
    xyxy = boxes.xyxy
    idx = int(confs.argmax().item())
    box = xyxy[idx].tolist()
    conf = float(confs[idx].item())
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3])), conf


def box_center_norm(box: Tuple[float, float, float, float], frame_shape) -> Tuple[float, float]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    dx_norm = (cx - (w * 0.5)) / (w * 0.5)
    dy_norm = (cy - (h * 0.5)) / (h * 0.5)
    return dx_norm, dy_norm


# -----------------------------
# Follow calibration (key 'c') - Multi-step robust calibration
# -----------------------------
def wait_for_new_boxes(shared: SharedState, n: int, timeout_s: float = 2.5) -> bool:
    got = 0
    last = None
    t0 = now_ts()
    while got < n and (now_ts() - t0) < timeout_s and not shared.stop_event.is_set():
        with shared.lock:
            b = shared.last_best_box
        if b is not None and b != last:
            last = b
            got += 1
        time.sleep(0.02)
    return got >= n


def run_follow_calibration(robot, shared: SharedState) -> None:
    # Event function: prints are allowed here
    with shared.lock:
        box0 = shared.last_best_box
        shape0 = shared.last_frame_shape
        pan_step_deg = float(shared.settings.follow.calibration_pan_step_deg)
        y_step_m = float(shared.settings.follow.calibration_y_step_m)
        settle_frames = int(shared.settings.follow.calibration_settle_frames)
        frac = float(shared.settings.follow.calibration_fraction)

    if box0 is None or shape0 is None:
        print("[CAL] ❌ Pas de détection disponible.")
        print("[CAL] → Place l'objet bien visible au centre du champ et relance 'c'")
        return

    base_dx, base_dy = box_center_norm(box0, (shape0[0], shape0[1], 3))
    print(f"[CAL] ✓ Position de base détectée: dx={base_dx:+.3f}, dy={base_dy:+.3f}")
    print()

    # ============================================================
    # ÉTAPE 1: Calibration PAN (rotation horizontale)
    # ============================================================
    print("[CAL] ÉTAPE 1 : Calibration PAN (rotation)")
    print(f"[CAL]   Direction: +{pan_step_deg:+.1f}°")
    pan_deltas = []
    
    for step_num in range(3):
        print(f"[CAL]   Mesure {step_num + 1}/3...")
        
        # Move pan positive
        with shared.lock:
            shared.target_positions["shoulder_pan"] = float(
                shared.target_positions["shoulder_pan"] + pan_step_deg
            )
        
        if not wait_for_new_boxes(shared, settle_frames, timeout_s=3.0):
            print(f"[CAL]   ⚠ Détection perdue après mouvement pan")
            with shared.lock:
                shared.target_positions["shoulder_pan"] = float(
                    shared.target_positions["shoulder_pan"] - pan_step_deg
                )
            continue

        with shared.lock:
            b_pan = shared.last_best_box
            s_pan = shared.last_frame_shape

        if b_pan is None or s_pan is None:
            print(f"[CAL]   ⚠ Détection invalide après pan")
            with shared.lock:
                shared.target_positions["shoulder_pan"] = float(
                    shared.target_positions["shoulder_pan"] - pan_step_deg
                )
            continue

        dx_pan, dy_pan = box_center_norm(b_pan, (s_pan[0], s_pan[1], 3))
        delta_dx = dx_pan - base_dx
        pan_deltas.append(delta_dx)
        print(f"[CAL]   ✓ delta_dx = {delta_dx:+.4f}")

        # Revert pan
        with shared.lock:
            shared.target_positions["shoulder_pan"] = float(
                shared.target_positions["shoulder_pan"] - pan_step_deg
            )
        time.sleep(0.15)

    if not pan_deltas:
        print("[CAL] ❌ Impossible d'effectuer la calibration PAN")
        return

    avg_delta_dx = sum(pan_deltas) / len(pan_deltas)
    print(f"[CAL] → Moyenne: delta_dx = {avg_delta_dx:+.4f}")
    print()

    # ============================================================
    # ÉTAPE 2: Calibration Y (mouvement vertical)
    # ============================================================
    print("[CAL] ÉTAPE 2 : Calibration Y (mouvement vertical)")
    print(f"[CAL]   Direction: +{y_step_m:+.4f}m")
    y_deltas = []

    for step_num in range(3):
        print(f"[CAL]   Mesure {step_num + 1}/3...")
        
        # Move Y positive
        with shared.lock:
            shared.current_y = float(shared.current_y + y_step_m)
            j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
            shared.target_positions["shoulder_lift"] = float(j2)
            shared.target_positions["elbow_flex"] = float(j3)

        if not wait_for_new_boxes(shared, settle_frames, timeout_s=3.0):
            print(f"[CAL]   ⚠ Détection perdue après mouvement y")
            with shared.lock:
                shared.current_y = float(shared.current_y - y_step_m)
                j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                shared.target_positions["shoulder_lift"] = float(j2)
                shared.target_positions["elbow_flex"] = float(j3)
            continue

        with shared.lock:
            b_y = shared.last_best_box
            s_y = shared.last_frame_shape

        if b_y is None or s_y is None:
            print(f"[CAL]   ⚠ Détection invalide après y")
            with shared.lock:
                shared.current_y = float(shared.current_y - y_step_m)
                j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                shared.target_positions["shoulder_lift"] = float(j2)
                shared.target_positions["elbow_flex"] = float(j3)
            continue

        dx_y, dy_y = box_center_norm(b_y, (s_y[0], s_y[1], 3))
        delta_dy = dy_y - base_dy
        y_deltas.append(delta_dy)
        print(f"[CAL]   ✓ delta_dy = {delta_dy:+.4f}")

        # Revert Y
        with shared.lock:
            shared.current_y = float(shared.current_y - y_step_m)
            j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
            shared.target_positions["shoulder_lift"] = float(j2)
            shared.target_positions["elbow_flex"] = float(j3)
        time.sleep(0.15)

    if not y_deltas:
        print("[CAL] ❌ Impossible d'effectuer la calibration Y")
        return

    avg_delta_dy = sum(y_deltas) / len(y_deltas)
    print(f"[CAL] → Moyenne: delta_dy = {avg_delta_dy:+.4f}")
    print()

    # ============================================================
    # ÉTAPE 3: Calcul des gains (avec facteur de fraction)
    # ============================================================
    print("[CAL] ÉTAPE 3 : Calcul des gains de suivi")
    
    eps = 1e-6
    ok_pan = abs(avg_delta_dx) > eps
    ok_y = abs(avg_delta_dy) > eps

    pan_gain = None
    y_gain = None

    if ok_pan:
        pan_gain = -(frac * pan_step_deg / avg_delta_dx)
        print(f"[CAL]   ✓ Gain PAN: {pan_gain:+.4f} deg/norm")
    else:
        print(f"[CAL]   ❌ Détection PAN invalide (delta trop petit)")

    if ok_y:
        y_gain = -(frac * y_step_m / avg_delta_dy)
        print(f"[CAL]   ✓ Gain Y:   {y_gain:+.6f} m/norm")
    else:
        print(f"[CAL]   ❌ Détection Y invalide (delta trop petit)")

    print()

    # ============================================================
    # ÉTAPE 4: Validation et sauvegarde
    # ============================================================
    if ok_pan and ok_y:
        print("[CAL] ✅ Calibration RÉUSSIE !")
        with shared.lock:
            shared.settings.gains.pan_deg_per_norm = pan_gain
            shared.settings.gains.y_m_per_norm = y_gain
            shared.settings.gains.calibrated = True
            gp = shared.settings.gains.pan_deg_per_norm
            gy = shared.settings.gains.y_m_per_norm

        save_settings(SETTINGS_PATH, shared.settings)
        print(f"[CAL] Gains sauvegardés: pan={gp:+.4f}, y={gy:+.6f}")
        print("[CAL] ========== CALIBRATION TERMINÉE (OK) ==========")
    else:
        print("[CAL] ❌ Calibration ÉCHOUÉE")
        print("[CAL] → Vérifications:")
        print(f"[CAL]   • Objet bien centré et visible?")
        print(f"[CAL]   • Caméra bien allumée et branchée?")
        print(f"[CAL]   • Détection assez stable (conf > 0.15)?")
        print("[CAL] ========== CALIBRATION TERMINÉE (ÉCHEC) ==========")


# -----------------------------
# Pose storing and goto
# -----------------------------
def read_joint_positions_deg(robot) -> Dict[str, float]:
    obs = robot.get_observation()
    joints = {}
    for k, v in obs.items():
        if k.endswith(".pos"):
            j = k.removesuffix(".pos")
            joints[j] = float(v)
    return joints


def goto_pose_blocking(robot, shared: SharedState, target: SavedPose, timeout_s: float = 6.0, threshold_deg: float = 2.0) -> None:
    # Event function: prints are allowed here
    # Temporarily disable follow during goto
    with shared.lock:
        follow_prev = shared.follow_enabled
        shared.follow_enabled = False
        shared.sync_from_saved_pose(target)

    t0 = now_ts()
    while (now_ts() - t0) < timeout_s and not shared.stop_event.is_set():
        with shared.lock:
            kp = float(shared.settings.control.kp)
            cf = int(shared.settings.control.control_freq)
            pitch = float(shared.pitch)
            # enforce wrist_flex rule
            shared.target_positions["wrist_flex"] = (
                -float(shared.target_positions["shoulder_lift"])
                -float(shared.target_positions["elbow_flex"])
                + pitch
            )
            targets = dict(shared.target_positions)

        obs = robot.get_observation()
        cur = {}
        for k, v in obs.items():
            if k.endswith(".pos"):
                j = k.removesuffix(".pos")
                cur[j] = float(v)

        action = {}
        total_err = 0.0
        for j, tgt in targets.items():
            if j in cur:
                err = float(tgt) - float(cur[j])
                total_err += abs(err)
                action[f"{j}.pos"] = float(cur[j] + kp * err)

        if action:
            robot.send_action(action)

        if total_err < threshold_deg:
            break

        time.sleep(1.0 / max(1, cf))

    with shared.lock:
        shared.follow_enabled = follow_prev


def store_pose(shared: SharedState, slot: str, joints_deg: Dict[str, float]) -> None:
    pose = shared.snapshot_pose(joints_deg)
    shared.settings.poses[str(slot)] = pose
    save_settings(SETTINGS_PATH, shared.settings)


# -----------------------------
# Manual tuning menu (key 'k')
# -----------------------------
def tune_params_interactive(shared: SharedState) -> None:
    # Event function: prints are allowed here
    with shared.lock:
        c = shared.settings.control

    print("[TUNE] paramètres actuels:")
    print(f"  CONTROL_FREQ = {c.control_freq}")
    print(f"  KP = {c.kp}")
    print(f"  EE_STEP = {c.ee_step}")
    print(f"  PITCH_STEP = {c.pitch_step}")
    print("[TUNE] entre une valeur ou laisse vide pour garder")

    def ask_int(name: str, cur: int) -> int:
        s = input(f"{name} [{cur}]: ").strip()
        if not s:
            return cur
        return int(s)

    def ask_float(name: str, cur: float) -> float:
        s = input(f"{name} [{cur}]: ").strip()
        if not s:
            return cur
        return float(s)

    try:
        new_freq = ask_int("CONTROL_FREQ", c.control_freq)
        new_kp = ask_float("KP", c.kp)
        new_ee = ask_float("EE_STEP", c.ee_step)
        new_pitch = ask_float("PITCH_STEP", c.pitch_step)
    except Exception:
        print("[TUNE] entrée invalide, aucun changement")
        return

    with shared.lock:
        shared.settings.control.control_freq = max(1, int(new_freq))
        shared.settings.control.kp = float(new_kp)
        shared.settings.control.ee_step = float(new_ee)
        shared.settings.control.pitch_step = float(new_pitch)

    save_settings(SETTINGS_PATH, shared.settings)
    print("[TUNE] ok, enregistré")


# -----------------------------
# YOLO thread
# -----------------------------
def yolo_loop(model: YOLO, cap, class_ids: List[int], shared: SharedState):
    last_follow_print_ts = 0.0

    while not shared.stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with shared.lock:
                yp = shared.settings.yolo
                fp = shared.settings.follow
                gains = shared.settings.gains
                follow_on = shared.follow_enabled

            results = model.predict(
                source=frame,
                imgsz=int(yp.imgsz),
                conf=float(yp.conf),
                iou=float(yp.iou),
                classes=class_ids if class_ids else None,
                verbose=False,
            )

            annotated = frame
            best_box = None
            best_conf = None

            if results and len(results) > 0:
                r0 = results[0]
                # annotated plot (no prints)
                try:
                    annotated = r0.plot()
                except Exception:
                    annotated = frame
                best_box, best_conf = pick_best_box(r0)

            with shared.lock:
                shared.last_frame_shape = frame.shape[:2]
                shared.last_best_box = best_box
                shared.last_best_conf = best_conf

            # Follow control: center object in frame
            if follow_on and best_box is not None:
                dx_norm, dy_norm = box_center_norm(best_box, frame.shape)

                if abs(dx_norm) < float(fp.deadzone_x):
                    dx_norm = 0.0
                if abs(dy_norm) < float(fp.deadzone_y):
                    dy_norm = 0.0

                delta_pan_deg = float(gains.pan_deg_per_norm) * dx_norm
                delta_y_m = float(gains.y_m_per_norm) * dy_norm

                delta_pan_deg = clamp(delta_pan_deg, -float(fp.pan_max_deg_per_frame), float(fp.pan_max_deg_per_frame))
                delta_y_m = clamp(delta_y_m, -float(fp.y_max_m_per_frame), float(fp.y_max_m_per_frame))

                did_move = (abs(delta_pan_deg) > 1e-4) or (abs(delta_y_m) > 1e-6)

                if did_move:
                    with shared.lock:
                        shared.target_positions["shoulder_pan"] = float(shared.target_positions["shoulder_pan"] + delta_pan_deg)

                        shared.current_y = float(shared.current_y + delta_y_m)
                        j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                        shared.target_positions["shoulder_lift"] = float(j2)
                        shared.target_positions["elbow_flex"] = float(j3)

                    # Print only when movement happens
                    t = now_ts()
                    if t - last_follow_print_ts > 0.12:
                        last_follow_print_ts = t
                        print(f"[FOLLOW] pan {delta_pan_deg:+.2f}deg, y {delta_y_m:+.4f}m, dx={dx_norm:+.2f}, dy={dy_norm:+.2f}")

            # Show window, ESC only to quit
            cv2.imshow("YOLO Live Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                shared.stop_event.set()
                break

        except Exception:
            shared.stop_event.set()
            break

    cv2.destroyAllWindows()


# -----------------------------
# Keyboard mapping helpers
# -----------------------------
ESC_KEYS = {"esc", "escape", "ESC", "Key.esc", "Key.ESC"}
LEFT_KEYS = {"s", "left", "LEFT", "Key.left", "KEY_LEFT", "arrow_left"}
RIGHT_KEYS = {"w", "right", "RIGHT", "Key.right", "KEY_RIGHT", "arrow_right"}
UP_KEYS = {"q", "up", "UP", "Key.up", "KEY_UP", "arrow_up"}
DOWN_KEYS = {"e", "down", "DOWN", "Key.down", "KEY_DOWN", "arrow_down"}


def is_arrow(key: str, keyset: set) -> bool:
    return str(key) in keyset


# -----------------------------
# Main P-control loop
# -----------------------------
def p_control_loop(robot, keyboard, shared: SharedState, start_positions: Dict[str, float]):
    joint_controls = {
        "a": ("shoulder_pan", -1.0),
        "d": ("shoulder_pan", 1.0),
        "t": ("wrist_roll", -1.0),
        "g": ("wrist_roll", 1.0),
        "y": ("gripper", -1.0),
        "h": ("gripper", 1.0),
    }

    while not shared.stop_event.is_set():
        try:
            keyboard_action = keyboard.get_action()

            # Handle keypress events only
            if keyboard_action:
                for key in keyboard_action.keys():
                    k = str(key)

                    # ESC to quit
                    if k in ESC_KEYS:
                        shared.stop_event.set()
                        print("[KEY] ESC, exit")
                        break

                    # Toggle follow
                    if k == "p":
                        with shared.lock:
                            shared.follow_enabled = not shared.follow_enabled
                            on = shared.follow_enabled
                        print(f"[KEY] follow {'ON' if on else 'OFF'}")
                        continue

                    # Auto-calibrate follow
                    if k == "c":
                        run_follow_calibration(robot, shared)
                        continue

                    # Edit params
                    if k == "k":
                        tune_params_interactive(shared)
                        continue

                    # Select slot 0..9
                    if len(k) == 1 and k.isdigit():
                        with shared.lock:
                            shared.selected_slot = k
                        print(f"[KEY] slot = {k}")
                        continue

                    # Store pose
                    if k == "o":
                        with shared.lock:
                            slot = shared.selected_slot
                        joints = read_joint_positions_deg(robot)
                        with shared.lock:
                            store_pose(shared, slot, joints)
                        print(f"[POSE] saved slot {slot}")
                        continue

                    # Go to pose
                    if k == "i":
                        with shared.lock:
                            slot = shared.selected_slot
                            pose = shared.settings.poses.get(slot)
                        if pose is None:
                            print(f"[POSE] slot {slot} vide")
                        else:
                            print(f"[POSE] goto slot {slot}")
                            goto_pose_blocking(robot, shared, pose)
                            print(f"[POSE] done slot {slot}")
                        continue

                    # Pitch adjust
                    if k == "r":
                        with shared.lock:
                            shared.pitch += float(shared.settings.control.pitch_step)
                            p = shared.pitch
                        print(f"[KEY] pitch {p:+.1f}deg")
                        continue

                    if k == "f":
                        with shared.lock:
                            shared.pitch -= float(shared.settings.control.pitch_step)
                            p = shared.pitch
                        print(f"[KEY] pitch {p:+.1f}deg")
                        continue

                    # Joint direct controls
                    if k in joint_controls:
                        jname, delta = joint_controls[k]
                        with shared.lock:
                            cur = float(shared.target_positions.get(jname, 0.0))
                            new = cur + float(delta)
                            shared.target_positions[jname] = new
                        print(f"[KEY] {jname} {cur:.2f} -> {new:.2f}deg")
                        continue

                    # EE arrows: Left/Right for x, Up/Down for y
                    moved_ee = False
                    with shared.lock:
                        step = float(shared.settings.control.ee_step)

                    if is_arrow(k, LEFT_KEYS):
                        with shared.lock:
                            shared.current_x -= step
                            j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                            shared.target_positions["shoulder_lift"] = float(j2)
                            shared.target_positions["elbow_flex"] = float(j3)
                            cx, cy = shared.current_x, shared.current_y
                        moved_ee = True

                    elif is_arrow(k, RIGHT_KEYS):
                        with shared.lock:
                            shared.current_x += step
                            j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                            shared.target_positions["shoulder_lift"] = float(j2)
                            shared.target_positions["elbow_flex"] = float(j3)
                            cx, cy = shared.current_x, shared.current_y
                        moved_ee = True

                    elif is_arrow(k, UP_KEYS):
                        with shared.lock:
                            shared.current_y -= step
                            j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                            shared.target_positions["shoulder_lift"] = float(j2)
                            shared.target_positions["elbow_flex"] = float(j3)
                            cx, cy = shared.current_x, shared.current_y
                        moved_ee = True

                    elif is_arrow(k, DOWN_KEYS):
                        with shared.lock:
                            shared.current_y += step
                            j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                            shared.target_positions["shoulder_lift"] = float(j2)
                            shared.target_positions["elbow_flex"] = float(j3)
                            cx, cy = shared.current_x, shared.current_y
                        moved_ee = True

                    if moved_ee:
                        print(f"[KEY] EE x={cx:.4f} y={cy:.4f} -> shoulder_lift={j2:.2f} elbow_flex={j3:.2f}")
                        continue

            # P-control tick
            with shared.lock:
                cf = int(shared.settings.control.control_freq)
                kp = float(shared.settings.control.kp)
                pitch = float(shared.pitch)

                # wrist_flex rule
                shared.target_positions["wrist_flex"] = (
                    -float(shared.target_positions["shoulder_lift"])
                    -float(shared.target_positions["elbow_flex"])
                    + pitch
                )
                targets = dict(shared.target_positions)

            obs = robot.get_observation()
            cur = {}
            for kk, vv in obs.items():
                if kk.endswith(".pos"):
                    j = kk.removesuffix(".pos")
                    cur[j] = float(vv)

            action = {}
            for j, tgt in targets.items():
                if j in cur:
                    err = float(tgt) - float(cur[j])
                    action[f"{j}.pos"] = float(cur[j] + kp * err)

            if action:
                robot.send_action(action)

            time.sleep(1.0 / max(1, cf))

        except KeyboardInterrupt:
            shared.stop_event.set()
            print("[KEY] KeyboardInterrupt, exit")
            break
        except Exception:
            shared.stop_event.set()
            break

    # Return to start (quiet)
    try:
        # quick return using same kp/freq
        t0 = now_ts()
        while (now_ts() - t0) < 5.0 and start_positions and not shared.stop_event.is_set():
            with shared.lock:
                cf = int(shared.settings.control.control_freq)
                kp = float(shared.settings.control.kp)

            obs = robot.get_observation()
            cur = {}
            for kk, vv in obs.items():
                if kk.endswith(".pos"):
                    j = kk.removesuffix(".pos")
                    cur[j] = float(vv)

            total_err = 0.0
            action = {}
            for j, target in start_positions.items():
                if j in cur:
                    err = float(target) - float(cur[j])
                    total_err += abs(err)
                    action[f"{j}.pos"] = float(cur[j] + 0.2 * err)

            if action:
                robot.send_action(action)

            if total_err < 2.0:
                break

            time.sleep(1.0 / max(1, cf))
    except Exception:
        pass


# -----------------------------
# Utility: camera probing (no prints)
# -----------------------------
def list_cameras(max_index=8):
    available = []
    for idx in range(max_index):
        cap_test = cv2.VideoCapture(idx)
        if cap_test.isOpened():
            available.append(idx)
            cap_test.release()
    return available


# -----------------------------
# Main
# -----------------------------
def main():
    settings = load_settings(SETTINGS_PATH)
    shared = SharedState(settings)

    robot = None
    keyboard = None
    cap = None

    try:
        # LeRobot imports depending on version
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except Exception:
            from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

        port = input("SO101 robot USB port (default /dev/ttyACM1): ").strip() or "/dev/ttyACM1"
        robot_id = input("Robot id (default SO101_follower): ").strip() or "SO101_follower"

        robot_config = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
        robot = SO101Follower(robot_config)

        keyboard = KeyboardTeleop(KeyboardTeleopConfig())

        robot.connect()
        keyboard.connect()

        # Read start positions
        start_positions = read_joint_positions_deg(robot)

        # YOLO model
        model = YOLO(settings.yolo.model_path)

        # Determine target classes from settings
        target_objects = normalize_target_names(settings.yolo.target_objects or ["cup"])
        class_ids = get_class_ids_from_names(model, target_objects)

        # Camera selection from settings or prompt once
        cam_idx = settings.yolo.camera_index
        if cam_idx is None:
            cams = list_cameras()
            if not cams:
                raise RuntimeError("No camera found")
            cam_idx = int(input(f"Select camera index {cams}: ").strip())
            settings.yolo.camera_index = cam_idx
            save_settings(SETTINGS_PATH, settings)

        cap = cv2.VideoCapture(int(cam_idx))
        if not cap.isOpened():
            raise RuntimeError("Camera open failed")

        # Start YOLO thread
        yolo_thread = threading.Thread(target=yolo_loop, args=(model, cap, class_ids, shared), daemon=True)
        yolo_thread.start()

        # Run control loop
        p_control_loop(robot, keyboard, shared, start_positions)

    except Exception:
        # Keep this quiet-ish (no stacktrace spam)
        print("Erreur: le programme s'est arrêté. (port/cam/poids YOLO/import lerobot)")
    finally:
        # Save settings on exit
        save_settings(SETTINGS_PATH, shared.settings)

        try:
            shared.stop_event.set()
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


if __name__ == "__main__":
    main()
