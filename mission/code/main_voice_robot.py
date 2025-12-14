#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice-Controlled Robot Main Script

This script combines:
- SO101 robot arm control
- YOLO object tracking
- Voice-controlled commands via OpenAI

Architecture:
- Main thread: Robot P-control loop (sends actions to hardware)
- YOLO thread: Camera capture and object detection
- Voice thread: Audio recording, STT, LLM, TTS

All threads communicate via the shared RobotController which
provides thread-safe access to robot state.

Usage:
    python main_voice_robot.py

Environment variables (optional, see .env):
    OPENAI_API_KEY - Required for voice control
    VAD_START_DB - Voice activity detection start threshold
    VAD_END_DB - Voice activity detection end threshold
    LLM_MODEL - OpenAI model for language understanding
    STT_MODEL - OpenAI model for speech-to-text
    TTS_MODEL - OpenAI model for text-to-speech
    TTS_VOICE - Voice for TTS
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from dotenv import load_dotenv

# Silence Ultralytics console logs
os.environ.setdefault("YOLO_VERBOSE", "False")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("ultralytics.utils").setLevel(logging.ERROR)

from ultralytics import YOLO

# Local imports
from voice_robot.robot_controller import RobotController, inverse_kinematics_2d
from voice_robot.voice_agent import VoiceRobotAgent, VoiceLoopRunner


# -----------------------------
# Configuration
# -----------------------------

SETTINGS_PATH = "voice_robot_settings.json"
SETTINGS_VERSION = 1

# Pretrained policy path
PRETRAINED_POLICY_PATH = "crc-amd-hackathon-2025/pi05-grab-cam"
SINGLE_TASK = "Grab the camera"

# COCO class aliases
COCO_ALIAS = {
    "human": "person",
    "man": "person",
    "woman": "person",
    "cellphone": "cell phone",
}


# -----------------------------
# Settings Classes
# -----------------------------

@dataclass
class FollowGains:
    pan_deg_per_norm: float = 3.0
    y_m_per_norm: float = -0.008
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
    deadzone_x: float = 0.08
    deadzone_y: float = 0.08
    pan_max_deg_per_frame: float = 2.5
    y_max_m_per_frame: float = 0.004


@dataclass
class VoiceParams:
    sample_rate: int = 16000
    vad_start_db: float = -19.0
    vad_end_db: float = -24.0
    vad_frame_ms: int = 20
    vad_preroll_ms: int = 250
    vad_end_silence_ms: int = 600
    vad_max_record_s: float = 15.0
    vad_required_start_frames: int = 3
    llm_model: str = "gpt-4.1"
    stt_model: str = "gpt-4o-transcribe"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "coral"


@dataclass
class AppSettings:
    version: int = SETTINGS_VERSION
    control: ControlParams = None
    yolo: YoloParams = None
    follow: FollowParams = None
    gains: FollowGains = None
    voice: VoiceParams = None
    
    def __post_init__(self):
        if self.control is None:
            self.control = ControlParams()
        if self.yolo is None:
            self.yolo = YoloParams()
        if self.follow is None:
            self.follow = FollowParams()
        if self.gains is None:
            self.gains = FollowGains()
        if self.voice is None:
            self.voice = VoiceParams()


def load_settings(path: str) -> AppSettings:
    """Load settings from JSON file."""
    if not os.path.exists(path):
        return AppSettings()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        s = AppSettings()
        
        # Control params
        c = data.get("control", {})
        s.control.control_freq = int(c.get("control_freq", s.control.control_freq))
        s.control.kp = float(c.get("kp", s.control.kp))
        s.control.ee_step = float(c.get("ee_step", s.control.ee_step))
        s.control.pitch_step = float(c.get("pitch_step", s.control.pitch_step))
        
        # YOLO params
        y = data.get("yolo", {})
        s.yolo.model_path = str(y.get("model_path", s.yolo.model_path))
        s.yolo.imgsz = int(y.get("imgsz", s.yolo.imgsz))
        s.yolo.conf = float(y.get("conf", s.yolo.conf))
        s.yolo.iou = float(y.get("iou", s.yolo.iou))
        s.yolo.target_objects = list(y.get("target_objects", s.yolo.target_objects or ["cup"]))
        s.yolo.camera_index = y.get("camera_index")
        
        # Follow params
        fp = data.get("follow", {})
        s.follow.deadzone_x = float(fp.get("deadzone_x", s.follow.deadzone_x))
        s.follow.deadzone_y = float(fp.get("deadzone_y", s.follow.deadzone_y))
        s.follow.pan_max_deg_per_frame = float(fp.get("pan_max_deg_per_frame", s.follow.pan_max_deg_per_frame))
        s.follow.y_max_m_per_frame = float(fp.get("y_max_m_per_frame", s.follow.y_max_m_per_frame))
        
        # Gains
        g = data.get("gains", {})
        s.gains.pan_deg_per_norm = float(g.get("pan_deg_per_norm", s.gains.pan_deg_per_norm))
        s.gains.y_m_per_norm = float(g.get("y_m_per_norm", s.gains.y_m_per_norm))
        s.gains.calibrated = bool(g.get("calibrated", s.gains.calibrated))
        
        # Voice params
        v = data.get("voice", {})
        s.voice.sample_rate = int(v.get("sample_rate", s.voice.sample_rate))
        s.voice.vad_start_db = float(v.get("vad_start_db", s.voice.vad_start_db))
        s.voice.vad_end_db = float(v.get("vad_end_db", s.voice.vad_end_db))
        s.voice.vad_frame_ms = int(v.get("vad_frame_ms", s.voice.vad_frame_ms))
        s.voice.vad_preroll_ms = int(v.get("vad_preroll_ms", s.voice.vad_preroll_ms))
        s.voice.vad_end_silence_ms = int(v.get("vad_end_silence_ms", s.voice.vad_end_silence_ms))
        s.voice.vad_max_record_s = float(v.get("vad_max_record_s", s.voice.vad_max_record_s))
        s.voice.vad_required_start_frames = int(v.get("vad_required_start_frames", s.voice.vad_required_start_frames))
        s.voice.llm_model = str(v.get("llm_model", s.voice.llm_model))
        s.voice.stt_model = str(v.get("stt_model", s.voice.stt_model))
        s.voice.tts_model = str(v.get("tts_model", s.voice.tts_model))
        s.voice.tts_voice = str(v.get("tts_voice", s.voice.tts_voice))
        
        return s
    except Exception:
        return AppSettings()


def save_settings(path: str, settings: AppSettings) -> None:
    """Save settings to JSON file."""
    try:
        out = {
            "version": SETTINGS_VERSION,
            "control": asdict(settings.control),
            "yolo": asdict(settings.yolo),
            "follow": asdict(settings.follow),
            "gains": asdict(settings.gains),
            "voice": asdict(settings.voice),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# -----------------------------
# Shared State (for YOLO and control threads)
# -----------------------------

class SharedState:
    """Thread-safe shared state for the application."""
    
    def __init__(self, settings: AppSettings):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.settings = settings
        
        # Follow toggle
        self.follow_enabled = False
        self.last_p_press_time: float = 0.0
        self.p_debounce_delay: float = 0.3
        
        # EE state (2D)
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Target joint positions (degrees)
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        
        # YOLO detection state
        self.last_frame_shape: Optional[Tuple[int, int]] = None
        self.last_best_box: Optional[Tuple[float, float, float, float]] = None
        self.last_best_conf: Optional[float] = None


# -----------------------------
# YOLO Helpers
# -----------------------------

def normalize_target_names(names: List[str]) -> List[str]:
    """Normalize COCO object names."""
    out = []
    for n in names:
        t = str(n).strip().lower()
        if not t:
            continue
        out.append(COCO_ALIAS.get(t, t))
    return out


def get_class_ids_from_names(model: YOLO, names: List[str]) -> List[int]:
    """Get YOLO class IDs from object names."""
    name_to_id = {str(v).lower(): int(k) for k, v in model.names.items()}
    ids = []
    for n in names:
        if n in name_to_id:
            ids.append(name_to_id[n])
    return ids


def pick_best_box(result0) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[float]]:
    """Pick the best detection box from YOLO results."""
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
    """Get normalized center position of a box."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    dx_norm = (cx - (w * 0.5)) / (w * 0.5)
    dy_norm = (cy - (h * 0.5)) / (h * 0.5)
    return dx_norm, dy_norm


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# -----------------------------
# YOLO Thread
# -----------------------------

def yolo_loop(model: YOLO, cap, shared: SharedState, controller: RobotController):
    """YOLO detection and tracking loop."""
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
                target_objects = list(yp.target_objects)
            
            # Get class IDs for current target objects
            normalized_targets = normalize_target_names(target_objects)
            class_ids = get_class_ids_from_names(model, normalized_targets)
            
            results = model.predict(
                source=frame,
                imgsz=int(yp.imgsz),
                conf=float(yp.conf),
                iou=float(yp.iou),
                classes=class_ids if class_ids else None,
                verbose=False,
            )
            
            annotated = frame.copy()
            best_box = None
            best_conf = None
            
            if results and len(results) > 0:
                r0 = results[0]
                try:
                    annotated = r0.plot()
                except Exception:
                    annotated = frame.copy()
                best_box, best_conf = pick_best_box(r0)
            
            with shared.lock:
                shared.last_frame_shape = frame.shape[:2]
                shared.last_best_box = best_box
                shared.last_best_conf = best_conf
            
            # Draw visual indicators
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Draw center cross
            cross_size = 20
            cross_color = (0, 255, 0) if follow_on else (128, 128, 128)
            cv2.line(annotated, (center_x - cross_size, center_y), (center_x + cross_size, center_y), cross_color, 2)
            cv2.line(annotated, (center_x, center_y - cross_size), (center_x, center_y + cross_size), cross_color, 2)
            
            # Draw deadzone
            if follow_on:
                dz_x = int(float(fp.deadzone_x) * w / 2)
                dz_y = int(float(fp.deadzone_y) * h / 2)
                cv2.rectangle(annotated, (center_x - dz_x, center_y - dz_y), (center_x + dz_x, center_y + dz_y), (0, 255, 255), 1)
            
            # Status text
            status_text = "FOLLOW: ON" if follow_on else "FOLLOW: OFF"
            status_color = (0, 255, 0) if follow_on else (0, 0, 255)
            cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Target objects text
            target_text = f"Target: {', '.join(target_objects)}"
            cv2.putText(annotated, target_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Follow control
            if follow_on and best_box is not None:
                dx_norm, dy_norm = box_center_norm(best_box, frame.shape)
                
                # Draw object center
                obj_cx = int((best_box[0] + best_box[2]) / 2)
                obj_cy = int((best_box[1] + best_box[3]) / 2)
                cv2.circle(annotated, (obj_cx, obj_cy), 8, (255, 0, 255), -1)
                
                # Apply deadzone
                if abs(dx_norm) < float(fp.deadzone_x):
                    dx_norm = 0.0
                if abs(dy_norm) < float(fp.deadzone_y):
                    dy_norm = 0.0
                
                # Calculate movement
                delta_pan_deg = float(gains.pan_deg_per_norm) * dx_norm
                delta_y_m = float(gains.y_m_per_norm) * dy_norm
                
                delta_pan_deg = clamp(delta_pan_deg, -float(fp.pan_max_deg_per_frame), float(fp.pan_max_deg_per_frame))
                delta_y_m = clamp(delta_y_m, -float(fp.y_max_m_per_frame), float(fp.y_max_m_per_frame))
                
                did_move = (abs(delta_pan_deg) > 1e-4) or (abs(delta_y_m) > 1e-6)
                
                if did_move:
                    # Draw arrow
                    if abs(dx_norm) >= float(fp.deadzone_x) or abs(dy_norm) >= float(fp.deadzone_y):
                        cv2.arrowedLine(annotated, (center_x, center_y), (obj_cx, obj_cy), (0, 165, 255), 2, tipLength=0.1)
                    
                    with shared.lock:
                        shared.target_positions["shoulder_pan"] = float(shared.target_positions["shoulder_pan"] - 0.15 * delta_pan_deg)
                        shared.current_y = float(shared.current_y + delta_y_m)
                        j2, j3 = inverse_kinematics_2d(shared.current_x, shared.current_y)
                        shared.target_positions["shoulder_lift"] = float(j2)
                        shared.target_positions["elbow_flex"] = float(j3)
                    
                    t = time.time()
                    if t - last_follow_print_ts > 0.12:
                        last_follow_print_ts = t
                        print(f"[FOLLOW] pan {delta_pan_deg:+.2f}deg, y {delta_y_m:+.4f}m")
                
                # Show error
                error_text = f"dx:{dx_norm:+.2f} dy:{dy_norm:+.2f}"
                cv2.putText(annotated, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            elif follow_on and best_box is None:
                cv2.putText(annotated, "NO TARGET", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show window
            cv2.imshow("Voice Robot - YOLO", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                shared.stop_event.set()
                break
        
        except Exception as e:
            print(f"[YOLO] Error: {e}")
            shared.stop_event.set()
            break
    
    cv2.destroyAllWindows()


# -----------------------------
# Robot Control Thread
# -----------------------------

def robot_control_loop(robot, shared: SharedState, stop_event: threading.Event):
    """P-control loop for robot movement."""
    while not stop_event.is_set():
        try:
            with shared.lock:
                cf = int(shared.settings.control.control_freq)
                kp = float(shared.settings.control.kp)
                pitch = float(shared.pitch)
                
                # Wrist flex rule
                shared.target_positions["wrist_flex"] = (
                    -float(shared.target_positions["shoulder_lift"])
                    - float(shared.target_positions["elbow_flex"])
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
            for j, tgt in targets.items():
                if j in cur:
                    err = float(tgt) - float(cur[j])
                    action[f"{j}.pos"] = float(cur[j] + kp * err)
            
            if action:
                robot.send_action(action)
            
            time.sleep(1.0 / max(1, cf))
        
        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            print(f"[CONTROL] Error: {e}")
            time.sleep(0.1)


# -----------------------------
# Camera utilities
# -----------------------------

def list_cameras(max_index=8) -> List[int]:
    """List available camera indices."""
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
    load_dotenv()
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + "VOICE-CONTROLLED ROBOT - SO101 + YOLO + OpenAI".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set!")
        print("   Voice control will not work without it.")
        print("   Set it in your environment or .env file.")
        print()
    
    settings = load_settings(SETTINGS_PATH)
    shared = SharedState(settings)
    
    robot = None
    cap = None
    voice_runner = None
    control_thread = None
    yolo_thread = None
    
    try:
        # Import LeRobot
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except Exception:
            from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
        
        # from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        
        # Robot configuration
        print("─" * 70)
        print("ROBOT CONFIGURATION")
        print("─" * 70)
        port = input("SO101 robot USB port (default /dev/ttyACM1): ").strip() or "/dev/ttyACM1"
        robot_id = input("Robot id (default SO101_follower): ").strip() or "SO101_follower"
        print()
        
        # Camera configuration for robot
        # cameras_config = {
        #     "wrist": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30),
        #     "top": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
        # }
        
        # robot_config = SO101FollowerConfig(port=port, id=robot_id, cameras=cameras_config, use_degrees=True)
        robot_config = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
        robot = SO101Follower(robot_config)
        robot.connect()
        
        # Initialize robot controller
        controller = RobotController(robot=robot, shared_state=shared)
        
        # YOLO camera selection
        print("─" * 70)
        print("YOLO CAMERA CONFIGURATION")
        print("─" * 70)
        cam_idx = settings.yolo.camera_index
        if cam_idx is None:
            cams = list_cameras()
            if not cams:
                raise RuntimeError("No camera found")
            print(f"Available cameras: {cams}")
            cam_idx = int(input(f"Select camera index {cams}: ").strip())
            settings.yolo.camera_index = cam_idx
            save_settings(SETTINGS_PATH, settings)
        else:
            print(f"Using camera {cam_idx}")
        print()
        
        cap = cv2.VideoCapture(int(cam_idx))
        if not cap.isOpened():
            raise RuntimeError("Camera open failed")
        
        # Target object configuration
        print("─" * 70)
        print("TARGET OBJECT CONFIGURATION")
        print("─" * 70)
        current_targets = settings.yolo.target_objects or ["cup"]
        print(f"Current target(s): {', '.join(current_targets)}")
        new_target = input(f"Enter new target (ENTER to keep): ").strip()
        if new_target:
            new_targets = [t.strip().lower() for t in new_target.split(",") if t.strip()]
            if new_targets:
                settings.yolo.target_objects = new_targets
                save_settings(SETTINGS_PATH, settings)
                print(f"✅ Target(s): {', '.join(new_targets)}")
        print()
        
        # Load YOLO model
        print("Loading YOLO model...")
        yolo_model = YOLO(settings.yolo.model_path)
        print("✅ YOLO model loaded")
        print()
        
        # Load pretrained policy (optional)
        print("─" * 70)
        print("PRETRAINED POLICY (disqbled)")
        print("─" * 70)
        policy = None
        try:
            raise 'Implementation error'
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            from lerobot.policies.factory import make_pre_post_processors
            
            print(f"Loading from: {PRETRAINED_POLICY_PATH}")
            policy = PI05Policy.from_pretrained(PRETRAINED_POLICY_PATH)
            if torch.cuda.is_available():
                policy.to("cuda")
                print("✅ Policy loaded on GPU (CUDA)")
            else:
                policy.to("cpu")
                print("⚠️  Policy loaded on CPU (no CUDA)")
            
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy,
                pretrained_path=PRETRAINED_POLICY_PATH,
            )
            controller.set_policy(policy, preprocessor, postprocessor)
        except Exception as e:
            print(f"⚠️  Could not load policy: {e}")
            print("   Task execution will not be available.")
        print()
        
        # Initialize voice agent
        print("─" * 70)
        print("CRC ASSISTANT - VOICE AGENT")
        print("─" * 70)
        
        # Exit callback
        def on_exit_request():
            print("[SYS] Exit requested by CRC Assistant...")
            shared.stop_event.set()
        
        voice_agent = VoiceRobotAgent(
            robot_controller=controller,
            llm_model=settings.voice.llm_model,
            stt_model=settings.voice.stt_model,
            tts_model=settings.voice.tts_model,
            tts_voice=settings.voice.tts_voice,
            on_exit_request=on_exit_request,
        )
        
        voice_runner = VoiceLoopRunner(
            agent=voice_agent,
            sample_rate=settings.voice.sample_rate,
            vad_start_db=settings.voice.vad_start_db,
            vad_end_db=settings.voice.vad_end_db,
            vad_frame_ms=settings.voice.vad_frame_ms,
            vad_preroll_ms=settings.voice.vad_preroll_ms,
            vad_end_silence_ms=settings.voice.vad_end_silence_ms,
            vad_max_record_s=settings.voice.vad_max_record_s,
            vad_required_start_frames=settings.voice.vad_required_start_frames,
            shared_state=shared,  # For keyboard mode
        )
        print("✅ CRC Assistant initialized")
        print()
        
        # Print controls
        print("─" * 70)
        print("CONTROLS")
        print("─" * 70)
        print("  Voice commands (speak naturally to CRC Assistant):")
        print("    - 'Move left', 'Go up a lot', 'Forward a little'")
        print("    - 'Turn right' (rotates base), 'Turn the head left' (rotates camera)")
        print("    - 'Open the gripper', 'Close the gripper'")
        print("    - 'Track the cup', 'Follow the person', 'Stop tracking'")
        print("    - 'Save this position as home', 'Go to home'")
        print("    - 'What positions are saved?', 'Get status'")
        print("    - 'Goodbye', 'Exit' (to quit the program)")
        print()
        print("  Slash commands (type in terminal):")
        print("    /voice    - Voice control mode (speak to control)")
        print("    /text     - Text control mode (type commands)")
        print("    /keyboard - Keyboard control mode (arrow keys, etc.)")
        print("    /reset    - Reset conversation history")
        print("    /status   - Show robot status")
        print("    /help     - Show detailed help")
        print("    /quit     - Exit program")
        print()
        print("  YOLO window:")
        print("    ESC - Exit program")
        print()
        print("─" * 70)
        print()
        print("🚀 Starting CRC Assistant...")
        print()
        
        # Start threads
        yolo_thread = threading.Thread(
            target=yolo_loop,
            args=(yolo_model, cap, shared, controller),
            daemon=True
        )
        yolo_thread.start()
        
        control_thread = threading.Thread(
            target=robot_control_loop,
            args=(robot, shared, shared.stop_event),
            daemon=True
        )
        control_thread.start()
        
        voice_runner.start()
        
        print("🎤 CRC Assistant is ready! Speak to control the robot.")
        print("   (Type /help for all commands, /keyboard for keyboard mode)")
        print()
        
        # Main loop - handle stdin for text commands
        while not shared.stop_event.is_set():
            try:
                # Non-blocking input check
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        voice_runner.send_text(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n[SYS] Ctrl+C received, exiting...")
                break
            except Exception:
                time.sleep(0.1)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    
    finally:
        print("\n[SYS] Shutting down...")
        
        # Save settings
        save_settings(SETTINGS_PATH, settings)
        
        # Stop threads
        shared.stop_event.set()
        
        if voice_runner:
            voice_runner.stop()
        
        if cap is not None:
            cap.release()
        
        cv2.destroyAllWindows()
        
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        
        print("[SYS] Goodbye!")


if __name__ == "__main__":
    main()
