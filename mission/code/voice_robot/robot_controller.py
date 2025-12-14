#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot Controller Module

Provides a thread-safe interface to the SO101 robot arm with methods for:
- End-effector movement (x, y, z axes)
- Gripper control (open/close)
- Head/pan rotation
- YOLO tracking toggle
- Policy inference
- Pose memory (save/recall)

All methods are designed to be called from the voice agent thread while
the robot control loop runs in another thread.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Default movement parameters (used when step_size is not provided)
DEFAULT_LINEAR_STEP = 0.015  # 15mm default step
DEFAULT_ROTATION_STEP = 10.0  # 10 degrees default step


def inverse_kinematics_2d(x: float, y: float, l1: float = 0.1159, l2: float = 0.1350) -> Tuple[float, float]:
    """Planar IK for SO101 arm. Returns (shoulder_lift_deg, elbow_flex_deg)."""
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


@dataclass
class RobotState:
    """Thread-safe robot state container."""
    # End-effector position (2D planar)
    current_x: float = 0.1629
    current_y: float = 0.1131
    pitch: float = 0.0
    
    # Joint target positions (degrees)
    target_positions: Dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    })
    
    # YOLO tracking state
    follow_enabled: bool = False
    target_objects: List[str] = field(default_factory=lambda: ["cup"])
    
    # Detection state (updated by YOLO thread)
    last_frame_shape: Optional[Tuple[int, int]] = None
    last_best_box: Optional[Tuple[float, float, float, float]] = None
    last_best_conf: Optional[float] = None
    
    # Control parameters
    control_freq: int = 20
    kp: float = 0.3
    ee_step: float = 0.004
    pitch_step: float = 1.0
    
    # Saved poses
    saved_poses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Debouncing
    last_toggle_time: float = 0.0
    
    # Selected pose slot
    selected_slot: str = "1"


class RobotController:
    """
    Thread-safe robot controller for the SO101 arm.
    
    Provides high-level methods for robot control that can be called
    from any thread. All methods acquire the lock before modifying state.
    """
    
    def __init__(self, robot=None, shared_state=None):
        """
        Initialize the robot controller.
        
        Args:
            robot: The actual SO101 robot object (can be set later)
            shared_state: Optional SharedState object from the original script
        """
        self.robot = robot
        self.shared_state = shared_state  # For compatibility with original SharedState
        self.state = RobotState()
        self.lock = threading.RLock()  # Reentrant lock for nested calls
        self.stop_event = threading.Event()
        
        # Callbacks for event notification
        self._on_tracking_started: Optional[Callable[[str], None]] = None
        self._on_tracking_stopped: Optional[Callable[[], None]] = None
        self._on_policy_started: Optional[Callable[[str], None]] = None
        self._on_policy_finished: Optional[Callable[[str], None]] = None
        
        # Policy components (set externally)
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.events = None  # Keyboard events for policy
    
    def set_robot(self, robot):
        """Set the robot hardware interface."""
        with self.lock:
            self.robot = robot
    
    def set_shared_state(self, shared_state):
        """Set the shared state object for compatibility with original code."""
        with self.lock:
            self.shared_state = shared_state
    
    def set_policy(self, policy, preprocessor=None, postprocessor=None, events=None):
        """Set the pretrained policy for inference."""
        with self.lock:
            self.policy = policy
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self.events = events
    
    # -------------------------
    # Movement Methods
    # -------------------------
    
    def move_ee(self, axis: str, direction: str, step_size: Optional[float] = None, amount: Optional[str] = None) -> Dict[str, Any]:
        """
        Move the end-effector along an axis.
        
        Args:
            axis: "x", "y", or "z" (x=left/right, y=up/down, z=forward/backward)
            direction: "positive" or "negative"
            step_size: Step size in meters (0.005 to 0.05 typical). If provided, overrides amount.
            amount: DEPRECATED - "un_peu" (small) or "beaucoup" (large). Use step_size instead.
        
        Returns:
            Dict with status and details
        """
        # Determine step size
        if step_size is not None:
            step = float(step_size)
        elif amount is not None:
            # Legacy support for amount strings (deprecated)
            if amount == "un_peu":
                step = 0.010  # 10mm
            elif amount == "beaucoup":
                step = 0.035  # 35mm
            else:
                step = DEFAULT_LINEAR_STEP
        else:
            step = DEFAULT_LINEAR_STEP
        
        # Clamp step to reasonable bounds
        step = max(0.001, min(0.05, step))
        
        # Apply direction
        if direction == "negative":
            step = -step
        
        with self.lock:
            # For SO101, we use 2D IK:
            # - x axis: moves current_x (horizontal reach)
            # - y axis: moves current_y (vertical)
            # - z axis: same as y for this arm (forward/back)
            
            if axis == "x":
                self.state.current_x += step
            elif axis == "y":
                self.state.current_y += step
            elif axis == "z":
                # Z maps to forward/backward, which is similar to Y in our 2D plane
                self.state.current_y += step
            
            # Compute IK
            j2, j3 = inverse_kinematics_2d(self.state.current_x, self.state.current_y)
            self.state.target_positions["shoulder_lift"] = float(j2)
            self.state.target_positions["elbow_flex"] = float(j3)
            
            # Also update SharedState if available
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.current_x = self.state.current_x
                    self.shared_state.current_y = self.state.current_y
                    self.shared_state.target_positions["shoulder_lift"] = float(j2)
                    self.shared_state.target_positions["elbow_flex"] = float(j3)
            
            return {
                "ok": True,
                "axis": axis,
                "direction": direction,
                "step_mm": abs(step) * 1000,
                "new_x": self.state.current_x,
                "new_y": self.state.current_y,
            }
    
    def head_turn(self, direction: str, step_size: Optional[float] = None, amount: Optional[str] = None) -> Dict[str, Any]:
        """
        Rotate the robot head/camera (shoulder_pan).
        
        Args:
            direction: "left" or "right"
            step_size: Rotation in degrees (3 to 35 typical). If provided, overrides amount.
            amount: DEPRECATED - "un_peu" (small) or "beaucoup" (large). Use step_size instead.
        
        Returns:
            Dict with status and details
        """
        # Determine rotation amount
        if step_size is not None:
            step = float(step_size)
        elif amount is not None:
            # Legacy support (deprecated)
            if amount == "un_peu":
                step = 8.0  # degrees
            elif amount == "beaucoup":
                step = 25.0  # degrees
            else:
                step = DEFAULT_ROTATION_STEP
        else:
            step = DEFAULT_ROTATION_STEP
        
        # Clamp step to reasonable bounds
        step = max(1.0, min(45.0, step))
        
        # Apply direction (left = positive, right = negative for shoulder_pan)
        if direction == "right":
            step = -step
        
        with self.lock:
            current_pan = self.state.target_positions.get("shoulder_pan", 0.0)
            new_pan = current_pan + step
            self.state.target_positions["shoulder_pan"] = new_pan
            
            # Update SharedState if available
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.target_positions["shoulder_pan"] = new_pan
            
            return {
                "ok": True,
                "direction": direction,
                "step_deg": abs(step),
                "new_pan_deg": new_pan,
            }
    
    def gripper(self, action: str) -> Dict[str, Any]:
        """
        Control the gripper.
        
        Args:
            action: "open" or "close"
        
        Returns:
            Dict with status and details
        """
        with self.lock:
            if action == "open":
                # Gripper open = positive position (e.g., 50 degrees)
                new_pos = 50.0
            elif action == "close":
                # Gripper close = negative position (e.g., -10 degrees)
                new_pos = -10.0
            else:
                return {"ok": False, "error": f"Unknown gripper action: {action}"}
            
            self.state.target_positions["gripper"] = new_pos
            
            # Update SharedState if available
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.target_positions["gripper"] = new_pos
            
            return {
                "ok": True,
                "action": action,
                "gripper_pos_deg": new_pos,
            }
    
    def adjust_pitch(self, direction: str, step_size: Optional[float] = None, amount: Optional[str] = None) -> Dict[str, Any]:
        """
        Adjust the wrist pitch (tilt).
        
        Args:
            direction: "up" or "down"
            step_size: Adjustment in degrees (3 to 35 typical). If provided, overrides amount.
            amount: DEPRECATED - "un_peu" (small) or "beaucoup" (large). Use step_size instead.
        
        Returns:
            Dict with status and details
        """
        if step_size is not None:
            step = float(step_size)
        elif amount is not None:
            # Legacy support (deprecated)
            if amount == "un_peu":
                step = 5.0
            elif amount == "beaucoup":
                step = 15.0
            else:
                step = DEFAULT_ROTATION_STEP
        else:
            step = DEFAULT_ROTATION_STEP
        
        # Clamp
        step = max(1.0, min(30.0, step))
        
        if direction == "down":
            step = -step
        
        with self.lock:
            self.state.pitch += step
            
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.pitch = self.state.pitch
            
            return {
                "ok": True,
                "direction": direction,
                "step_deg": abs(step),
                "new_pitch_deg": self.state.pitch,
            }
    
    def wrist_roll(self, direction: str, step_size: Optional[float] = None, amount: Optional[str] = None) -> Dict[str, Any]:
        """
        Rotate the wrist roll joint.
        
        Args:
            direction: "left" or "right"
            step_size: Rotation in degrees (3 to 35 typical). If provided, overrides amount.
            amount: DEPRECATED - "un_peu" or "beaucoup". Use step_size instead.
        
        Returns:
            Dict with status
        """
        if step_size is not None:
            step = float(step_size)
        elif amount is not None:
            # Legacy support (deprecated)
            if amount == "un_peu":
                step = 8.0
            elif amount == "beaucoup":
                step = 20.0
            else:
                step = DEFAULT_ROTATION_STEP
        else:
            step = DEFAULT_ROTATION_STEP
        
        # Clamp
        step = max(1.0, min(45.0, step))
        
        if direction == "left":
            step = -step
        
        with self.lock:
            current = self.state.target_positions.get("wrist_roll", 0.0)
            new_val = current + step
            self.state.target_positions["wrist_roll"] = new_val
            
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.target_positions["wrist_roll"] = new_val
            
            return {
                "ok": True,
                "direction": direction,
                "step_deg": abs(step),
                "new_wrist_roll_deg": new_val,
            }
    
    # -------------------------
    # Tracking Methods
    # -------------------------
    
    def start_tracking(self, object_name: str) -> Dict[str, Any]:
        """
        Start YOLO tracking for a specific object.
        
        Args:
            object_name: Name of the COCO object to track (e.g., "cup", "person")
        
        Returns:
            Dict with status and details
        """
        with self.lock:
            # Update target objects
            self.state.target_objects = [object_name.lower().strip()]
            self.state.follow_enabled = True
            
            # Update SharedState if available
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.settings.yolo.target_objects = self.state.target_objects
                    self.shared_state.follow_enabled = True
            
            if self._on_tracking_started:
                self._on_tracking_started(object_name)
            
            return {
                "ok": True,
                "object_name": object_name,
                "tracking_enabled": True,
            }
    
    def stop_tracking(self) -> Dict[str, Any]:
        """
        Stop YOLO tracking.
        
        Returns:
            Dict with status
        """
        with self.lock:
            self.state.follow_enabled = False
            
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.follow_enabled = False
            
            if self._on_tracking_stopped:
                self._on_tracking_stopped()
            
            return {
                "ok": True,
                "tracking_enabled": False,
            }
    
    def toggle_tracking(self) -> Dict[str, Any]:
        """
        Toggle YOLO tracking on/off.
        
        Returns:
            Dict with new tracking state
        """
        with self.lock:
            current_time = time.time()
            
            # Debounce (300ms)
            if current_time - self.state.last_toggle_time < 0.3:
                return {
                    "ok": False,
                    "error": "Toggle too fast, debounced",
                    "tracking_enabled": self.state.follow_enabled,
                }
            
            self.state.last_toggle_time = current_time
            self.state.follow_enabled = not self.state.follow_enabled
            new_state = self.state.follow_enabled
            
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.follow_enabled = new_state
            
            return {
                "ok": True,
                "tracking_enabled": new_state,
            }
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """
        Get current tracking status.
        
        Returns:
            Dict with tracking info
        """
        with self.lock:
            has_detection = self.state.last_best_box is not None
            if self.shared_state:
                with self.shared_state.lock:
                    has_detection = self.shared_state.last_best_box is not None
                    conf = self.shared_state.last_best_conf
            else:
                conf = self.state.last_best_conf
            
            return {
                "tracking_enabled": self.state.follow_enabled,
                "target_objects": self.state.target_objects,
                "has_detection": has_detection,
                "detection_confidence": conf,
            }
    
    # -------------------------
    # Policy Methods
    # -------------------------
    
    def start_task(self, task_name: str) -> Dict[str, Any]:
        """
        Start a high-level task using the pretrained policy.
        
        Args:
            task_name: Name of the task to execute (e.g., "grab_camera")
        
        Returns:
            Dict with status
        """
        if self.policy is None:
            return {
                "ok": False,
                "error": "No policy loaded",
            }
        
        if self._on_policy_started:
            self._on_policy_started(task_name)
        
        # Note: Actual policy execution happens in the control loop
        # This just signals that a task should start
        print(f"[ROBOT] Starting task: {task_name}")
        
        # The actual inference is run by run_policy_inference in the main control
        # For now, we just return success and the control loop handles it
        
        if self._on_policy_finished:
            self._on_policy_finished(task_name)
        
        return {
            "ok": True,
            "task_name": task_name,
            "message": f"Task '{task_name}' started",
        }
    
    # -------------------------
    # Pose Memory Methods
    # -------------------------
    
    def save_pose(self, slot: str) -> Dict[str, Any]:
        """
        Save the current robot pose to a memory slot.
        
        Args:
            slot: Slot name/number (e.g., "1", "home")
        
        Returns:
            Dict with status
        """
        with self.lock:
            # Get current joint positions from robot
            if self.robot:
                obs = self.robot.get_observation()
                joints = {}
                for k, v in obs.items():
                    if k.endswith(".pos"):
                        j = k.removesuffix(".pos")
                        joints[j] = float(v)
            else:
                joints = dict(self.state.target_positions)
            
            pose_data = {
                "joints_deg": joints,
                "current_x": self.state.current_x,
                "current_y": self.state.current_y,
                "pitch": self.state.pitch,
                "timestamp": time.time(),
            }
            
            self.state.saved_poses[str(slot)] = pose_data
            
            return {
                "ok": True,
                "slot": slot,
                "message": f"Pose saved to slot '{slot}'",
            }
    
    def goto_pose(self, slot: str) -> Dict[str, Any]:
        """
        Move to a previously saved pose.
        
        Args:
            slot: Slot name/number
        
        Returns:
            Dict with status
        """
        with self.lock:
            pose = self.state.saved_poses.get(str(slot))
            if pose is None:
                return {
                    "ok": False,
                    "error": f"No pose saved in slot '{slot}'",
                    "available_slots": list(self.state.saved_poses.keys()),
                }
            
            # Update target positions
            if "joints_deg" in pose:
                for k, v in pose["joints_deg"].items():
                    if k in self.state.target_positions:
                        self.state.target_positions[k] = float(v)
            
            if "current_x" in pose:
                self.state.current_x = float(pose["current_x"])
            if "current_y" in pose:
                self.state.current_y = float(pose["current_y"])
            if "pitch" in pose:
                self.state.pitch = float(pose["pitch"])
            
            # Update SharedState if available
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.current_x = self.state.current_x
                    self.shared_state.current_y = self.state.current_y
                    self.shared_state.pitch = self.state.pitch
                    for k, v in self.state.target_positions.items():
                        self.shared_state.target_positions[k] = v
            
            return {
                "ok": True,
                "slot": slot,
                "message": f"Moving to pose in slot '{slot}'",
            }
    
    # -------------------------
    # Status Methods
    # -------------------------
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get complete robot status.
        
        Returns:
            Dict with all status info
        """
        with self.lock:
            return {
                "current_x": self.state.current_x,
                "current_y": self.state.current_y,
                "pitch": self.state.pitch,
                "target_positions": dict(self.state.target_positions),
                "tracking_enabled": self.state.follow_enabled,
                "target_objects": self.state.target_objects,
                "has_policy": self.policy is not None,
                "saved_pose_slots": list(self.state.saved_poses.keys()),
                "selected_slot": self.state.selected_slot,
            }
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop all robot movement.
        
        Returns:
            Dict with status
        """
        with self.lock:
            self.state.follow_enabled = False
            
            if self.shared_state:
                with self.shared_state.lock:
                    self.shared_state.follow_enabled = False
            
            return {
                "ok": True,
                "message": "Robot stopped",
            }
