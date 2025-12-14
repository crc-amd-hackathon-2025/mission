#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice Agent Module

Provides a voice-controlled interface for the robot arm using:
- OpenAI Whisper for speech-to-text
- GPT-4 with tool calling for command interpretation
- OpenAI TTS for speech synthesis
- VAD (Voice Activity Detection) for hands-free operation

The agent understands natural language commands and translates them
to robot actions via the RobotController interface.
"""

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

from .robot_controller import RobotController


# -----------------------------
# Environment helpers
# -----------------------------

def env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    return float(v) if v else default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    return int(v) if v else default

def env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


# -----------------------------
# Audio: VAD with absolute thresholds
# -----------------------------

def _rms_db(x: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate RMS level in dB."""
    x = x.reshape(-1).astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x) + eps))
    return 20.0 * np.log10(rms + eps)


def record_wav_vad_absolute(
    out_path: Path,
    sample_rate: int,
    frame_ms: int,
    start_db: float,
    end_db: float,
    required_start_frames: int,
    end_silence_ms: int,
    max_record_s: float,
    preroll_ms: int,
    verbose: bool = True,
    stop_event: Optional[threading.Event] = None,
) -> Path:
    """
    Record audio with Voice Activity Detection.
    
    - Start recording when level >= start_db for N consecutive frames
    - Stop when level <= end_db for end_silence_ms
    - Maintains a pre-roll buffer to catch speech onset
    
    Args:
        out_path: Path to save the WAV file
        sample_rate: Audio sample rate
        frame_ms: Frame duration in milliseconds
        start_db: Threshold to start recording (dB)
        end_db: Threshold to stop recording (dB)
        required_start_frames: Number of consecutive frames above threshold to start
        end_silence_ms: Duration of silence to stop recording
        max_record_s: Maximum recording duration
        preroll_ms: Pre-roll buffer duration
        verbose: Print status messages
        stop_event: Optional event to stop recording early
    
    Returns:
        Path to the recorded WAV file
    """
    if start_db <= end_db and verbose:
        print("[WARN] VAD_START_DB should be > VAD_END_DB (hysteresis).")

    frame_len = int(sample_rate * frame_ms / 1000)
    preroll_frames = max(0, int(preroll_ms / frame_ms))
    end_silence_frames = max(1, int(end_silence_ms / frame_ms))
    max_frames = int(max_record_s * 1000 / frame_ms)

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    if verbose:
        print(f"[AUDIO] VAD: start_db={start_db:.1f} end_db={end_db:.1f}")

    preroll_buf: List[np.ndarray] = []
    frames_buf: List[np.ndarray] = []

    speech_started = False
    start_count = 0
    silence_count = 0

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_len,
        callback=callback,
    ):
        for _ in range(max_frames):
            # Check stop event
            if stop_event and stop_event.is_set():
                break
            
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                continue

            db = _rms_db(chunk)

            # Pre-roll buffer
            preroll_buf.append(chunk)
            if len(preroll_buf) > preroll_frames:
                preroll_buf.pop(0)

            if not speech_started:
                if db >= start_db:
                    start_count += 1
                else:
                    start_count = 0

                if start_count >= required_start_frames:
                    speech_started = True
                    frames_buf.extend(preroll_buf)
                    frames_buf.append(chunk)
                    if verbose:
                        print(f"[AUDIO] Speech start (db={db:.1f})")
                continue

            # Speech in progress
            frames_buf.append(chunk)

            if db <= end_db:
                silence_count += 1
            else:
                silence_count = 0

            if silence_count >= end_silence_frames:
                if verbose:
                    print(f"[AUDIO] Speech end (db={db:.1f})")
                break

    if not frames_buf:
        if verbose:
            print("[AUDIO] Nothing captured.")
        sf.write(str(out_path), np.zeros((0, 1), dtype="float32"), sample_rate, subtype="PCM_16")
        return out_path

    audio = np.concatenate(frames_buf, axis=0)
    sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
    if verbose:
        dur = audio.shape[0] / sample_rate
        print(f"[AUDIO] Saved: {out_path} ({dur:.2f}s)")
    return out_path


def play_wav(path: Path):
    """Play a WAV file."""
    try:
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(str(path))
        play_obj = wave_obj.play()
        play_obj.wait_done()
        return
    except Exception:
        data, sr = sf.read(str(path), dtype="float32")
        sd.play(data, sr)
        sd.wait()


# -----------------------------
# Robot Command Data Class
# -----------------------------

@dataclass
class RobotCommand:
    """Represents a command sent to the robot."""
    name: str
    args: Dict[str, Any]


# -----------------------------
# Voice Robot Agent
# -----------------------------

SYSTEM_PROMPT = """You are CRC Assistant, a friendly and helpful voice assistant that controls a SO101 robot arm.

You have a cheerful personality and enjoy helping users interact with the robot. You can chat about anything, but when the user asks for a physical action, you call the appropriate tool.

## Robot Capabilities:
- Move the end-effector (gripper) along x (forward/backward), y (up/down) axes
- Rotate the robot base left or right (this is how the robot "turns")
- Open or close the gripper
- Adjust wrist pitch (tilt up/down)
- Rotate the wrist (wrist roll) left or right
- Enable visual tracking (YOLO) of an object (e.g., "cup", "person", "bottle")
- Disable visual tracking
- Save the current position to a memory slot
- Go to a saved position
- Get robot status
- Stop all movements
- Exit the program

## Movement Conventions:
- `amount`: A float value between 0.0 and 1.0 where:
  - 0.1 = very small movement (~5mm or ~3 degrees)
  - 0.3 = small movement (~15mm or ~10 degrees)  
  - 0.5 = medium movement (~25mm or ~15 degrees)
  - 0.7 = large movement (~35mm or ~25 degrees)
  - 1.0 = maximum movement (~50mm or ~35 degrees)
- EE axes: x = forward/backward, y = up/down
- direction: "positive" or "negative" for linear axes, "left" or "right" for rotations
- gripper: "open" or "close"

## IMPORTANT Behavior Rules:
1. When the user says "turn right", "turn left", "go left", "go right" → Use head_turn to rotate the robot base. This is how the robot turns/rotates. In your response, you can mention that this rotates the entire robot base.

2. Always choose an appropriate `amount` value based on context:
   - "a little" / "slightly" / "un peu" → 0.2 to 0.3
   - Default (no qualifier) → 0.4 to 0.5
   - "more" / "a lot" / "beaucoup" → 0.7 to 0.9
   - "maximum" / "all the way" → 1.0

3. If a request is ambiguous, ask ONE short clarifying question.

4. After each action, briefly confirm what you did.

5. Be friendly and conversational! You're CRC Assistant, and you enjoy your job.

## Saved Positions:
{saved_positions}

Respond concisely and naturally. Feel free to add personality but stay helpful!"""


class VoiceRobotAgent:
    """
    Voice-controlled robot agent using OpenAI APIs.
    
    Handles:
    - Speech-to-text transcription
    - Natural language understanding with tool calling
    - Tool dispatch to RobotController
    - Text-to-speech response generation
    """
    
    def __init__(
        self,
        robot_controller: RobotController,
        llm_model: str = "gpt-4.1",
        stt_model: str = "gpt-4o-transcribe",
        tts_model: str = "gpt-4o-mini-tts",
        tts_voice: str = "coral",
        on_exit_request: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the voice agent.
        
        Args:
            robot_controller: The robot controller instance
            llm_model: OpenAI model for language understanding
            stt_model: OpenAI model for speech-to-text
            tts_model: OpenAI model for text-to-speech
            tts_voice: Voice for TTS
            on_exit_request: Callback when user requests program exit
        """
        self.client = OpenAI()
        self.robot = robot_controller
        self.llm_model = llm_model
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.on_exit_request = on_exit_request
        
        self.tools = self._build_tools()
        self.input_list: List[Dict[str, Any]] = []
        self._seed_system()
    
    def _get_saved_positions_info(self) -> str:
        """Get information about saved positions for the system prompt."""
        status = self.robot.get_status()
        slots = status.get("saved_pose_slots", [])
        if not slots:
            return "No positions are currently saved."
        return f"Available saved positions: {', '.join(slots)}"
    
    def _seed_system(self):
        """Initialize the conversation with the system prompt."""
        prompt = SYSTEM_PROMPT.format(saved_positions=self._get_saved_positions_info())
        self.input_list = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self._seed_system()
    
    def _build_tools(self) -> List[Dict[str, Any]]:
        """Build the tool definitions for the LLM."""
        return [
            {
                "type": "function",
                "name": "move_ee",
                "description": "Move the end-effector (EE) of the robot arm forward/backward or up/down. Use axis x for forward/backward, axis y for up/down. Do NOT use this for left/right rotation - use head_turn instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "axis": {
                            "type": "string",
                            "enum": ["x", "y"],
                            "description": "Movement axis: x=forward/backward, y=up/down"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                            "description": "Movement direction. For x: positive=forward, negative=backward. For y: positive=up, negative=down."
                        },
                        "amount": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Movement amount from 0.0 (tiny) to 1.0 (maximum). 0.3=small, 0.5=medium, 0.8=large."
                        },
                    },
                    "required": ["axis", "direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "head_turn",
                "description": "Rotate the robot base left or right. Use this for ANY 'turn left', 'turn right', 'go left', 'go right' command. This rotates the entire robot arm base (shoulder_pan).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["left", "right"],
                            "description": "Rotation direction"
                        },
                        "amount": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Rotation amount from 0.0 (tiny) to 1.0 (maximum). 0.3=small, 0.5=medium, 0.8=large."
                        },
                    },
                    "required": ["direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "gripper",
                "description": "Open or close the robot gripper.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["open", "close"],
                            "description": "Action to perform"
                        }
                    },
                    "required": ["action"],
                },
            },
            {
                "type": "function",
                "name": "adjust_pitch",
                "description": "Adjust the wrist pitch (tilt) up or down.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down"],
                            "description": "Tilt direction"
                        },
                        "amount": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Adjustment amount from 0.0 to 1.0"
                        },
                    },
                    "required": ["direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "wrist_roll",
                "description": "Rotate the wrist (roll) left or right.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["left", "right"],
                            "description": "Rotation direction"
                        },
                        "amount": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Rotation amount from 0.0 to 1.0"
                        },
                    },
                    "required": ["direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "start_tracking",
                "description": "Enable visual tracking (YOLO) of an object. The robot will follow the object with the camera.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Name of the object to track (e.g., cup, person, bottle, cell phone)"
                        }
                    },
                    "required": ["object_name"],
                },
            },
            {
                "type": "function",
                "name": "stop_tracking",
                "description": "Disable visual tracking. The robot stops following the object.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "type": "function",
                "name": "save_pose",
                "description": "Save the current robot position to a memory slot (0-9 or a name like 'home').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot": {
                            "type": "string",
                            "description": "Slot number or name (e.g., '1', 'home', 'grab_position')"
                        }
                    },
                    "required": ["slot"],
                },
            },
            {
                "type": "function",
                "name": "goto_pose",
                "description": "Move to a previously saved position.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot": {
                            "type": "string",
                            "description": "Slot number or name"
                        }
                    },
                    "required": ["slot"],
                },
            },
            {
                "type": "function",
                "name": "list_saved_poses",
                "description": "Get the list of all saved positions/poses.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "type": "function",
                "name": "get_status",
                "description": "Get the current status of the robot (position, tracking state, etc.).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "type": "function",
                "name": "stop",
                "description": "Stop all robot movements (emergency stop or pause).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "type": "function",
                "name": "exit_program",
                "description": "Exit the voice control program and shut down the robot. Use when user says 'quit', 'exit', 'goodbye', 'shut down', etc.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]
    
    def stt(self, wav_path: Path) -> str:
        """
        Transcribe audio to text.
        
        Args:
            wav_path: Path to the WAV file
        
        Returns:
            Transcribed text
        """
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            return ""
        with open(wav_path, "rb") as f:
            tr = self.client.audio.transcriptions.create(
                model=self.stt_model,
                file=f,
                response_format="text",
            )
        return (tr.text if hasattr(tr, "text") else str(tr)).strip()
    
    def tts_to_wav(self, text: str, out_path: Path) -> Path:
        """
        Convert text to speech and save to WAV.
        
        Args:
            text: Text to synthesize
            out_path: Path to save the WAV file
        
        Returns:
            Path to the generated WAV file
        """
        with self.client.audio.speech.with_streaming_response.create(
            model=self.tts_model,
            voice=self.tts_voice,
            input=text,
            response_format="wav",
            instructions="Natural, clear, friendly voice. Speak in the same language as the input text.",
        ) as response:
            response.stream_to_file(out_path)
        return out_path
    
    def _dispatch_tool(self, name: str, arguments_json: str) -> Dict[str, Any]:
        """
        Dispatch a tool call to the robot controller.
        
        Args:
            name: Tool name
            arguments_json: JSON string of arguments
        
        Returns:
            Tool result dictionary
        """
        args = json.loads(arguments_json) if arguments_json else {}
        
        # Convert float amount to appropriate step size
        def get_linear_step(amount: float) -> float:
            """Convert 0-1 amount to meters (0.005 to 0.05m)."""
            return 0.005 + amount * 0.045  # 5mm to 50mm
        
        def get_rotation_step(amount: float) -> float:
            """Convert 0-1 amount to degrees (3 to 35 degrees)."""
            return 3.0 + amount * 32.0  # 3° to 35°
        
        # Map tool names to RobotController methods
        if name == "move_ee":
            amount = float(args.get("amount", 0.5))
            step = get_linear_step(amount)
            return self.robot.move_ee(
                axis=args["axis"],
                direction=args["direction"],
                step_size=step
            )
        
        elif name == "head_turn":
            amount = float(args.get("amount", 0.5))
            step = get_rotation_step(amount)
            return self.robot.head_turn(
                direction=args["direction"],
                step_size=step
            )
        
        elif name == "gripper":
            return self.robot.gripper(action=args["action"])
        
        elif name == "adjust_pitch":
            amount = float(args.get("amount", 0.5))
            step = get_rotation_step(amount)
            return self.robot.adjust_pitch(
                direction=args["direction"],
                step_size=step
            )
        
        elif name == "wrist_roll":
            amount = float(args.get("amount", 0.5))
            step = get_rotation_step(amount)
            return self.robot.wrist_roll(
                direction=args["direction"],
                step_size=step
            )
        
        elif name == "start_tracking":
            return self.robot.start_tracking(object_name=args["object_name"])
        
        elif name == "stop_tracking":
            return self.robot.stop_tracking()
        
        elif name == "save_pose":
            return self.robot.save_pose(slot=args["slot"])
        
        elif name == "goto_pose":
            return self.robot.goto_pose(slot=args["slot"])
        
        elif name == "list_saved_poses":
            status = self.robot.get_status()
            slots = status.get("saved_pose_slots", [])
            if slots:
                return {"ok": True, "saved_poses": slots, "message": f"Saved positions: {', '.join(slots)}"}
            else:
                return {"ok": True, "saved_poses": [], "message": "No positions are currently saved."}
        
        elif name == "get_status":
            return self.robot.get_status()
        
        elif name == "stop":
            return self.robot.stop()
        
        elif name == "exit_program":
            if self.on_exit_request:
                self.on_exit_request()
            return {"ok": True, "message": "Exiting program..."}
        
        else:
            return {"ok": False, "error": f"Unknown tool: {name}"}
    
    def agent_step(self, user_text: str) -> Tuple[str, List[RobotCommand]]:
        """
        Process a user message and return the response.
        
        Args:
            user_text: The user's message
        
        Returns:
            Tuple of (response_text, list_of_robot_commands)
        """
        robot_cmds: List[RobotCommand] = []
        
        # Update system prompt with current saved positions
        self.input_list[0]["content"] = SYSTEM_PROMPT.format(
            saved_positions=self._get_saved_positions_info()
        )
        
        self.input_list.append({"role": "user", "content": user_text})
        
        # First API call - may result in tool calls
        resp = self.client.responses.create(
            model=self.llm_model,
            tools=self.tools,
            input=self.input_list,
        )
        self.input_list += resp.output
        
        # Process any tool calls
        has_tool_calls = False
        for item in resp.output:
            if getattr(item, "type", None) == "function_call":
                has_tool_calls = True
                tool_name = item.name
                tool_args = item.arguments
                robot_cmds.append(RobotCommand(name=tool_name, args=json.loads(tool_args) if tool_args else {}))
                
                tool_result = self._dispatch_tool(tool_name, tool_args)
                print(f"[TOOL] {tool_name} -> {tool_result}")
                
                self.input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(tool_result),
                    }
                )
        
        # If we had tool calls, make a second call for the final response
        if has_tool_calls:
            resp2 = self.client.responses.create(
                model=self.llm_model,
                tools=self.tools,
                input=self.input_list,
            )
            self.input_list += resp2.output
            return resp2.output_text.strip(), robot_cmds
        
        # No tool calls, just return the response
        return resp.output_text.strip(), robot_cmds


# -----------------------------
# Voice Loop Runner
# -----------------------------

class VoiceLoopRunner:
    """
    Manages the voice interaction loop in a separate thread.
    
    Handles:
    - Audio recording with VAD
    - Speech-to-text
    - Agent processing
    - Text-to-speech playback
    - Keyboard control mode
    """
    
    def __init__(
        self,
        agent: VoiceRobotAgent,
        sample_rate: int = 16000,
        vad_start_db: float = -19.0,
        vad_end_db: float = -24.0,
        vad_frame_ms: int = 20,
        vad_preroll_ms: int = 250,
        vad_end_silence_ms: int = 600,
        vad_max_record_s: float = 15.0,
        vad_required_start_frames: int = 3,
        tmp_dir: Optional[Path] = None,
        keyboard_controller: Optional[Any] = None,
        shared_state: Optional[Any] = None,
    ):
        """
        Initialize the voice loop runner.
        
        Args:
            agent: The VoiceRobotAgent instance
            sample_rate: Audio sample rate
            vad_start_db: VAD start threshold
            vad_end_db: VAD end threshold
            vad_frame_ms: VAD frame size
            vad_preroll_ms: Pre-roll buffer size
            vad_end_silence_ms: Silence duration to stop
            vad_max_record_s: Maximum recording duration
            vad_required_start_frames: Frames above threshold to start
            tmp_dir: Directory for temporary audio files
            keyboard_controller: Optional keyboard controller for keyboard mode
            shared_state: Optional shared state for keyboard mode
        """
        self.agent = agent
        self.sample_rate = sample_rate
        self.vad_start_db = vad_start_db
        self.vad_end_db = vad_end_db
        self.vad_frame_ms = vad_frame_ms
        self.vad_preroll_ms = vad_preroll_ms
        self.vad_end_silence_ms = vad_end_silence_ms
        self.vad_max_record_s = vad_max_record_s
        self.vad_required_start_frames = vad_required_start_frames
        
        self.tmp_dir = tmp_dir or Path("./_tmp_audio")
        self.tmp_dir.mkdir(exist_ok=True)
        
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.thread: Optional[threading.Thread] = None
        
        # Command queue for text input mode
        self.cmd_queue: "queue.Queue[str]" = queue.Queue()
        self.mode = "voice"  # "voice", "text", or "keyboard"
        
        # Keyboard mode components
        self.keyboard_controller = keyboard_controller
        self.shared_state = shared_state
        self.keyboard_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_listening: Optional[Callable[[], None]] = None
        self.on_processing: Optional[Callable[[], None]] = None
        self.on_speaking: Optional[Callable[[], None]] = None
        self.on_idle: Optional[Callable[[], None]] = None
    
    def _voice_loop(self):
        """Main voice loop (runs in background thread)."""
        print("[VOICE] Voice loop started")
        
        while not self.stop_event.is_set():
            # Check for pause
            if self.paused.is_set():
                time.sleep(0.1)
                continue
            
            # Process text commands if any
            while not self.cmd_queue.empty():
                try:
                    cmd = self.cmd_queue.get_nowait().strip()
                except queue.Empty:
                    break
                
                if not cmd:
                    continue
                
                if cmd.startswith("/"):
                    self._handle_command(cmd)
                elif self.mode == "text":
                    self._process_text_input(cmd)
            
            # Keyboard mode: don't process voice, just wait
            if self.mode == "keyboard":
                time.sleep(0.1)
                continue
            
            # Voice mode: listen for speech
            if self.mode == "voice" and not self.paused.is_set():
                if self.on_listening:
                    self.on_listening()
                
                wav_in = self.tmp_dir / "in.wav"
                record_wav_vad_absolute(
                    wav_in,
                    sample_rate=self.sample_rate,
                    frame_ms=self.vad_frame_ms,
                    start_db=self.vad_start_db,
                    end_db=self.vad_end_db,
                    required_start_frames=self.vad_required_start_frames,
                    end_silence_ms=self.vad_end_silence_ms,
                    max_record_s=self.vad_max_record_s,
                    preroll_ms=self.vad_preroll_ms,
                    verbose=True,
                    stop_event=self.stop_event,
                )
                
                if self.stop_event.is_set():
                    break
                
                if self.on_processing:
                    self.on_processing()
                
                user_text = self.agent.stt(wav_in)
                if not user_text:
                    continue
                
                print(f"[YOU] {user_text}")
                self._process_and_respond(user_text)
            
            time.sleep(0.01)
        
        print("[VOICE] Voice loop stopped")
    
    def _handle_command(self, cmd: str):
        """Handle slash commands."""
        cmd_lower = cmd.lower()
        
        if cmd_lower == "/quit" or cmd_lower == "/exit":
            self.stop()
        elif cmd_lower == "/reset":
            self.agent.reset_conversation()
            print("[SYS] Conversation reset.")
        elif cmd_lower == "/text":
            self.mode = "text"
            self._stop_keyboard_mode()
            print("[SYS] Text mode activated. Type your commands.")
        elif cmd_lower == "/voice":
            self.mode = "voice"
            self._stop_keyboard_mode()
            print("[SYS] Voice mode activated. Speak to control the robot.")
        elif cmd_lower == "/keyboard":
            self._start_keyboard_mode()
        elif cmd_lower == "/pause":
            self.paused.set()
            print("[SYS] Voice loop paused.")
        elif cmd_lower == "/resume":
            self.paused.clear()
            print("[SYS] Voice loop resumed.")
        elif cmd_lower == "/status":
            status = self.agent.robot.get_status()
            print(f"[STATUS] Mode: {self.mode}")
            print(f"[STATUS] Robot: {json.dumps(status, indent=2)}")
        elif cmd_lower == "/help":
            self._print_help()
        else:
            print(f"[SYS] Unknown command: {cmd}")
            print("[SYS] Type /help for available commands.")
    
    def _print_help(self):
        """Print help information."""
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                         CRC ASSISTANT - HELP                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  MODES:                                                              ║
║    /voice    - Voice control mode (speak to control)                 ║
║    /text     - Text control mode (type commands)                     ║
║    /keyboard - Keyboard control mode (use keys directly)             ║
║                                                                      ║
║  COMMANDS:                                                           ║
║    /reset    - Reset conversation history                            ║
║    /pause    - Pause voice listening                                 ║
║    /resume   - Resume voice listening                                ║
║    /status   - Show current robot status                             ║
║    /help     - Show this help                                        ║
║    /quit     - Exit the program                                      ║
║                                                                      ║
║  KEYBOARD MODE CONTROLS:                                             ║
║    Arrow keys    - Move end-effector (left/right/up/down)            ║
║    a/d           - Rotate base left/right                            ║
║    t/g           - Wrist roll left/right                             ║
║    y/h           - Gripper close/open                                ║
║    r/f           - Pitch up/down                                     ║
║    0-9           - Select memory slot                                ║
║    o             - Save current pose to slot                         ║
║    i             - Go to saved pose in slot                          ║
║    p             - Toggle YOLO tracking                              ║
║    ESC           - Return to voice mode                              ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    def _start_keyboard_mode(self):
        """Start keyboard control mode."""
        if self.shared_state is None:
            print("[SYS] Keyboard mode not available (no shared state).")
            return
        
        self.mode = "keyboard"
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║                      KEYBOARD CONTROL MODE                           ║")
        print("╠══════════════════════════════════════════════════════════════════════╣")
        print("║  Arrow keys: Move EE (left/right/up/down)                            ║")
        print("║  a/d: Base rotation    t/g: Wrist roll    y/h: Gripper               ║")
        print("║  r/f: Pitch            0-9: Select slot   o: Save   i: Go to         ║")
        print("║  p: Toggle tracking    ESC: Return to voice mode                     ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print("[KEYBOARD] Use keyboard to control. Press ESC to return to voice mode.")
        
        # Start keyboard handling thread
        self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.keyboard_thread.start()
    
    def _stop_keyboard_mode(self):
        """Stop keyboard control mode."""
        if self.keyboard_thread and self.keyboard_thread.is_alive():
            # The thread will exit when mode changes
            pass
    
    def _keyboard_loop(self):
        """Handle keyboard input in keyboard mode."""
        try:
            from pynput import keyboard
            
            # Key state tracking
            pressed_keys = set()
            last_action_time = {}
            action_interval = 0.05  # 50ms between repeated actions
            
            def on_press(key):
                if self.mode != "keyboard":
                    return False  # Stop listener
                
                try:
                    k = key.char if hasattr(key, 'char') and key.char else str(key)
                except:
                    k = str(key)
                
                pressed_keys.add(k)
                
                # ESC to exit keyboard mode
                if k in ['Key.esc', 'esc', 'ESC']:
                    self.mode = "voice"
                    print("[SYS] Returning to voice mode...")
                    return False
                
                return True
            
            def on_release(key):
                if self.mode != "keyboard":
                    return False
                
                try:
                    k = key.char if hasattr(key, 'char') and key.char else str(key)
                except:
                    k = str(key)
                
                pressed_keys.discard(k)
                return True
            
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            
            # Movement parameters
            ee_step = 0.008  # 8mm per action
            rotation_step = 3.0  # 3 degrees per action
            
            while self.mode == "keyboard" and not self.stop_event.is_set():
                current_time = time.time()
                
                for k in list(pressed_keys):
                    # Check if enough time has passed for this key
                    if k in last_action_time and current_time - last_action_time[k] < action_interval:
                        continue
                    
                    action_performed = False
                    
                    # Arrow keys for EE movement
                    if k in ['Key.left', 'left', 's']:
                        self.agent.robot.move_ee(axis="x", direction="negative", step_size=ee_step)
                        action_performed = True
                    elif k in ['Key.right', 'right', 'w']:
                        self.agent.robot.move_ee(axis="x", direction="positive", step_size=ee_step)
                        action_performed = True
                    elif k in ['Key.up', 'up', 'q']:
                        self.agent.robot.move_ee(axis="y", direction="positive", step_size=ee_step)
                        action_performed = True
                    elif k in ['Key.down', 'down', 'e']:
                        self.agent.robot.move_ee(axis="y", direction="negative", step_size=ee_step)
                        action_performed = True
                    
                    # Base rotation
                    elif k == 'a':
                        self.agent.robot.head_turn(direction="left", step_size=rotation_step)
                        action_performed = True
                    elif k == 'd':
                        self.agent.robot.head_turn(direction="right", step_size=rotation_step)
                        action_performed = True
                    
                    # Wrist roll
                    elif k == 't':
                        self.agent.robot.wrist_roll(direction="left", step_size=rotation_step)
                        action_performed = True
                    elif k == 'g':
                        self.agent.robot.wrist_roll(direction="right", step_size=rotation_step)
                        action_performed = True
                    
                    # Gripper
                    elif k == 'y':
                        self.agent.robot.gripper(action="close")
                        action_performed = True
                        pressed_keys.discard(k)  # One-shot
                    elif k == 'h':
                        self.agent.robot.gripper(action="open")
                        action_performed = True
                        pressed_keys.discard(k)  # One-shot
                    
                    # Pitch
                    elif k == 'r':
                        self.agent.robot.adjust_pitch(direction="up", step_size=rotation_step)
                        action_performed = True
                    elif k == 'f':
                        self.agent.robot.adjust_pitch(direction="down", step_size=rotation_step)
                        action_performed = True
                    
                    # Toggle tracking
                    elif k == 'p':
                        self.agent.robot.toggle_tracking()
                        print("[KEYBOARD] Tracking toggled")
                        action_performed = True
                        pressed_keys.discard(k)  # One-shot
                    
                    # Slot selection (0-9)
                    elif len(k) == 1 and k.isdigit():
                        with self.shared_state.lock:
                            self.shared_state.selected_slot = k
                        print(f"[KEYBOARD] Selected slot: {k}")
                        pressed_keys.discard(k)  # One-shot
                    
                    # Save pose
                    elif k == 'o':
                        if hasattr(self.shared_state, 'selected_slot'):
                            slot = self.shared_state.selected_slot
                            result = self.agent.robot.save_pose(slot=slot)
                            print(f"[KEYBOARD] {result.get('message', 'Pose saved')}")
                        pressed_keys.discard(k)  # One-shot
                    
                    # Go to pose
                    elif k == 'i':
                        if hasattr(self.shared_state, 'selected_slot'):
                            slot = self.shared_state.selected_slot
                            result = self.agent.robot.goto_pose(slot=slot)
                            print(f"[KEYBOARD] {result.get('message', 'Moving to pose')}")
                        pressed_keys.discard(k)  # One-shot
                    
                    if action_performed:
                        last_action_time[k] = current_time
                
                time.sleep(0.01)
            
            listener.stop()
            
        except ImportError:
            print("[SYS] Keyboard mode requires 'pynput' package. Install with: pip install pynput")
            self.mode = "voice"
        except Exception as e:
            print(f"[SYS] Keyboard mode error: {e}")
            self.mode = "voice"
    
    def _process_text_input(self, text: str):
        """Process text input in text mode."""
        print(f"[YOU] {text}")
        self._process_and_respond(text)
    
    def _process_and_respond(self, user_text: str):
        """Process user input and generate response."""
        if self.on_processing:
            self.on_processing()
        
        answer, commands = self.agent.agent_step(user_text)
        print(f"[CRC ASSISTANT] {answer}")
        
        if self.on_speaking:
            self.on_speaking()
        
        # Generate and play TTS response
        wav_out = self.tmp_dir / "out.wav"
        self.agent.tts_to_wav(answer, wav_out)
        play_wav(wav_out)
        
        if self.on_idle:
            self.on_idle()
    
    def start(self):
        """Start the voice loop in a background thread."""
        if self.thread is not None and self.thread.is_alive():
            print("[VOICE] Already running")
            return
        
        self.stop_event.clear()
        self.paused.clear()
        self.thread = threading.Thread(target=self._voice_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the voice loop."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def pause(self):
        """Pause the voice loop."""
        self.paused.set()
    
    def resume(self):
        """Resume the voice loop."""
        self.paused.clear()
    
    def send_text(self, text: str):
        """Send text to be processed (for text mode or commands)."""
        self.cmd_queue.put(text)
    
    def is_running(self) -> bool:
        """Check if the voice loop is running."""
        return self.thread is not None and self.thread.is_alive()
