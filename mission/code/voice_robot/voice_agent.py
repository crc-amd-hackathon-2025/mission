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

SYSTEM_PROMPT = """Tu es un assistant vocal qui contrôle un bras robot SO101.
Tu peux discuter de tout normalement, mais quand l'utilisateur demande une action physique du robot, tu dois appeler le tool approprié.

Le robot peut effectuer les actions suivantes:
- Déplacer l'effecteur (pince) selon les axes x (gauche/droite), y (haut/bas), z (avant/arrière)
- Tourner la tête/caméra à gauche ou à droite
- Ouvrir ou fermer la pince (gripper)
- Ajuster l'inclinaison du poignet (pitch) vers le haut ou le bas
- Tourner le poignet (wrist roll) à gauche ou à droite
- Activer le suivi visuel (tracking) d'un objet (ex: "cup", "person", "bottle")
- Désactiver le suivi visuel
- Lancer une tâche pré-entraînée (ex: "grab_camera")
- Sauvegarder la position actuelle dans un slot mémoire
- Aller à une position sauvegardée
- Obtenir le statut du robot
- Arrêter tous les mouvements

Conventions:
- amount: "un_peu" (petit mouvement) ou "beaucoup" (grand mouvement)
- axes EE: x=gauche/droite, y=haut/bas, z=avant/arrière
- direction: "positive" ou "negative" pour les axes, "left"/"right" pour rotations
- pince: "open" ou "close"

Si c'est ambigu, pose UNE question courte pour clarifier.
Réponds toujours en français et de manière concise.
Après chaque action, confirme brièvement ce que tu as fait."""


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
    ):
        """
        Initialize the voice agent.
        
        Args:
            robot_controller: The robot controller instance
            llm_model: OpenAI model for language understanding
            stt_model: OpenAI model for speech-to-text
            tts_model: OpenAI model for text-to-speech
            tts_voice: Voice for TTS
        """
        self.client = OpenAI()
        self.robot = robot_controller
        self.llm_model = llm_model
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        
        self.tools = self._build_tools()
        self.input_list: List[Dict[str, Any]] = []
        self._seed_system()
    
    def _seed_system(self):
        """Initialize the conversation with the system prompt."""
        self.input_list = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
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
                "description": "Déplace l'effecteur (EE) du bras robot sur un axe. x=gauche/droite, y=haut/bas, z=avant/arrière.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "axis": {
                            "type": "string",
                            "enum": ["x", "y", "z"],
                            "description": "Axe de déplacement"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                            "description": "Direction du mouvement"
                        },
                        "amount": {
                            "type": "string",
                            "enum": ["un_peu", "beaucoup"],
                            "description": "Amplitude du mouvement"
                        },
                    },
                    "required": ["axis", "direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "head_turn",
                "description": "Tourne la tête/caméra du robot à gauche ou à droite (rotation du shoulder_pan).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["left", "right"],
                            "description": "Direction de rotation"
                        },
                        "amount": {
                            "type": "string",
                            "enum": ["un_peu", "beaucoup"],
                            "description": "Amplitude de rotation"
                        },
                    },
                    "required": ["direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "gripper",
                "description": "Ouvre ou ferme la pince du robot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["open", "close"],
                            "description": "Action à effectuer"
                        }
                    },
                    "required": ["action"],
                },
            },
            {
                "type": "function",
                "name": "adjust_pitch",
                "description": "Ajuste l'inclinaison du poignet (pitch) vers le haut ou le bas.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down"],
                            "description": "Direction de l'inclinaison"
                        },
                        "amount": {
                            "type": "string",
                            "enum": ["un_peu", "beaucoup"],
                            "description": "Amplitude"
                        },
                    },
                    "required": ["direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "wrist_roll",
                "description": "Effectue une rotation du poignet (wrist roll) à gauche ou à droite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["left", "right"],
                            "description": "Direction de rotation"
                        },
                        "amount": {
                            "type": "string",
                            "enum": ["un_peu", "beaucoup"],
                            "description": "Amplitude"
                        },
                    },
                    "required": ["direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "start_tracking",
                "description": "Active le suivi visuel (YOLO tracking) d'un objet. Le robot suivra l'objet avec la caméra.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Nom de l'objet à suivre (ex: cup, person, bottle, cell phone)"
                        }
                    },
                    "required": ["object_name"],
                },
            },
            {
                "type": "function",
                "name": "stop_tracking",
                "description": "Désactive le suivi visuel. Le robot arrête de suivre l'objet.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "type": "function",
                "name": "start_task",
                "description": "Lance une tâche pré-entraînée par l'IA (policy inference). Ex: grab_camera pour attraper la caméra.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_name": {
                            "type": "string",
                            "description": "Nom de la tâche (ex: grab_camera)"
                        }
                    },
                    "required": ["task_name"],
                },
            },
            {
                "type": "function",
                "name": "save_pose",
                "description": "Sauvegarde la position actuelle du robot dans un slot mémoire (0-9 ou nom).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot": {
                            "type": "string",
                            "description": "Numéro ou nom du slot (ex: '1', 'home')"
                        }
                    },
                    "required": ["slot"],
                },
            },
            {
                "type": "function",
                "name": "goto_pose",
                "description": "Va à une position précédemment sauvegardée.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot": {
                            "type": "string",
                            "description": "Numéro ou nom du slot"
                        }
                    },
                    "required": ["slot"],
                },
            },
            {
                "type": "function",
                "name": "get_status",
                "description": "Obtient le statut actuel du robot (position, tracking, etc.).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "type": "function",
                "name": "stop",
                "description": "Arrête tous les mouvements du robot (urgence ou pause).",
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
            instructions="Voix naturelle, claire, en français.",
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
        
        # Map tool names to RobotController methods
        tool_handlers = {
            "move_ee": lambda: self.robot.move_ee(
                axis=args["axis"],
                direction=args["direction"],
                amount=args["amount"]
            ),
            "head_turn": lambda: self.robot.head_turn(
                direction=args["direction"],
                amount=args["amount"]
            ),
            "gripper": lambda: self.robot.gripper(action=args["action"]),
            "adjust_pitch": lambda: self.robot.adjust_pitch(
                direction=args["direction"],
                amount=args["amount"]
            ),
            "wrist_roll": lambda: self.robot.wrist_roll(
                direction=args["direction"],
                amount=args["amount"]
            ),
            "start_tracking": lambda: self.robot.start_tracking(
                object_name=args["object_name"]
            ),
            "stop_tracking": lambda: self.robot.stop_tracking(),
            "start_task": lambda: self.robot.start_task(task_name=args["task_name"]),
            "save_pose": lambda: self.robot.save_pose(slot=args["slot"]),
            "goto_pose": lambda: self.robot.goto_pose(slot=args["slot"]),
            "get_status": lambda: self.robot.get_status(),
            "stop": lambda: self.robot.stop(),
        }
        
        handler = tool_handlers.get(name)
        if handler:
            try:
                result = handler()
                print(f"[TOOL] {name} -> {result}")
                return result
            except Exception as e:
                print(f"[TOOL ERROR] {name}: {e}")
                return {"ok": False, "error": str(e)}
        
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
        self.mode = "voice"  # "voice" or "text"
        
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
        
        if cmd_lower == "/quit":
            self.stop()
        elif cmd_lower == "/reset":
            self.agent.reset_conversation()
            print("[SYS] Conversation reset.")
        elif cmd_lower == "/text":
            self.mode = "text"
            print("[SYS] Text mode activated.")
        elif cmd_lower == "/voice":
            self.mode = "voice"
            print("[SYS] Voice mode activated.")
        elif cmd_lower == "/pause":
            self.paused.set()
            print("[SYS] Voice loop paused.")
        elif cmd_lower == "/resume":
            self.paused.clear()
            print("[SYS] Voice loop resumed.")
        else:
            print(f"[SYS] Unknown command: {cmd}")
    
    def _process_text_input(self, text: str):
        """Process text input in text mode."""
        print(f"[YOU] {text}")
        self._process_and_respond(text)
    
    def _process_and_respond(self, user_text: str):
        """Process user input and generate response."""
        if self.on_processing:
            self.on_processing()
        
        answer, commands = self.agent.agent_step(user_text)
        print(f"[AGENT] {answer}")
        
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
