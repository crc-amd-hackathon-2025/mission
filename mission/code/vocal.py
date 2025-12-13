#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Helpers config
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
# Robot "backend" (placeholder)
# -----------------------------

@dataclass
class RobotCommand:
    name: str
    args: Dict[str, Any]

class RobotConsole:
    def move(self, axis: str, direction: str, amount: str):
        print(f"[ROBOT] MOVE axis={axis} direction={direction} amount={amount}")

    def gripper(self, action: str):
        print(f"[ROBOT] GRIPPER action={action}")

    def head_turn(self, direction: str, amount: str):
        print(f"[ROBOT] HEAD direction={direction} amount={amount}")

    def start_tracking(self, object_name: str):
        print(f"[ROBOT] TRACKING start object={object_name}")

    def start_task(self, task_name: str):
        print(f"[ROBOT] TASK start name={task_name}")


# -----------------------------
# Audio: VAD with absolute thresholds + hysteresis
# -----------------------------

def _rms_db(x: np.ndarray, eps: float = 1e-12) -> float:
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
) -> Path:
    """
    VAD simple et robuste:
    - Début si db >= start_db pendant N frames consécutives
    - Fin si db <= end_db pendant end_silence_ms
    - start_db > end_db (hystérésis) recommandé
    """
    if start_db <= end_db and verbose:
        print("[WARN] VAD_START_DB devrait être > VAD_END_DB (hystérésis).")

    frame_len = int(sample_rate * frame_ms / 1000)
    preroll_frames = max(0, int(preroll_ms / frame_ms))
    end_silence_frames = max(1, int(end_silence_ms / frame_ms))
    max_frames = int(max_record_s * 1000 / frame_ms)

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    if verbose:
        print(f"[AUDIO] VAD abs: start_db={start_db:.1f} end_db={end_db:.1f} "
              f"req_start_frames={required_start_frames} end_silence_ms={end_silence_ms}")

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
            try:
                chunk = q.get(timeout=1.0)
            except queue.Empty:
                continue

            db = _rms_db(chunk)

            # pré-roll
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
                        print(f"[AUDIO] speech start (db={db:.1f})")
                continue

            # speech en cours
            frames_buf.append(chunk)

            if db <= end_db:
                silence_count += 1
            else:
                silence_count = 0

            if silence_count >= end_silence_frames:
                if verbose:
                    print(f"[AUDIO] speech end (db={db:.1f})")
                break

    if not frames_buf:
        if verbose:
            print("[AUDIO] Rien capturé.")
        sf.write(str(out_path), np.zeros((0, 1), dtype="float32"), sample_rate, subtype="PCM_16")
        return out_path

    audio = np.concatenate(frames_buf, axis=0)
    sf.write(str(out_path), audio, sample_rate, subtype="PCM_16")
    if verbose:
        dur = audio.shape[0] / sample_rate
        print(f"[AUDIO] Sauvé: {out_path} ({dur:.2f}s)")
    return out_path


def play_wav(path: Path):
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
# OpenAI: STT, agent (tools), TTS
# -----------------------------

class VoiceRobotAgent:
    def __init__(
        self,
        robot: RobotConsole,
        llm_model: str,
        stt_model: str,
        tts_model: str,
        tts_voice: str,
    ):
        self.client = OpenAI()
        self.robot = robot
        self.llm_model = llm_model
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice

        self.tools = self._build_tools()
        self.input_list: List[Dict[str, Any]] = []
        self._seed_system()

    def _seed_system(self):
        self.input_list = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant vocal qui contrôle un bras robot.\n"
                    "Tu peux discuter de tout normalement.\n"
                    "Quand l'utilisateur demande une action physique du robot, appelle un tool.\n"
                    "Si c'est ambigu, pose UNE question courte.\n"
                    "Conventions:\n"
                    "- amount: 'un_peu' ou 'beaucoup'\n"
                    "- axes EE: x=gauche/droite, y=haut/bas, z=avant/arriere\n"
                    "- direction axe: positive/negative\n"
                    "- tête: left/right\n"
                    "- pince: open/close\n"
                    "- tâche: start_task('grab_camera')\n"
                    "Réponses courtes."
                ),
            }
        ]

    def _build_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "move_ee",
                "description": "Déplace l'effecteur (EE) du bras sur un axe.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "axis": {"type": "string", "enum": ["x", "y", "z"]},
                        "direction": {"type": "string", "enum": ["positive", "negative"]},
                        "amount": {"type": "string", "enum": ["un_peu", "beaucoup"]},
                    },
                    "required": ["axis", "direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "gripper",
                "description": "Ouvre ou ferme la pince.",
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string", "enum": ["open", "close"]}},
                    "required": ["action"],
                },
            },
            {
                "type": "function",
                "name": "head_turn",
                "description": "Tourne la tête/caméra à gauche ou à droite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["left", "right"]},
                        "amount": {"type": "string", "enum": ["un_peu", "beaucoup"]},
                    },
                    "required": ["direction", "amount"],
                },
            },
            {
                "type": "function",
                "name": "start_tracking",
                "description": "Lance le tracking d'un objet nommé.",
                "parameters": {
                    "type": "object",
                    "properties": {"object_name": {"type": "string"}},
                    "required": ["object_name"],
                },
            },
            {
                "type": "function",
                "name": "start_task",
                "description": "Lance une tâche high-level (ex: grab_camera).",
                "parameters": {
                    "type": "object",
                    "properties": {"task_name": {"type": "string"}},
                    "required": ["task_name"],
                },
            },
        ]

    def stt(self, wav_path: Path) -> str:
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
        args = json.loads(arguments_json) if arguments_json else {}

        if name == "move_ee":
            self.robot.move(axis=args["axis"], direction=args["direction"], amount=args["amount"])
            return {"ok": True}
        if name == "gripper":
            self.robot.gripper(action=args["action"])
            return {"ok": True}
        if name == "head_turn":
            self.robot.head_turn(direction=args["direction"], amount=args["amount"])
            return {"ok": True}
        if name == "start_tracking":
            self.robot.start_tracking(object_name=args["object_name"])
            return {"ok": True}
        if name == "start_task":
            self.robot.start_task(task_name=args["task_name"])
            return {"ok": True}

        return {"ok": False, "error": f"Unknown tool: {name}"}

    def agent_step(self, user_text: str) -> Tuple[str, List[RobotCommand]]:
        robot_cmds: List[RobotCommand] = []
        self.input_list.append({"role": "user", "content": user_text})

        resp = self.client.responses.create(
            model=self.llm_model,
            tools=self.tools,
            input=self.input_list,
        )
        self.input_list += resp.output

        for item in resp.output:
            if getattr(item, "type", None) == "function_call":
                tool_name = item.name
                tool_args = item.arguments
                robot_cmds.append(RobotCommand(name=tool_name, args=json.loads(tool_args)))

                tool_result = self._dispatch_tool(tool_name, tool_args)
                self.input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(tool_result),
                    }
                )

        resp2 = self.client.responses.create(
            model=self.llm_model,
            tools=self.tools,
            input=self.input_list,
        )
        self.input_list += resp2.output
        return resp2.output_text.strip(), robot_cmds


# -----------------------------
# Stdin commands while voice loop runs
# -----------------------------

def start_stdin_reader(cmd_q: "queue.Queue[str]") -> threading.Thread:
    def _reader():
        while True:
            try:
                line = input().strip()
            except EOFError:
                return
            cmd_q.put(line)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


def main():
    load_dotenv()  # charge OPENAI_API_KEY + configs

    # Config
    sample_rate = env_int("AUDIO_SAMPLE_RATE", 16000)

    vad_start_db = env_float("VAD_START_DB", -19.0)
    vad_end_db = env_float("VAD_END_DB", -24.0)
    vad_frame_ms = env_int("VAD_FRAME_MS", 20)
    vad_preroll_ms = env_int("VAD_PREROLL_MS", 250)
    vad_end_silence_ms = env_int("VAD_END_SILENCE_MS", 600)
    vad_max_record_s = env_float("VAD_MAX_RECORD_S", 15.0)
    vad_required_start_frames = env_int("VAD_REQUIRED_START_FRAMES", 3)

    llm_model = env_str("LLM_MODEL", "gpt-4.1")
    stt_model = env_str("STT_MODEL", "gpt-4o-transcribe")
    tts_model = env_str("TTS_MODEL", "gpt-4o-mini-tts")
    tts_voice = env_str("TTS_VOICE", "coral")

    robot = RobotConsole()
    agent = VoiceRobotAgent(
        robot=robot,
        llm_model=llm_model,
        stt_model=stt_model,
        tts_model=tts_model,
        tts_voice=tts_voice,
    )

    print("Agent vocal robot prêt.")
    print("Tape /text, /voice, /reset, /quit (tu peux les taper même en mode voice).")
    print(f"[CFG] VAD start={vad_start_db} dBFS end={vad_end_db} dBFS")

    mode = "voice"
    cmd_q: "queue.Queue[str]" = queue.Queue()
    start_stdin_reader(cmd_q)

    tmp_dir = Path("./_tmp_audio")
    tmp_dir.mkdir(exist_ok=True)

    while True:
        # Traite les commandes / messages texte
        while not cmd_q.empty():
            cmd = cmd_q.get_nowait().strip()
            if not cmd:
                continue
            if cmd == "/quit":
                return
            if cmd == "/reset":
                agent._seed_system()
                print("[SYS] Contexte reset.")
                continue
            if cmd == "/text":
                mode = "text"
                print("[SYS] Mode text.")
                continue
            if cmd == "/voice":
                mode = "voice"
                print("[SYS] Mode voice.")
                continue

            if mode == "text":
                user_text = cmd
                answer, _ = agent.agent_step(user_text)
                print(f"[AGENT] {answer}")
                wav_out = tmp_dir / "out.wav"
                agent.tts_to_wav(answer, wav_out)
                play_wav(wav_out)

        if mode == "text":
            sd.sleep(50)
            continue

        # Mode voice: écoute direct (pas besoin d'appuyer Entrée)
        wav_in = tmp_dir / "in.wav"
        record_wav_vad_absolute(
            wav_in,
            sample_rate=sample_rate,
            frame_ms=vad_frame_ms,
            start_db=vad_start_db,
            end_db=vad_end_db,
            required_start_frames=vad_required_start_frames,
            end_silence_ms=vad_end_silence_ms,
            max_record_s=vad_max_record_s,
            preroll_ms=vad_preroll_ms,
            verbose=True,
        )

        user_text = agent.stt(wav_in)
        if not user_text:
            continue

        print(f"[YOU] {user_text}")
        answer, _ = agent.agent_step(user_text)
        print(f"[AGENT] {answer}")

        wav_out = tmp_dir / "out.wav"
        agent.tts_to_wav(answer, wav_out)
        play_wav(wav_out)


if __name__ == "__main__":
    main()