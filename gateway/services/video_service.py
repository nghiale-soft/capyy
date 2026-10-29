from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import edge_tts
import numpy as np

logger = logging.getLogger("gateway.services.video")

DEFAULT_VOICE_FEMALE = "vi-VN-HoaiMyNeural"
DEFAULT_VOICE_MALE = "vi-VN-NamMinhNeural"


class VideoTranslationService:
    def __init__(self, gateway_service: Any) -> None:
        self.gateway = gateway_service

    async def translate_video(
        self,
        video_bytes: bytes,
        *,
        target_language: str = "Vietnamese",
        voice: str = "female",  # 'female' | 'male' or exact voice name
        instruction: str | None = None,
        model: str | None = None,
        cover_old_sub: bool = True,
    ) -> tuple[bytes, list[dict[str, Any]]]:
        """Transcribe Chinese audio from video, translate to Vietnamese, generate TTS, and burn subtitle."""
        chosen_voice = DEFAULT_VOICE_FEMALE if voice == "female" else (
            DEFAULT_VOICE_MALE if voice == "male" else voice
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            in_video = tmp_path / "input_video.mp4"
            extracted_audio = tmp_path / "extracted_audio.mp3"
            out_video = tmp_path / "output_dubbed.mp4"
            srt_file = tmp_path / "subtitles.srt"

            in_video.write_bytes(video_bytes)

            # 1. Extract audio from input video
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(in_video), "-vn", "-ar", "24000", "-ac", "1", "-b:a", "128k", str(extracted_audio)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Probe video duration and dimensions
            probe_vid = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height", "-of", "json", str(in_video)],
                capture_output=True,
                text=True,
                check=True,
            )
            probe_data = json.loads(probe_vid.stdout or "{}")
            vid_duration = float(probe_data.get("format", {}).get("duration", 0.0) or 0.0)
            vid_w = 720
            vid_h = 1280
            for s in probe_data.get("streams", []):
                if s.get("width") and s.get("height"):
                    vid_w = int(s["width"])
                    vid_h = int(s["height"])
                    break

            # 2. Transcribe & translate segments via Gemini Audio Multimodal
            audio_b64 = base64.b64encode(extracted_audio.read_bytes()).decode("utf-8")

            user_inst = f"\nUSER INSTRUCTION:\n{instruction.strip()}\n" if instruction and instruction.strip() else ""

            prompt = (
                f"Listen to this audio track from an e-commerce product video.\n"
                f"{user_inst}"
                f"Your tasks:\n"
                f"1. Transcribe each spoken segment with start_ms and end_ms timestamps (in milliseconds).\n"
                f"2. Translate each segment into natural, punchy, persuasive {target_language} for an e-commerce product video.\n"
                f"Keep translations concise so they fit the original spoken duration.\n"
                f"Return ONLY valid JSON in this exact schema:\n"
                f'{{"segments": [{{"start_ms": 0, "end_ms": 2500, "original_text": "...", "translated_text": "..."}}]}}'
            )

            chosen_provider, real_model = self.gateway.resolve(model)
            if not real_model and not model:
                real_model = "gemini-3.5-flash-lite"

            chat_payload = {
                "model": real_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": "mp3"},
                            },
                        ],
                    }
                ],
                "temperature": 0.1,
            }

            chat_resp = await self.gateway.chat(chosen_provider, chat_payload, real_model=real_model)
            content = ""
            choices = chat_resp.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

            segments = self._parse_segments(content)
            logger.info("Transcribed & translated video segments: %d", len(segments))

            if not segments:
                logger.warning("No speech segments detected, returning original video.")
                return video_bytes, []

            # 3. Generate Vietnamese TTS for each segment with atempo time-stretching & precise timestamp alignment
            # We synthesize full 24000Hz mono PCM directly into a master timeline array.
            # This completely avoids FFmpeg amix attenuation bugs (which divide volume by input count).
            sr = 24000
            total_samples = max(sr, int(vid_duration * sr) + sr)
            master_pcm = np.zeros(total_samples, dtype=np.int16)

            srt_lines = []
            srt_idx = 1

            for seg in segments:
                start_ms = int(seg.get("start_ms", 0))
                end_ms = int(seg.get("end_ms", 0))
                target_sec = max(0.4, (end_ms - start_ms) / 1000.0)
                trans_text = seg.get("translated_text", "").strip()

                # Clean punctuation-only or empty strings that cause TTS to fail
                clean_for_speech = re.sub(r"[^\w\s\d]", "", trans_text).strip()
                if not clean_for_speech:
                    continue

                raw_seg_mp3 = tmp_path / f"raw_seg_{srt_idx}.mp3"
                raw_seg_wav = tmp_path / f"raw_seg_{srt_idx}.wav"
                synced_seg_wav = tmp_path / f"synced_seg_{srt_idx}.wav"

                # Generate TTS with Edge-TTS (with fallback on retry)
                try:
                    communicate = edge_tts.Communicate(trans_text, chosen_voice)
                    await communicate.save(str(raw_seg_mp3))
                except Exception as tts_err:
                    logger.warning("TTS failed for '%s': %s", trans_text, tts_err)
                    continue

                if not raw_seg_mp3.exists() or raw_seg_mp3.stat().st_size == 0:
                    continue

                # Convert to PCM 16-bit 24000Hz mono wav
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(raw_seg_mp3), "-ar", str(sr), "-ac", "1", "-f", "wav", str(raw_seg_wav)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Measure actual generated duration
                p = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(raw_seg_wav)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                actual_sec = float(p.stdout.strip() or target_sec)

                # Time stretch via atempo (safe bounds 0.85x to 1.35x) to fit target slot
                factor = max(0.85, min(1.35, actual_sec / target_sec))
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(raw_seg_wav), "-filter:a", f"atempo={factor:.3f}", "-ar", str(sr), "-ac", "1", "-f", "wav", str(synced_seg_wav)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Read actual synced PCM samples
                with wave.open(str(synced_seg_wav), "rb") as wf:
                    n_frames = wf.getnframes()
                    pcm_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)

                synced_dur_ms = int(len(pcm_data) / sr * 1000)
                actual_end_ms = start_ms + synced_dur_ms

                # Place onto master audio timeline
                start_sample = max(0, int(start_ms * sr / 1000))
                end_sample = min(total_samples, start_sample + len(pcm_data))
                avail_len = end_sample - start_sample
                if avail_len > 0:
                    master_pcm[start_sample:end_sample] = np.clip(
                        master_pcm[start_sample:end_sample].astype(np.int32) + pcm_data[:avail_len].astype(np.int32),
                        -32768,
                        32767,
                    ).astype(np.int16)

                # Synchronize Subtitle timeline EXACTLY with the dubbed audio utterance
                s_str = self._format_srt_time(start_ms)
                e_str = self._format_srt_time(actual_end_ms)
                srt_lines.append(f"{srt_idx}\n{s_str} --> {e_str}\n{trans_text}\n")
                srt_idx += 1

            srt_file.write_text("\n".join(srt_lines), encoding="utf-8")

            # 4. Save master dubbed audio track
            combined_dubbed_wav = tmp_path / "combined_dubbed.wav"
            with wave.open(str(combined_dubbed_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(master_pcm.tobytes())

            # 6. Burn subtitle directly over the delogo/blurred bottom subtitle zone
            # Most Chinese e-commerce videos have hardcoded subtitles on the bottom 10-12% of the screen.
            # Using delogo cleanly obscures the original Chinese characters, and placing the new Vietnamese
            # subtitle inside that zone creates a seamless, professional dubbing result.
            sub_y = int(vid_h * 0.88)
            sub_box_h = int(vid_h * 0.11)
            delogo_w = int(vid_w * 0.96)
            delogo_x = int((vid_w - delogo_w) / 2)

            # CapCut-style prominent subtitle: yellow primary color & black outline
            font_size = max(13, int(vid_w * 0.024))
            sub_style = (
                f"FontName=DejaVuSans,FontSize={font_size},PrimaryColour=&H0000FFFF,"
                "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,"
                "Alignment=2,MarginV=18"
            )
            if cover_old_sub:
                vf_filter = (
                    f"delogo=x={delogo_x}:y={sub_y}:w={delogo_w}:h={sub_box_h},"
                    f"subtitles={str(srt_file)}:force_style='{sub_style}'"
                )
            else:
                vf_filter = f"subtitles={str(srt_file)}:force_style='{sub_style}'"

            # Mix original video audio (ducked to 5% so Chinese speech is virtually silent)
            # and Vietnamese dubbed audio boosted to 1.8x with volume normalization
            audio_filter = "[0:a]volume=0.05[a_bg];[1:a]volume=1.8[a_voice];[a_bg][a_voice]amix=inputs=2:duration=first:normalize=0[afinal]"

            final_cmd = [
                "ffmpeg", "-y",
                "-i", str(in_video),
                "-i", str(combined_dubbed_wav),
                "-filter_complex", f"[0:v]{vf_filter}[v];{audio_filter}",
                "-map", "[v]",
                "-map", "[afinal]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", "128k",
                "-t", f"{vid_duration:.2f}",
                str(out_video),
            ]

            subprocess.run(final_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_video.read_bytes(), segments

    def _parse_segments(self, content: str) -> list[dict[str, Any]]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "segments" in data:
                return data["segments"]
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    @staticmethod
    def _format_srt_time(ms: int) -> str:
        hours = ms // 3600000
        ms %= 3600000
        minutes = ms // 60000
        ms %= 60000
        seconds = ms // 1000
        millis = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
