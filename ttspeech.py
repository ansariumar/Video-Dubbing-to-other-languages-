import torch
from TTS.api import TTS
import os
import time
from typing import TypedDict
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip
import os


class TranslationEntry(TypedDict):
    timestamp: tuple[float, float]
    text: str


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)


output_dir = "tts_outputs"
os.makedirs(output_dir, exist_ok=True)

def text_to_speech(translated_chunk: list[TranslationEntry], speaker_wav:str="./audio_samples/pointerVoice.wav", language:str="hi") -> None:

    print("\033[96m Converting text to speech... \033[0m")

    for i, chunk in enumerate(translated_chunk):
        out_path = os.path.join(output_dir, f"chunk_{i}.wav")
    
        print(f"Generating audio for chunk {i}...", end="")
        tts.tts_to_file(
            text=chunk['text'],
            speaker_wav=speaker_wav,
            language=language,
            file_path=out_path,
        )

    
    print("\033[92m Text to speech completed \033[0m")

def merge_audio_chunks_with_timestamps(translated_chunk: list[TranslationEntry], chunk_folder: str = "tts_outputs", output_path: str = "final_output.wav"):
    print("\033[96m Merging all the Audio chunks \033[0m")

    final_audio = AudioSegment.empty()

    for i, chunk in enumerate(translated_chunk):

        # Load the chunk audio
        chunk_path = os.path.join(chunk_folder, f"chunk_{i}.wav")
        chunk_audio = AudioSegment.from_wav(chunk_path)

        # Append to final audio
        final_audio += chunk_audio

    # Export the final merged audio
    final_audio.export(output_path, format="wav")
    print("\033[92m Merging complete \033[0m")
    print(f"\033[92m Final audio saved to: {output_path} \033[0m")



def speed_up_audio_to_match_video(audio_path: str, video_path: str, output_path: str = "speedup_audio.wav"):
    print("\033[96m synchronizing Audio with Video \033[0m")

    audio = AudioSegment.from_wav(audio_path)
    video_duration_sec = VideoFileClip(video_path).duration
    original_duration_sec = len(audio) / 1000

    # Calculate speed factor
    speed_factor = original_duration_sec / video_duration_sec
    if speed_factor <= 1.0:
        print("No need to speed up, audio fits within video.")
        audio.export(output_path, format="wav")
        return

    print(f"Speeding up audio by factor of {speed_factor:.2f}")

    # Use ffmpeg filter for time compression
    import subprocess
    subprocess.call([
        "ffmpeg", "-y",
        "-i", audio_path,
        "-filter:a", f"atempo={speed_factor:.5f}",
        output_path
    ])
    print("\033[92m Synchronization complete \033[0m")


if __name__ == "__main__":
    text_to_speech()