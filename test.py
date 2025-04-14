import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
from moviepy import VideoFileClip, AudioFileClip
import sys
from pathlib import Path
from datetime import datetime
import os
from ttspeech import text_to_speech, merge_audio_chunks_with_timestamps, speed_up_audio_to_match_video
from translation import query_ollama_for_translation, unload_ollama_model

# from google import genai
# from google.genai import types
from dotenv import load_dotenv
import os

# load_dotenv()

# client = genai.Client(api_key=os.getenv('GENAI_API_KEY'))

# def translationFunction(language, text):
#     response = client.models.generate_content(
#     model='gemini-2.0-flash-lite', contents=f"can you translate this text to {language} and only give the transaltion: {text} "
#     )

#     return response.text



def extract_audio(video_path):
    """
    Extract audio from a video file and save it as an MP3.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        str: Path to the extracted audio file
    """
    print(f"Extracting audio from: {video_path}")
    filename = Path(video_path).stem  
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs("audio_samples", exist_ok=True)
    
    unique_filename = f"{filename}_{current_time}.mp3"
    output_audio_path = f"audio_samples/{unique_filename}"
    
    try:
        with VideoFileClip(video_path) as video:    
            audio = video.audio
            audio.write_audiofile(output_audio_path)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        sys.exit(1)
        
    return output_audio_path


def initialize_model():
    """Initialize and return the speech recognition pipeline."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=27,
            stride_length_s=5,
            return_timestamps=True,
            device=device
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)





def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <video_file_path>")
        sys.exit(1)
        
    start_time = time.time()
    
    # Extract audio from video
    audio_path = extract_audio(sys.argv[1])
    
    # Initialize the model
    pipe = initialize_model()
    
    # Perform transcription
    try:
        result = pipe(f"./{audio_path}")
        
        # Save transcription to file
        with open('test.txt', 'w', encoding='utf-8') as f:
            f.write(result['text']) 
            
        transcribed_chunk = result['chunks']

        print(transcribed_chunk) # <-- Original english transcription

        print('\x1b[6;30;42m' + f"Translation has started. It has {len(transcribed_chunk)} chunks" + '\x1b[0m')

        translated_chunk = [
            {
                'timestamp': entry['timestamp'],
                'text': query_ollama_for_translation(language="hindi", text=entry['text'])  
            }
            for entry in transcribed_chunk
        ]
        
        print(translated_chunk) # <-- Translated hindi transcription

        unload_ollama_model(model="gemma3")     # <-- Free up gpu and RAM after translation

        text_to_speech(translated_chunk)
        merge_audio_chunks_with_timestamps(translated_chunk)
        speed_up_audio_to_match_video(audio_path="./final_output.wav", video_path=sys.argv[1])


        import subprocess
        subprocess.call([
            'ffmpeg',
            '-i', sys.argv[1],
            '-i', 'new_audio.wav',
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'copy',
            '-shortest',
            'dubbedVideo.mp4'
        ])
                
        print("\033[92m Video dubbing Successful \033[0m")

        print(f"--- {time.time() - start_time:.2f} seconds ---")
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
