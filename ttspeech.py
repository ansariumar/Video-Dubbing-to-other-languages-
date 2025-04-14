import torch
from TTS.api import TTS
import os
import time
from pydub import AudioSegment
from typing import TypedDict


class TranslationEntry(TypedDict):
    timestamp: tuple[float, float]
    text: str


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)


output_dir = "tts_outputs"
os.makedirs(output_dir, exist_ok=True)

def text_to_speech(translated_chunk: list[TranslationEntry], speaker_wav:str="./audio_samples/modi.wav", language:str="hi") -> None:
    
    for i, chunk in enumerate(translated_chunk):
        out_path = os.path.join(output_dir, f"chunk_{i}.wav")
    
        print(f"Generating audio for chunk {i}...", end="")
        tts.tts_to_file(
            text=chunk['text'],
            speaker_wav=speaker_wav,
            language=language,
            file_path=out_path,
        )

def merge_audio_chunks_with_timestamps(translated_chunk: list[TranslationEntry], chunk_folder: str = "tts_outputs", output_path: str = "final_output.wav"):
    final_audio = AudioSegment.empty()

    for i, chunk in enumerate(translated_chunk):
        start_time_sec = chunk["timestamp"][0]
        if i == 0:
            prev_end_time_sec = 0.0
        else:
            prev_end_time_sec = translated_chunk[i - 1]["timestamp"][1]

        # Duration of silence between chunks
        silence_duration_ms = int((start_time_sec - prev_end_time_sec) * 1000)
        if silence_duration_ms > 0:
            final_audio += AudioSegment.silent(duration=silence_duration_ms)

        # Load the chunk audio
        chunk_path = os.path.join(chunk_folder, f"chunk_{i}.wav")
        chunk_audio = AudioSegment.from_wav(chunk_path)

        # Append to final audio
        final_audio += chunk_audio

    # Export the final merged audio
    final_audio.export(output_path, format="wav")
    print(f"Final audio saved to: {output_path}")

    

# chunk = [{'timestamp': (0.0, 4.0), 'text': 'इस व्याख्यान में हम एक नई विषयवस्तु शुरू करेंगे और विषय है'}, {'timestamp': (4.0, 12.0), 'text': 'नियमित अभिव्यक्तियाँ। तो, अब तक हमने कई भाषाओं और कुछ प्रकार के स्ट्रिंग्स को देखा है।'}, {'timestamp': (12.0, 19.0), 'text': 'स्वीकृत भाषाओं द्वारा। और अब तक, हमने उन स्ट्रिंग्स को दर्शाने का तरीका बस सरल का उपयोग करके था।'}, {'timestamp': (19.2, 24.0), 'text': 'अंग्रेजी भाषा। अब, नियमित अभिव्यक्तियाँ क्या हैं? नियमित अभिव्यक्तियाँ।'}, {'timestamp': (24.0, 31.26), 'text': 'ये कुछ स्ट्रिंग सेट को गणितीय तरीके से दर्शाने के लिए इस्तेमाल किए जाते हैं। तो, इसके बजाय'}, {'timestamp': (31.26, 36.24), 'text': 'सरल तरीके से उन्हें दर्शाने के, हम अब उन्हें वैसे ही दर्शा सकते हैं जैसा हमने पहले तक किया था, ये नियमित।'}, {'timestamp': (36.24, 41.84), 'text': 'अभिव्यक्तियाँ इस तरह से उपयोग किए जाते हैं ताकि हमारे स्ट्रिंग को बीजगणितीय तरीके से दर्शाया जा सके।'}, {'timestamp': (41.84, 47.14), 'text': 'तो, नियमित अभिव्यक्ति के बारे में अधिक जानने से पहले, कुछ बातें या नियम हैं जो आपको…'}, {'timestamp': (47.14, 50.0), 'text': 'हमें याद रखना होगा और देखते हैं कि वे क्या हैं।'}, {'timestamp': (50.0, 55.0), 'text': 'ठीक है, तो ये नियम या बिंदु हैं जिन्हें हमें नियमित अभिव्यक्तियों के बारे में याद रखना चाहिए।'}, {'timestamp': (55.0, 58.0), 'text': 'तो, चलो पहले बिंदु को देखते हैं। कोई भी टर्मिनल प्रतीक,'}, {'timestamp': (58.0, 66.0), 'text': 'मैं. यानी सिग्मा से संबंधित प्रतीक, जिसमें खाली और शून्य प्रतीक भी शामिल हैं, नियमित अभिव्यक्ति हैं। इसलिए, तक'}, {'timestamp': (66.0, 75.0), 'text': 'अब हमने कई प्रतीकों जैसे A, B, C और इसी तरह के कई को देखा है जिनका हम इनपुट और आउटपुट को दर्शाने के लिए उपयोग करते हैं।'}, {'timestamp': (75.0, 78.0), 'text': 'तो, ये सभी प्रतीक, जिन्हें टर्मिनल प्रतीक के रूप में जाना जाता है।'}, {'timestamp': (78.0, 83.0), 'text': 'शामिल हैं खाली और शून्य प्रतीक भी सभी नियमित अभिव्यक्तियाँ हैं।'}, {'timestamp': (83.0, 88.82), 'text': 'तो, यही पहली बात कहती है। अब, आइए दूसरी बात देखते हैं। द यूनियन'}, {'timestamp': (88.82, 99.0), 'text': 'दो नियमित अभिव्यक्तियों का संयोजन भी एक नियमित अभिव्यक्ति है। तो, मान लीजिए कि हमारे पास दो नियमित अभिव्यक्तियाँ हैं जिन्हें हम R1 और R2 कहते हैं।'}, {'timestamp': (99.0, 101.0), 'text': 'ये दो नियमित अभिव्यक्तियाँ हैं।\n'}]
# text_to_speech(chunk)