from flask import Flask, request, jsonify, render_template, send_file
import os
from werkzeug.utils import secure_filename
import time
from test import extract_audio, transcribe_audio
from subtitleGenerator import save_srt
from translation import query_ollama_for_translation, unload_ollama_model
from ttspeech import text_to_speech, merge_audio_chunks_with_timestamps, speed_up_audio_to_match_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    process_type = request.form.get('type', 'subtitle')  # 'subtitle' or 'dubbing'
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract audio and transcribe
        audio_path = extract_audio(filepath)
        result = transcribe_audio(audio_path)
        
        if process_type == 'subtitle':
            # Generate subtitles
            save_srt(result['chunks'])
            return jsonify({
                'message': 'Subtitles generated successfully',
                'subtitles': result['chunks']
            })
        else:
            # Perform dubbing
            translated_chunk = []
            for entry in result['chunks']:
                translated_text = query_ollama_for_translation(language="hindi", text=entry['text'])
                translated_chunk.append({
                    'timestamp': entry['timestamp'],
                    'text': translated_text
                })
            
            unload_ollama_model(model="gemma3")
            text_to_speech(translated_chunk)
            merge_audio_chunks_with_timestamps(translated_chunk)
            speed_up_audio_to_match_video(audio_path="./final_output.wav", video_path=filepath)
            
            # Create final dubbed video
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dubbed_' + filename)
            os.system(f'ffmpeg -i {filepath} -i ./speedup_audio.wav -map 0:v -map 1:a -c:v copy -shortest {output_path}')
            
            return jsonify({
                'message': 'Video dubbed successfully',
                'video_path': output_path
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
