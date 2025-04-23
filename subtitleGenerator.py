def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def save_srt(translated_chunk, output_path="./static/subtitles.srt"):
    with open(output_path, "w", encoding="utf-8") as srt_file:
        for idx, item in enumerate(translated_chunk, start=1):
            start, end = item["timestamp"]
            if end is None:
                continue  # skip incomplete timestamps
            srt_file.write(f"{idx}\n")
            srt_file.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            srt_file.write(f"{item['text']}\n\n")

if __name__ == '__main__':
    save_srt()

