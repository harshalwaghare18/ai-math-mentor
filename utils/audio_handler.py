import os
from pathlib import Path

class AudioHandler:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
    
    def transcribe_audio(self, audio_file_path: str):
        """
        Transcribe audio using OpenAI Whisper API
        Returns: (transcript, confidence_estimate, math_phrases_detected)
        """
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key)
        
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        
        text = transcript.text
        
        # Detect math-specific phrases
        math_phrases = {
            "square root": "√",
            "raised to": "^",
            "to the power": "^",
            "divided by": "÷",
            "times": "*",
            "plus": "+",
            "minus": "-",
            "integral": "∫",
            "summation": "Σ"
        }
        
        detected_phrases = [p for p in math_phrases.keys() if p.lower() in text.lower()]
        
        return {
            "text": text,
            "math_phrases_detected": detected_phrases,
            "needs_review": len(detected_phrases) > 0  # Ask user to confirm
        }
