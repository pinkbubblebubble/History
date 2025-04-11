import whisper
import os
from smolagents import Tool
from pydub import AudioSegment
import time
from smolagents.models import MessageRole, Model

class SpeechRecognitionTool(Tool):
    name = "Speech_Recognition_Tool"
    description = "Convert speech to text and optimize with LLM using OpenAI Whisper and GPT"
    inputs = {
        "audio_file": {
            "type": "string",
            "description": "Path to the audio file",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, model=None):
        super().__init__()
        try:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully")
            self.TEMP_ROOT = "temp_audio"
            self.OUTPUT_ROOT = "speech_recognition_output"
            os.makedirs(self.TEMP_ROOT, exist_ok=True)
            os.makedirs(self.OUTPUT_ROOT, exist_ok=True)
            
            # Store the LLM model
            self.model = model
            if not self.model:
                raise ValueError("LLM model is required for text optimization")
            
        except Exception as e:
            raise Exception(f"Failed to initialize models: {str(e)}")

    def _optimize_text_with_llm(self, text: str) -> str:
        """Use LLM to optimize the transcribed text"""
        try:
            print("Optimizing text with LLM...")
            
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are a speech-to-text optimization expert. 
                            Your task is to improve the transcribed text while maintaining its original meaning. That is what you do is to make the sentences meaningful and fluent.You cannot omit any information and you cannot add any information. 
                            Optimize in its original language.
                            Provide the output in the following format:

                            === Optimized Transcription ===
                            [optimized text]

                            === Summary ===
                            [brief summary]

                            === Key Points ===
                            - [key point 1]
                            - [key point 2]
                            - [etc.]"""
                        }
                    ],
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Here is the transcribed text to optimize:\n\n{text}"
                        }
                    ],
                }
            ]
            
            # Get response from LLM
            response = self.model(messages).content
            return text, response  # 返回原始文本和优化后的文本
            
        except Exception as e:
            print(f"LLM optimization failed: {str(e)}")
            return text, text  # 如果优化失败，返回相同的文本

    def _process_audio(self, audio_file: str) -> str:
        """Process audio file and return transcribed text"""
        try:
            print(f"Processing audio file: {audio_file}")
            
            # Load audio
            audio = whisper.load_audio(audio_file)
            
            # Detect language
            # audio = whisper.pad_or_trim(audio)
            # mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
            # _, probs = self.whisper_model.detect_language(mel)
            # detected_lang = max(probs, key=probs.get)
            # print(f"Detected language: {detected_lang}")
            
            # Transcribe
            print("Starting transcription...")
            result = self.whisper_model.transcribe(
                audio_file,
                verbose=True,
                word_timestamps=False
            )
            
            # Get transcribed text
            transcribed_text = result["text"].strip()
            
            # Optimize with LLM
            original_text, optimized_text = self._optimize_text_with_llm(transcribed_text)
            
            return original_text, optimized_text
            
        except Exception as e:
            raise Exception(f"Audio processing failed: {str(e)}")

    def forward(self, audio_file: str = None) -> str:
        """Main processing function"""
        try:
            if audio_file is None:
                return "Error: No audio file provided"

            if not os.path.exists(audio_file):
                return f"Error: File not found: {audio_file}"
            
            # Process audio file
            start_time = time.time()
            original_text, optimized_text = self._process_audio(audio_file)
            processing_time = time.time() - start_time
            
            if original_text and optimized_text:
                output_filename = os.path.join(
                    self.OUTPUT_ROOT,
                    f"transcript_{os.path.basename(audio_file)}_{int(time.time())}.txt"
                )
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write("=== Original Transcription ===\n")
                    f.write(original_text)
                    f.write("\n\n")
                    f.write("=== LLM Optimized Result ===\n")
                    f.write(optimized_text)
                print(f"✓ Results saved to: {output_filename}")
            
            print(f"✓ Processing completed in {processing_time:.2f} seconds")
            
            return f"""=== Original Transcription ===
{original_text}

=== LLM Optimized Result ===
{optimized_text}"""
                
        except Exception as e:
            return f"Speech recognition error: {str(e)}"