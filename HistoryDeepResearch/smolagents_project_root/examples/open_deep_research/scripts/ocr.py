# Create in scripts/ocr_agent.py
from smolagents import Tool
from smolagents.models import MessageRole, Model
import easyocr
import os
import base64
import re
from typing import List

class OCRTool(Tool):
    name = "OCR_Tool"
    description = "Extract text from images with automatic language detection"
    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image file",
            "nullable": True
        },
        "languages": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of OCR language codes, auto-detected if not provided",
            "default": ["en", "ch_sim"],
            "nullable": True
        },
        "auto_detect": {
            "type": "boolean",
            "description": "Whether to automatically detect languages in the image",
            "default": True,
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, model=None):
        super().__init__()
        self.model = model  # LLM model for result optimization
        self._easyocr_readers = {}  # Cache readers for different language combinations
        
        # Common language groups and their codes
        self.language_groups = {
            "cjk": ["ch_sim", "ch_tra", "ja", "ko", "en"],  # Chinese, Japanese, Korean + English
            "european": ["en", "fr", "de", "es", "it", "pt", "ru"],  # Major European languages
            "indic": ["hi", "ta", "te", "kn", "mr", "ne", "en"],  # Indic languages + English
            "arabic": ["ar", "fa", "ur", "en"],  # Arabic script languages + English
            "default": ["en", "ch_sim"]  # Default combination
        }

    # 保留原有的_get_reader、_detect_language、_optimize_text_with_llm和_describe_image_with_llm方法
    # 以及forward方法的实现...
    def _get_reader(self, languages=None):
        """Get or create an EasyOCR reader"""
        if languages is None:
            languages = self.language_groups["default"]
            
        # Convert language list to sorted tuple for dictionary key
        lang_key = tuple(sorted(languages))
        
        # Reuse cached reader if available
        if lang_key in self._easyocr_readers:
            return self._easyocr_readers[lang_key]
            
        # Otherwise create a new reader
        try:
            print(f"Initializing EasyOCR with languages: {languages}")
            # Modification: explicitly set GPU=False to avoid CUDA issues
            reader = easyocr.Reader(languages, gpu=False)
            self._easyocr_readers[lang_key] = reader
            return reader
        except Exception as e:
            print(f"EasyOCR initialization failed: {str(e)}")
            # Modification: return default reader instead of None
            if lang_key != tuple(sorted(self.language_groups["default"])):
                print("Trying default language combination...")
                return self._get_reader(self.language_groups["default"])
            return None

    def _detect_language(self, image_path):
        """Detect the main language in the image using image analysis"""
        try:
            # First try using basic English OCR to get some text samples
            basic_reader = self._get_reader(["en"])
            if not basic_reader:
                return self.language_groups["default"]
                
            sample_text = basic_reader.readtext(image_path, detail=0)
            
            # If no text detected, return default language combination
            if not sample_text:
                return self.language_groups["default"]
                
            # Analyze text features to guess the language
            text = " ".join(sample_text)
            
            # Detect CJK characters (Chinese, Japanese, Korean)
            if any(ord(c) > 0x4E00 and ord(c) < 0x9FFF for c in text):
                return self.language_groups["cjk"]
                
            # Detect Cyrillic characters (Russian, etc.)
            if any(ord(c) > 0x0400 and ord(c) < 0x04FF for c in text):
                return ["ru", "en"]
                
            # Detect Arabic characters
            if any(ord(c) > 0x0600 and ord(c) < 0x06FF for c in text):
                return self.language_groups["arabic"]
                
            # Detect Indic script characters
            if any(ord(c) > 0x0900 and ord(c) < 0x097F for c in text):
                return self.language_groups["indic"]
                
            # If no special characters, assume European languages
            return self.language_groups["european"]
            
        except Exception as e:
            print(f"Language detection failed: {str(e)}")
            return self.language_groups["default"]

    def forward(self, image_path: str = None, languages: List[str] = None, auto_detect: bool = True) -> str:
        try:
            if image_path is None:
                return "Error: No image file path provided"
                
            if not os.path.exists(image_path):
                return f"Error: File path not found: {image_path}"
                
            # Check file extension
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                return "Error: Unsupported image format. Supported formats include: JPG, JPEG, PNG, BMP, TIFF, WEBP"
            
            # If no languages provided and auto-detect is enabled, detect languages
            if languages is None and auto_detect:
                print("Auto-detecting image language...")
                languages = self._detect_language(image_path)
                print(f"Detected possible language combination: {languages}")
            elif languages is None:
                languages = self.language_groups["default"]
                
            # Get OCR reader
            reader = self._get_reader(languages)
            if reader is None:
                # Modification: return more detailed error and try system description
                print("Unable to initialize OCR engine, trying to describe image with system tools...")
                if self.model:
                    return self._describe_image_with_llm(image_path)
                return "Error: Unable to initialize OCR engine, please check EasyOCR installation and language support"
                
            # Perform OCR
            print(f"Performing OCR on image {image_path} with languages: {languages}")
            try:
                # Add timeout handling
                results = reader.readtext(image_path, detail=0)  # detail=0 returns only text
            except Exception as ocr_error:
                print(f"OCR processing failed: {str(ocr_error)}")
                # Try with default languages
                if languages != self.language_groups["default"]:
                    print("Trying default language combination...")
                    reader = self._get_reader(self.language_groups["default"])
                    if reader:
                        try:
                            results = reader.readtext(image_path, detail=0)
                        except:
                            results = []
                    else:
                        results = []
                else:
                    results = []
            
            # Check results
            if not results:
                print("OCR detected no text, trying other language combinations...")
                # Try other language combinations
                for group_name, group_langs in self.language_groups.items():
                    if group_langs != languages:
                        print(f"Trying {group_name} language combination: {group_langs}")
                        alt_reader = self._get_reader(group_langs)
                        if alt_reader:
                            try:
                                alt_results = alt_reader.readtext(image_path, detail=0)
                                if alt_results:
                                    results = alt_results
                                    print(f"Successfully detected text using {group_name} language combination")
                                    break
                            except Exception as e:
                                print(f"OCR with {group_name} language combination failed: {str(e)}")
                
                # If still no results, try using LLM to describe the image
                if not results and self.model:
                    return self._describe_image_with_llm(image_path)
                elif not results:
                    return "No text detected in the image. Please try another tool or process manually"
                
            # Combine results into text
            text = "\n".join(results)
            
            # Use LLM to optimize OCR results (if model available)
            if self.model and self._needs_optimization(text):
                return self._optimize_with_llm(text, image_path)
            
            return text
            
        except Exception as e:
            print(f"OCR processing failed: {str(e)}")
            # If model available, try using LLM to describe the image
            if self.model:
                return self._describe_image_with_llm(image_path)
            return f"OCR processing failed: {str(e)}"

    def _needs_optimization(self, text: str) -> bool:
        """Determine if OCR results need optimization"""
        if not text:
            return False
            
        # Check if text contains too high a proportion of special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / len(text) > 0.3:  # If special characters exceed 30%
            return True
            
        # Check for too many consecutive non-word characters (possible recognition errors)
        if re.search(r'[^\w\s]{3,}', text):
            return True
            
        # Check for unusual character combinations
        if re.search(r'[a-z][A-Z]{2,}|[A-Z][a-z]{2,}[A-Z]', text):
            return True
            
        return True
        
    def _optimize_with_llm(self, text: str, image_path: str) -> str:
        """Use LLM to optimize OCR results"""
        try:
            # Use the same message format as TextInspectorTool
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an OCR text correction expert. Your task is to fix OCR recognition errors and provide accurate text. Return only the corrected text without any explanations, analysis, or comments."
                        }
                    ],
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Below is the original text from OCR recognition, which may contain errors:\n\n{text}\n\nPlease provide the corrected plain text only, without any additional content:"
                        }
                    ],
                }
            ]
            
            print("DEBUG: Using consistent call method with TextInspectorTool")
            
            # Directly call the model and get content attribute
            response = self.model(messages).content
            
            # Ensure non-empty content is returned
            if response and response.strip():
                return response.strip()
            else:
                print("DEBUG: Response is empty, returning original text")
                return text
            
        except Exception as e:
            print(f"LLM optimization failed: {str(e)}")
            # Ensure all possible exceptions are handled
            return text

    # Add new method: Use LLM to describe image
    def _describe_image_with_llm(self, image_path: str) -> str:
        """When OCR fails, use LLM to describe image content"""
        try:
            # Read image file and convert to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare messages
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an image analysis expert. Please describe the content of the image in detail, with special attention to any text, symbols, charts, or important visual elements."
                        }
                    ],
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": "Please describe in detail the content of this image, especially any visible text:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ],
                }
            ]
            
            # Call the model
            response = self.model(messages).content
            
            if response and response.strip():
                return f"[Image Description] {response.strip()}"
            else:
                return "Unable to analyze image content"
            
        except Exception as e:
            print(f"Image description failed: {str(e)}")
            return f"Unable to process image: {str(e)}"
