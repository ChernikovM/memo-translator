from pathlib import Path
from typing import Optional
import fasttext
from transformers import pipeline
from dotenv import load_dotenv
import os
import torch
import requests
from tqdm import tqdm

from .text_processor import TextProcessor
from .model_manager import ModelManager


class Translator:

    def __init__(self):
        load_dotenv()
        self.model_checkpoint = os.getenv('MODEL_CHECKPOINT')
        self.pretrained_dir = Path('pretrained')
        self.language_model_path = self.pretrained_dir / 'lid218e.bin'
        self.local_models_dir = os.getenv('LOCAL_MODELS_DIR', 'models')
        self.max_length = int(os.getenv('MAX_LENGTH', 500))
        self.lang_model_url = os.getenv('LANG_MODEL_URL')

        # Create pretrained directory if it doesn't exist
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)

        self.validate_config()
        self._ensure_language_model()

        self.model_manager = ModelManager(self.model_checkpoint, self.local_models_dir)
        self.text_processor = TextProcessor(max_length=self.max_length)
        self.lang_model = None
        self.translation_model = None
        self.tokenizer = None
        self.translation_pipeline = None

    def _download_file(self, url: str, destination: Path) -> None:
        """
        Download a file with progress bar
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading language model") as progress_bar:
            with open(destination, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

    def _ensure_language_model(self) -> None:
        """
        Check if language model exists and download if it doesn't
        """
        if not self.language_model_path.exists():
            print(f"Language model not found at {self.language_model_path}")
            print(f"Downloading from {self.lang_model_url}...")
            try:
                self._download_file(self.lang_model_url, self.language_model_path)
                print(f"Language model downloaded successfully to {self.language_model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download language model: {str(e)}")

    def validate_config(self):
        required_vars = {
            'MODEL_CHECKPOINT': self.model_checkpoint,
            'LOCAL_MODELS_DIR': self.local_models_dir
        }

        missing_vars = [var for var, value in required_vars.items() if not value]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def setup_models(self) -> None:
        try:
            print("Loading language model...")
            self.lang_model = self._load_language_model()
            print("Language model loaded successfully!")

            print("Loading translation model and tokenizer...")
            self.translation_model, self.tokenizer = self.model_manager.get_model_and_tokenizer()
            print("Translation model and tokenizer loaded successfully!")

        except Exception as e:
            print(f"Error in setup_models: {str(e)}")
            raise RuntimeError(f"Error setting up models: {str(e)}")

    def _load_language_model(self) -> fasttext.FastText._FastText:
        if not self.language_model_path.exists():
            raise FileNotFoundError(f"Language model not found at {self.language_model_path}")
        return fasttext.load_model(str(self.language_model_path))

    def detect_language(self, text: str) -> str:
        if not self.lang_model:
            raise RuntimeError("Language model not initialized. Call setup_models() first.")

        predictions = self.lang_model.predict(text, k=1)
        return predictions[0][0].replace('__label__', '')

    def setup_translation_pipeline(self, source_language: str, target_language: str) -> None:
        try:
            self.translation_pipeline = pipeline(
                task='translation',
                model=self.translation_model,
                tokenizer=self.tokenizer,
                src_lang=source_language,
                tgt_lang=target_language,
                max_length=self.max_length,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            print(f"Error setting up translation pipeline: {str(e)}")
            raise RuntimeError(f"Pipeline setup error: {str(e)}")

    def translate(self, text: str, target_language: str) -> str:
        # Process input text
        processed_text, error = self.text_processor.process_text(text)
        if error:
            raise ValueError(error)

        try:
            # Detect source language
            source_language = self.detect_language(processed_text)

            # Setup translation pipeline if needed or if target language changed
            if (not self.translation_pipeline or
                    self.translation_pipeline.tokenizer.src_lang != source_language or
                    self.translation_pipeline.tokenizer.tgt_lang != target_language):
                self.setup_translation_pipeline(source_language, target_language)

            # Perform translation
            output = self.translation_pipeline(
                processed_text,
                src_lang=source_language,
                tgt_lang=target_language,
                max_length=self.max_length
            )
            return output[0]['translation_text']

        except Exception as e:
            print(f"Error during translation process: {str(e)}")
            raise RuntimeError(f"Translation error: {str(e)}")