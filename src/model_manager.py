from pathlib import Path
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Tuple, Any


class ModelManager:
    def __init__(self, checkpoint_name: str, local_models_dir: str):
        self.checkpoint_name = checkpoint_name
        self.model_name = checkpoint_name.split('/')[-1]
        self.local_models_dir = Path(local_models_dir)
        self.model_path = self.local_models_dir / self.model_name

    def get_model_and_tokenizer(self) -> Tuple[Any, Any]:
        os.makedirs(self.local_models_dir, exist_ok=True)

        try:
            if self._check_local_files():
                print(f"Loading model {self.model_name} from local storage...")
                model = self._load_model_from_local()
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    local_files_only=True
                )
            else:
                print(f"Downloading model {self.model_name} from Hugging Face hub...")
                try:
                    # Download and save the model first
                    print("Downloading and saving the model...")
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.checkpoint_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_name)

                    # Save the model and tokenizer
                    print(f"Saving model to {self.model_path}...")
                    model.save_pretrained(str(self.model_path))
                    tokenizer.save_pretrained(str(self.model_path))

                    # Load the model with device mapping
                    print("Loading model with device mapping...")
                    model = self._load_model_from_local()

                except Exception as e:
                    print(f"Error downloading model: {str(e)}")
                    raise

            return model, tokenizer

        except Exception as e:
            print(f"Error in get_model_and_tokenizer: {str(e)}")
            raise

    def _load_model_from_local(self) -> Any:
        return AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_path),
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )

    def _check_local_files(self) -> bool:
        print(f"Checking local path '{str(self.model_path)}'...")

        required_files = [
            'config.json',
            'tokenizer.json',
            'tokenizer_config.json'
        ]

        if not self.model_path.exists():
            return False

        return all((self.model_path / file).exists() for file in required_files)