from typing import Optional
import re


class TextProcessor:
    def __init__(self, max_length: int = 500):
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters (keeping punctuation)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def validate_text(self, text: str) -> tuple[bool, Optional[str]]:
        if not text:
            return False, "Text cannot be empty"

        if len(text) > self.max_length:
            return False, f"Text exceeds maximum length of {self.max_length} characters"

        if not any(c.isalpha() for c in text):
            return False, "Text must contain at least one letter"

        return True, None

    def process_text(self, text: str) -> tuple[str, Optional[str]]:
        try:
            if not isinstance(text, str):
                return "", "Input must be a string"

            cleaned_text = self.clean_text(text)

            is_valid, error = self.validate_text(cleaned_text)
            if not is_valid:
                return "", error

            return cleaned_text, None

        except Exception as e:
            return "", f"Error processing text: {str(e)}"