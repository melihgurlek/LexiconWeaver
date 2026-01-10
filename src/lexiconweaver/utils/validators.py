"""Input validation utilities."""

from pathlib import Path

from lexiconweaver.exceptions import ValidationError

try:
    import chardet
except ImportError:
    chardet = None  # type: ignore


def validate_encoding(file_path: Path, encoding: str = "utf-8") -> str:
    """Validate and detect file encoding."""
    if chardet is None:
        # Fallback: try to read with UTF-8
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(1024)
            return "utf-8"
        except UnicodeDecodeError:
            raise ValidationError(
                f"Cannot validate encoding for {file_path}. Install chardet for encoding detection."
            )

    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)  # Sample first 10KB
            result = chardet.detect(raw_data)

        detected_encoding = result.get("encoding", encoding)
        confidence = result.get("confidence", 0.0)

        if confidence < 0.7:
            # Try to read with specified encoding as fallback
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    f.read(1024)
                return encoding
            except UnicodeDecodeError:
                raise ValidationError(
                    f"Cannot determine encoding for {file_path}. "
                    f"Detected: {detected_encoding} (confidence: {confidence:.2f})"
                )

        return detected_encoding

    except Exception as e:
        raise ValidationError(f"Failed to validate encoding for {file_path}: {e}") from e


def validate_text_file(file_path: Path) -> tuple[Path, str]:
    """Validate a text file and return path and encoding."""
    if not file_path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    # Check file size (warn if very large)
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValidationError(f"File is empty: {file_path}")

    if file_size > 100 * 1024 * 1024:  # 100MB
        raise ValidationError(
            f"File is too large ({file_size / 1024 / 1024:.1f}MB). "
            "Maximum size is 100MB."
        )

    encoding = validate_encoding(file_path)

    return file_path, encoding
