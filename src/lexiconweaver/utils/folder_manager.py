"""Folder management utilities for batch translation workflow."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lexiconweaver.exceptions import ValidationError
from lexiconweaver.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ChapterFile:
    """Represents a chapter file discovered in the input folder."""
    
    path: Path
    filename: str
    number: int
    name: Optional[str] = None


def setup_workspace(base_path: Path) -> dict[str, Path]:
    """
    Set up workspace folder structure for batch translation.
    
    Creates:
    - input/    - User places raw chapters here
    - output/   - Individual translated chapters
    - merged/   - Final merged output with TOC
    - .weavecodex/ - Metadata cache
    
    Args:
        base_path: Base directory for the workspace
        
    Returns:
        Dictionary mapping folder names to their paths
        
    Raises:
        ValidationError: If folder creation fails
    """
    folders = {
        "input": base_path / "input",
        "output": base_path / "output",
        "merged": base_path / "merged",
        "metadata": base_path / ".weavecodex",
    }
    
    try:
        for folder_name, folder_path in folders.items():
            folder_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified {folder_name} folder", path=str(folder_path))
    except Exception as e:
        raise ValidationError(
            f"Failed to create workspace folders: {e}",
            details=f"Base path: {base_path}"
        ) from e
    
    return folders


def discover_chapters(input_path: Path, pattern: str = "*.txt") -> list[ChapterFile]:
    """
    Discover and sort chapter files in the input folder.
    
    Looks for files matching the pattern and extracts chapter numbers
    from filenames. Supports various naming conventions:
    - 01_prologue.txt
    - chapter_01.txt
    - 001.txt
    - Chapter 1 - Introduction.txt
    
    Args:
        input_path: Directory containing chapter files
        pattern: Glob pattern for matching files (default: "*.txt")
        
    Returns:
        List of ChapterFile objects sorted by chapter number
        
    Raises:
        ValidationError: If no chapters found or path doesn't exist
    """
    if not input_path.exists():
        raise ValidationError(
            f"Input path does not exist: {input_path}",
            details="Create the input folder and place chapter files there"
        )
    
    if not input_path.is_dir():
        raise ValidationError(
            f"Input path is not a directory: {input_path}",
            details="Provide a directory path, not a file path"
        )
    
    files = list(input_path.glob(pattern))
    
    if not files:
        raise ValidationError(
            f"No chapter files found in {input_path}",
            details=f"Looking for files matching: {pattern}"
        )
    
    chapters = []
    for file_path in files:
        try:
            chapter_num, chapter_name = parse_chapter_number(file_path.name)
            chapters.append(ChapterFile(
                path=file_path,
                filename=file_path.name,
                number=chapter_num,
                name=chapter_name
            ))
        except ValueError as e:
            logger.warning(
                f"Skipping file (couldn't parse chapter number)",
                file=file_path.name,
                error=str(e)
            )
            continue
    
    if not chapters:
        raise ValidationError(
            f"No valid chapter files found in {input_path}",
            details="Chapter files must have numbers in their names (e.g., 01_prologue.txt)"
        )
    
    chapters.sort(key=lambda ch: ch.number)
    
    logger.info(
        f"Discovered {len(chapters)} chapters",
        first=chapters[0].filename,
        last=chapters[-1].filename
    )
    
    return chapters


def parse_chapter_number(filename: str) -> tuple[int, Optional[str]]:
    """
    Extract chapter number from filename.
    
    Supports various naming conventions:
    - 01_prologue.txt → (1, "prologue")
    - chapter_05.txt → (5, None)
    - 003.txt → (3, None)
    - Chapter 12 - The Beginning.txt → (12, "The Beginning")
    - ch10_battle.txt → (10, "battle")
    
    Args:
        filename: The filename to parse
        
    Returns:
        Tuple of (chapter_number, chapter_name)
        
    Raises:
        ValueError: If no chapter number found in filename
    """
    name_without_ext = Path(filename).stem
    
    match = re.match(r"^(\d+)(?:_(.+))?$", name_without_ext)
    if match:
        chapter_num = int(match.group(1))
        chapter_name = match.group(2) if match.group(2) else None
        return chapter_num, chapter_name
    
    match = re.match(r"^(?:chapter|ch)[\s_-]*(\d+)(?:[\s_-]+(.+))?$", name_without_ext, re.IGNORECASE)
    if match:
        chapter_num = int(match.group(1))
        chapter_name = match.group(2) if match.group(2) else None
        return chapter_num, chapter_name
    
    match = re.search(r"(\d+)", name_without_ext)
    if match:
        chapter_num = int(match.group(1))
        name_part = name_without_ext[match.end():].strip("_- ")
        chapter_name = name_part if name_part else None
        return chapter_num, chapter_name
    
    raise ValueError(f"No chapter number found in filename: {filename}")


def get_translated_chapters(output_path: Path, pattern: str = "*_translated.txt") -> set[int]:
    """
    Get set of already-translated chapter numbers.
    
    Used for checkpoint/resume functionality.
    
    Args:
        output_path: Directory containing translated chapters
        pattern: Glob pattern for translated files (default: "*_translated.txt")
        
    Returns:
        Set of chapter numbers that have already been translated
    """
    if not output_path.exists():
        return set()
    
    translated = set()
    for file_path in output_path.glob(pattern):
        try:
            chapter_num, _ = parse_chapter_number(file_path.name)
            translated.add(chapter_num)
        except ValueError:
            continue
    
    return translated


def get_output_filename(chapter: ChapterFile, suffix: str = "_translated") -> str:
    """
    Generate output filename for a translated chapter.
    
    Args:
        chapter: The chapter file
        suffix: Suffix to add before extension (default: "_translated")
        
    Returns:
        Output filename (e.g., "01_prologue_translated.txt")
    """
    stem = Path(chapter.filename).stem
    ext = Path(chapter.filename).suffix or ".txt"
    return f"{stem}{suffix}{ext}"
