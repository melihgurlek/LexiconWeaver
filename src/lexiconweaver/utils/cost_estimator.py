"""Cost estimation utilities for batch translation workflow."""

from dataclasses import dataclass
from pathlib import Path

from lexiconweaver.exceptions import ValidationError
from lexiconweaver.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CostEstimate:
    """Cost estimation result for batch translation."""
    
    file_count: int
    total_words: int
    total_chars: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    estimated_cost_min: float
    estimated_cost_max: float
    estimated_time_minutes: float
    provider: str


PRICING = {
    "deepseek": {
        "input": 0.27 / 1_000_000,
        "output": 1.10 / 1_000_000,
    },
    "ollama": {
        "input": 0.0,
        "output": 0.0,
    }
}


def calculate_dry_run_summary(
    input_folder: Path,
    provider: str = "deepseek",
    max_parallel: int = 5,
    pattern: str = "*.txt"
) -> CostEstimate:
    """
    Calculate token counts and estimated costs for batch translation.
    
    Args:
        input_folder: Path to folder containing chapter files
        provider: LLM provider name (for pricing)
        max_parallel: Maximum parallel chapters (for time estimation)
        pattern: Glob pattern for chapter files
        
    Returns:
        CostEstimate object with detailed cost breakdown
        
    Raises:
        ValidationError: If input folder doesn't exist or has no files
    """
    if not input_folder.exists():
        raise ValidationError(
            f"Input folder does not exist: {input_folder}",
            details="Create the folder and place chapter files there"
        )
    
    if not input_folder.is_dir():
        raise ValidationError(
            f"Input path is not a directory: {input_folder}",
            details="Provide a directory path"
        )
    
    chapters = list(input_folder.glob(pattern))
    
    if not chapters:
        raise ValidationError(
            f"No chapter files found in {input_folder}",
            details=f"Looking for files matching: {pattern}"
        )
    
    total_words = 0
    total_chars = 0
    
    for chapter_file in sorted(chapters):
        try:
            text = chapter_file.read_text(encoding='utf-8')
            total_words += len(text.split())
            total_chars += len(text)
        except Exception as e:
            logger.warning(f"Failed to read chapter {chapter_file.name}: {e}")
            continue
    
    tokens_per_char = 0.27
    input_tokens = int(total_chars * tokens_per_char)
    
    output_tokens = input_tokens
    
    pricing = PRICING.get(provider.lower(), PRICING["deepseek"])
    input_cost = input_tokens * pricing["input"]
    output_cost = output_tokens * pricing["output"]
    total_cost = input_cost + output_cost
    
    cost_min = total_cost * 0.8
    cost_max = total_cost * 1.2
    
    avg_time_per_chapter = 0.6
    est_time = (len(chapters) * avg_time_per_chapter) / max_parallel
    
    logger.info(
        "Cost estimation completed",
        files=len(chapters),
        words=total_words,
        tokens=input_tokens + output_tokens,
        cost=f"${total_cost:.2f}"
    )
    
    return CostEstimate(
        file_count=len(chapters),
        total_words=total_words,
        total_chars=total_chars,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=total_cost,
        estimated_cost_min=cost_min,
        estimated_cost_max=cost_max,
        estimated_time_minutes=est_time,
        provider=provider
    )


def format_dry_run_summary(estimate: CostEstimate) -> str:
    """
    Format cost estimate as a pretty summary.
    
    Args:
        estimate: CostEstimate object
        
    Returns:
        Formatted string for display
    """
    # Format provider name
    provider_display = estimate.provider.capitalize()
    if estimate.provider.lower() == "ollama":
        provider_display = "Ollama (Local/Free)"
    
    summary = f"""
📊 Dry Run Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Files:           {estimate.file_count} chapters
  Total Words:     {estimate.total_words:,}
  Total Chars:     {estimate.total_chars:,}
  
  Est. Input:      ~{estimate.input_tokens:,} tokens
  Est. Output:     ~{estimate.output_tokens:,} tokens
  
  Provider:        {provider_display}
"""
    
    # Add cost info (skip if free)
    if estimate.estimated_cost_usd > 0:
        summary += f"  💰 Est. Cost:    ${estimate.estimated_cost_min:.2f} - ${estimate.estimated_cost_max:.2f}\n"
        summary += f"                   (Base: ${estimate.estimated_cost_usd:.2f}, ±20%)\n"
    else:
        summary += f"  💰 Est. Cost:    Free (local model)\n"
    
    summary += f"""  
  ⏱️  Est. Time:    ~{estimate.estimated_time_minutes:.1f} minutes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Note: Estimates are approximate. Actual costs may vary based on:
- Glossary size (more terms = slightly higher input tokens)
- Chapter complexity (technical terms, dialogue, etc.)
- API rate limiting (may increase time)
"""
    
    return summary


def estimate_tokens_for_text(text: str) -> int:
    """
    Estimate token count for a given text.
    
    Uses approximation: ~4 characters per token.
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    return int(len(text) * 0.25)
