"""
Cross-Language Embedding Matcher using LaBSE

Uses LaBSE (Language-agnostic BERT Sentence Embeddings) to find
where English source text matches Korean translation, enabling
us to only translate NEW content.
"""

import time
from typing import Optional, Tuple, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Global model instance (loaded once at startup)
_model: Optional[SentenceTransformer] = None
_model_name = "LaBSE"


def load_model(model_name: str = "LaBSE") -> SentenceTransformer:
    """
    Load the embedding model. Called once at server startup.
    Model is cached after first download (~1.88GB for LaBSE).
    
    Usage in main.py:
        from src.utils.embeddings import load_model
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            load_model()  # Preload at startup
            yield
    """
    global _model, _model_name
    
    if _model is None:
        print(f"🔄 Loading {model_name} embedding model...")
        print(f"   (First time: ~3-4 min download, then instant from cache)")
        start = time.time()
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        print(f"✅ {model_name} loaded in {time.time() - start:.2f}s")
    
    return _model


def get_model() -> SentenceTransformer:
    """Get the loaded model, or load it if not already loaded."""
    if _model is None:
        return load_model()
    return _model


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts (any language).
    Returns value between -1 and 1 (higher = more similar).
    """
    model = get_model()
    emb_a = model.encode([text_a])
    emb_b = model.encode([text_b])
    return float(cosine_similarity(emb_a, emb_b)[0][0])


def get_prefixes(text: str) -> List[str]:
    """
    Get all prefix combinations of a text.
    
    "Hey Vsauce Michael here" → [
        "Hey",
        "Hey Vsauce",
        "Hey Vsauce Michael",
        "Hey Vsauce Michael here"
    ]
    """
    words = text.split()
    prefixes = []
    for i in range(1, len(words) + 1):
        prefixes.append(" ".join(words[:i]))
    return prefixes


def find_source_boundary(
    source_text: str,
    previous_translation: str,
    threshold: float = 0.55,
    tolerance: float = 0.03,
    debug: bool = False,
) -> Tuple[str, str, float]:
    """
    Find where in source_text the previous_translation ends.
    
    Uses prefix-based matching:
    1. Generate all prefixes of source_text
    2. Compare each prefix to previous_translation using embeddings
    3. Find the peak similarity score
    4. Among prefixes ending with punctuation, pick the one closest to peak score
    
    Args:
        source_text: Full English transcript from ASR
        previous_translation: Korean translation from previous cycle
        threshold: Minimum similarity to consider a match (default 0.55)
        tolerance: Accept longer prefix if within this range of peak (default 0.03)
        debug: Print detailed logging if True
    
    Returns:
        (matched_source, remaining_source, best_score)
    """
    if not previous_translation or not source_text:
        return "", source_text, 0.0
    
    model = get_model()
    words = source_text.split()
    
    if not words:
        return "", source_text, 0.0
    
    # Encode previous translation once
    prev_emb = model.encode([previous_translation])
    
    # Calculate similarity for all prefixes
    if debug:
        print(f"\n      📊 Prefix similarities:")
    scores = []
    for i in range(1, len(words) + 1):
        prefix = " ".join(words[:i])
        prefix_emb = model.encode([prefix])
        score = float(cosine_similarity(prefix_emb, prev_emb)[0][0])
        scores.append((i, prefix, score))
        if debug:
            print(f"         [{i:2d}] {score:.3f} | \"{prefix}\"")
    
    # Find peak score
    peak_idx, peak_prefix, peak_score = max(scores, key=lambda x: x[2])
    if debug:
        print(f"\n      🎯 Peak: [{peak_idx}] {peak_score:.3f} | \"{peak_prefix}\"")
    
    # Find all prefixes ending with sentence punctuation
    punctuation_chars = '.!?。！？'
    punctuated = []
    for i, prefix, score in scores:
        if prefix and prefix.rstrip()[-1] in punctuation_chars:
            distance = abs(peak_score - score)
            punctuated.append((i, prefix, score, distance))
    
    # Pick the punctuated prefix closest to peak score
    if punctuated:
        best_punct = min(punctuated, key=lambda x: x[3])  # min distance from peak
        final_idx = best_punct[0]
        final_prefix = best_punct[1]
        final_score = best_punct[2]
        if debug:
            print(f"      🛑 Selected punctuation boundary [{final_idx}] {final_score:.3f} | \"{final_prefix}\"")
    else:
        # No punctuation found, use peak
        final_idx = peak_idx
        if debug:
            print(f"      ⚠️ No punctuation found, using peak")
    
    # If best score is below threshold, no reliable match
    if peak_score < threshold:
        if debug:
            print(f"      ⚠️  Below threshold ({peak_score:.3f} < {threshold})")
        return "", source_text, peak_score
    
    matched = " ".join(words[:final_idx])
    remaining = " ".join(words[final_idx:])
    
    return matched, remaining, peak_score


class EmbeddingMatcher:
    """
    Stateful embedding matcher for streaming translation.
    Tracks previous translations and finds new content to translate.
    """
    
    def __init__(self, threshold: float = 0.55, tolerance: float = 0.03):
        self.threshold = threshold
        self.tolerance = tolerance
        self.previous_translation = ""
        self.confirmed_source = ""
        
        # Ensure model is loaded
        load_model()
    
    def get_new_content(self, full_source: str) -> Tuple[str, float]:
        """
        Get the portion of source that hasn't been translated yet.
        
        Args:
            full_source: Complete ASR transcript
        
        Returns:
            (new_content, match_score)
        """
        if not self.previous_translation:
            return full_source, 0.0
        
        matched, remaining, score = find_source_boundary(
            full_source,
            self.previous_translation,
            self.threshold,
            self.tolerance,
        )
        
        return remaining, score
    
    def update(self, full_source: str, new_translation: str):
        """
        Update with new translation result.
        
        Args:
            full_source: Complete ASR transcript
            new_translation: Latest full translation
        """
        self.previous_translation = new_translation
    
    def reset(self):
        """Reset state for new session."""
        self.previous_translation = ""
        self.confirmed_source = ""