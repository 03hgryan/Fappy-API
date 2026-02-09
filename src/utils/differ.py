import re


def normalize(word: str) -> str:
    """Strip punctuation and lowercase for comparison."""
    return re.sub(r'[^\w]', '', word).lower()


def diff_new_words(prev_english: str, new_english: str) -> str:
    """
    Given two ElevenLabs partials (which are sliding windows),
    find genuinely new words in new_english that aren't covered by prev_english.
    
    Strategy: Take progressively smaller chunks from the tail of prev,
    and search for them anywhere in new. Once found, everything AFTER
    that match position in new is genuinely new content.
    """
    if not prev_english:
        return new_english.strip()
    if not new_english:
        return ""

    prev_norm = [normalize(w) for w in prev_english.split()]
    new_raw = new_english.split()
    new_norm = [normalize(w) for w in new_raw]

    if not prev_norm or not new_norm:
        return new_english.strip()

    max_tail = min(len(prev_norm), len(new_norm))

    for tail_len in range(max_tail, 0, -1):
        tail = prev_norm[-tail_len:]

        for start in range(len(new_norm) - tail_len + 1):
            if new_norm[start:start + tail_len] == tail:
                after = start + tail_len
                if after < len(new_raw):
                    return " ".join(new_raw[after:]).strip()
                else:
                    return ""

    return new_english.strip()