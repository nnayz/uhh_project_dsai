"""Segment-level operations helper placeholders."""


def split_segments(x, segment_size):
    """Split a sequence `x` into segments of `segment_size` (last may be shorter)."""
    return [x[i:i+segment_size] for i in range(0, len(x), segment_size)]
