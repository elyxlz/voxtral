"""
composite tokenizer mimi + word level whisper into mistral text tokens
coarse to fine [text (5hz) -> semantic tokens (12.5hz) -> acoustic tokens (87.5hz)].
total = 105hz
"""
