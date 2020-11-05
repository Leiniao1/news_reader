from typing import List
import tensorflow as tf
import sentencepiece as sp

def create_text_encoder(encoder_type: str, vocab_filename: str):
    if encoder_type == "sentencepiece":
        return SentencePieceEncoder(vocab_filename)
    elif encoder_type == "sentencepiece_new_line":
        return SentencePieceEncoder(vocab_filename, "<n>")


class SentencePieceEncoder(object):
    """Provides encoding and decoding logic from string to id.
    """
    
    def __init__(self, 
                 model_file: str,
                 reserved_tokens: int = 103,
                 new_line_symbol: str = ""):
        self._tokenizer = sp.SentencePieceProcessor()
        self._model = tf.io.gfile.GFile(model_file, "rb").read()
        self._tokenizer.LoadFromSerializedProto(self._model)
        self._reserved_tokens = reserved_tokens
        self._new_line_symbol = new_line_symbol
        
    def encode(self, text: str) -> List[int]:
        if self._new_line_symbol:
            text = text.replace("\n", self._new_line_symbol)
        ids = self._tokenizer.EncodeAsIds(text)
        ids = [i + self._reserved_tokens if i > 1 else i for i in ids]
        return ids
        
    def decode(self, ids: List[int]) -> str:
        ids = [
            i - self._reserved_tokens
            if i > 1 + self._reserved_tokens else i for i in ids
        ]
        text = self._tokenizer.DecodeIds(ids)
        if self._new_line_symbol:
            text = text.replace(self._new_line_symbol, "\n")
        return text
    
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.GetPieceSize() + self._reserved_tokens
