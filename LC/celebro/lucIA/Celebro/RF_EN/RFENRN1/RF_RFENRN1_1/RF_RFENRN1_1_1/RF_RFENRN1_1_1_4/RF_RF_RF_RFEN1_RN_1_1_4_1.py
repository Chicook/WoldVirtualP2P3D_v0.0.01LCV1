from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class ENRN_TKNZ:
    """Tokenizer neuron implementing a tiny BPE-like encoder/decoder.

    This class provides:
    - fit_bpe: learn merge ranks from a small corpus
    - encode: text -> list[int]
    - decode: list[int] -> text
    - normalize: optional unicode normalization and lowercasing
    - train_bpe_with_tokenizers: optional HF tokenizers backend
    - train_unigram_with_sentencepiece: optional SentencePiece backend
    - load_external_vocab: load a pretrained vocab file
    """

    vocab: Dict[str, int] = field(default_factory=dict)
    merges: Dict[Tuple[str, str], int] = field(default_factory=dict)
    unk_token: str = "<unk>"
    lowercase: bool = False
    normalize_nfkc: bool = True

    def fit_bpe(self, corpus: List[str], vocab_size: int = 2000) -> None:
        tokens: Dict[str, int] = {}
        for text in corpus:
            for ch in text:
                tokens[ch] = tokens.get(ch, 0) + 1
        # Initialize vocab with single characters
        self.vocab = {self.unk_token: 0}
        for i, ch in enumerate(sorted(tokens.keys()), start=1):
            self.vocab[ch] = i

        # Dummy merge ranks: no heavy BPE, just adjacent pairs frequency
        pair_freq: Dict[Tuple[str, str], int] = {}
        for text in corpus:
            chars = list(text)
            for a, b in zip(chars, chars[1:]):
                pair = (a, b)
                pair_freq[pair] = pair_freq.get(pair, 0) + 1

        # Keep top pairs as merges
        top_pairs = sorted(pair_freq.items(), key=lambda kv: kv[1], reverse=True)
        self.merges = {p: rank for rank, (p, _) in enumerate(top_pairs[: max(0, vocab_size - len(self.vocab))])}

    def normalize(self, text: str) -> str:
        t = text
        if self.normalize_nfkc:
            try:
                import unicodedata  # optional stdlib
                t = unicodedata.normalize("NFKC", t)
            except Exception:
                pass
        if self.lowercase:
            t = t.lower()
        return t

    def encode(self, text: str) -> List[int]:
        text = self.normalize(text)
        if not self.vocab:
            # minimal fallback: ascii codes clipped to range
            return [min(ord(ch), 1023) for ch in text]

        # greedy char/pair encoding
        i = 0
        ids: List[int] = []
        while i < len(text):
            ch = text[i]
            token = ch
            if i + 1 < len(text):
                pair = (text[i], text[i + 1])
                if pair in self.merges:
                    token = "".join(pair)
                    i += 1
            ids.append(self.vocab.get(token, self.vocab[self.unk_token]))
            i += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        if not self.vocab:
            return "".join(chr(i) for i in ids)

        inv_vocab = {v: k for k, v in self.vocab.items()}
        return "".join(inv_vocab.get(i, self.unk_token) for i in ids)

    # Optional: integrate HuggingFace Tokenizers if present (no hard dependency)
    def train_bpe_with_tokenizers(self, corpus: List[str], vocab_size: int = 2000) -> Optional[object]:
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import ByteLevel
        except Exception:
            return None

        tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[self.unk_token])
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        # Mirror minimal vocab into self for compatibility
        self.vocab = {tok: idx for tok, idx in tokenizer.get_vocab().items()}
        return tokenizer

    def train_unigram_with_sentencepiece(self, corpus: List[str], vocab_size: int = 2000, model_prefix: str = "spm") -> Optional[str]:
        try:
            import sentencepiece as spm  # type: ignore
        except Exception:
            return None
        input_text = "\n".join(corpus)
        # Use SentencePieceTrainer directly from API
        try:
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(corpus),
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="unigram",
                unk_id=0,
                character_coverage=0.9995,
            )
            return f"{model_prefix}.model"
        except Exception:
            # Fallback to temp file path-based training if iterator unsupported
            import tempfile
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
                f.write(input_text)
                tmp = f.name
            spm.SentencePieceTrainer.train(
                input=tmp,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="unigram",
                unk_id=0,
                character_coverage=0.9995,
            )
            return f"{model_prefix}.model"

    def load_external_vocab(self, vocab: Dict[str, int]) -> None:
        self.vocab = dict(vocab)
