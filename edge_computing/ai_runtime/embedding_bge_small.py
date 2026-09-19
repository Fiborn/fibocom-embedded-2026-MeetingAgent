#!/usr/bin/env python3
from pathlib import Path
import re

import numpy as np


BASE = Path("/home/fibo")
PROJECT_ROOT = BASE / "MeetingAgent"
EMBEDDING_DIR = PROJECT_ROOT / "ai_models" / "embedding_models" / "bge-small-zh-v1.5"
EMBEDDING_DLC = EMBEDDING_DIR / "model.dlc"
EMBEDDING_ONNX = EMBEDDING_DIR / "model.onnx"
VOCAB_PATH = EMBEDDING_DIR / "vocab.txt"
SEQ_LEN = 128
HIDDEN_SIZE = 512


class BertWordPieceTokenizer:
    def __init__(self, vocab_path=VOCAB_PATH, max_length=SEQ_LEN):
        self.vocab_path = Path(vocab_path)
        self.max_length = max_length
        self.token_to_id = {}
        with self.vocab_path.open("r", encoding="utf-8") as f:
            for idx, token in enumerate(line.strip() for line in f):
                if token:
                    self.token_to_id[token] = idx
        self.unk = self.token_to_id.get("[UNK]", 100)
        self.cls = self.token_to_id.get("[CLS]", 101)
        self.sep = self.token_to_id.get("[SEP]", 102)
        self.pad = self.token_to_id.get("[PAD]", 0)

    @staticmethod
    def _is_chinese_char(ch):
        cp = ord(ch)
        return (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F
            or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF
            or 0x2F800 <= cp <= 0x2FA1F
        )

    def _basic_tokens(self, text):
        text = text.strip().lower()
        spaced = []
        for ch in text:
            if self._is_chinese_char(ch):
                spaced.extend([" ", ch, " "])
            else:
                spaced.append(ch)
        text = "".join(spaced)
        return re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)

    def _wordpiece(self, token):
        if token in self.token_to_id:
            return [token]
        if len(token) > 100:
            return ["[UNK]"]
        pieces = []
        start = 0
        while start < len(token):
            end = len(token)
            current = None
            while start < end:
                piece = token[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.token_to_id:
                    current = piece
                    break
                end -= 1
            if current is None:
                return ["[UNK]"]
            pieces.append(current)
            start = end
        return pieces

    def encode(self, text):
        tokens = ["[CLS]"]
        for token in self._basic_tokens(text):
            tokens.extend(self._wordpiece(token))
        tokens = tokens[: self.max_length - 1] + ["[SEP]"]

        input_ids = [self.token_to_id.get(token, self.unk) for token in tokens]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad] * pad_len)
            attention_mask.extend([0] * pad_len)
            token_type_ids.extend([0] * pad_len)

        return {
            "input_ids": np.asarray([input_ids], dtype=np.int64),
            "attention_mask": np.asarray([attention_mask], dtype=np.int64),
            "token_type_ids": np.asarray([token_type_ids], dtype=np.int64),
        }


def pool_and_normalize(hidden, attention_mask):
    mask = attention_mask.astype(np.float32).reshape(SEQ_LEN, 1)
    denom = float(mask.sum())
    if denom <= 0:
        pooled = hidden[0]
    else:
        pooled = (hidden * mask).sum(axis=0) / denom
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32)


class OnnxBgeSmallEmbedder:
    def __init__(self, model_path=EMBEDDING_ONNX, vocab_path=VOCAB_PATH):
        self.model_path = Path(model_path)
        self.tokenizer = BertWordPieceTokenizer(vocab_path)
        self.session = None
        self.input_names = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def open(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = [item.name for item in self.session.get_inputs()]

    def close(self):
        self.session = None
        self.input_names = None

    def embed(self, text):
        if self.session is None:
            self.open()
        encoded = self.tokenizer.encode(text)
        feed = {name: value for name, value in encoded.items() if name in self.input_names}
        outputs = self.session.run(["last_hidden_state"], feed)
        hidden = np.asarray(outputs[0], dtype=np.float32).reshape(1, SEQ_LEN, HIDDEN_SIZE)[0]
        return pool_and_normalize(hidden, encoded["attention_mask"])

    def embed_many(self, texts):
        return np.vstack([self.embed(text) for text in texts]).astype(np.float32)


class FiboDlcBgeSmallEmbedder:
    def __init__(self, model_path=EMBEDDING_DLC, vocab_path=VOCAB_PATH, runtime="CPU"):
        self.model_path = Path(model_path)
        self.tokenizer = BertWordPieceTokenizer(vocab_path)
        self.runtime = runtime
        self.api = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def open(self):
        from fiboaisdk.api_aisdk_py import api_infer_py

        params = api_infer_py.InferParams(
            str(self.model_path),
            "QUALCOMM",
            "SNPE",
            self.runtime,
            "ERROR",
            5,
        )
        self.api = api_infer_py.InferAPI()
        ret = self.api.Init(params)
        if ret != 0:
            raise RuntimeError(f"embedding model init failed: {ret}")

    def close(self):
        if self.api is not None:
            self.api.Release()
            self.api = None

    def embed(self, text):
        if self.api is None:
            self.open()
        encoded = self.tokenizer.encode(text)
        feed = {name: value.flatten().tolist() for name, value in encoded.items()}
        ret = self.api.Execute_int64(feed)
        if ret != 0:
            raise RuntimeError(f"embedding execute failed: {ret}")
        outputs = self.api.FetchOutputs_float(["last_hidden_state"])
        hidden = np.asarray(outputs["last_hidden_state"], dtype=np.float32).reshape(1, SEQ_LEN, HIDDEN_SIZE)[0]
        return pool_and_normalize(hidden, encoded["attention_mask"])

    def embed_many(self, texts):
        return np.vstack([self.embed(text) for text in texts]).astype(np.float32)


FiboBgeSmallEmbedder = OnnxBgeSmallEmbedder
