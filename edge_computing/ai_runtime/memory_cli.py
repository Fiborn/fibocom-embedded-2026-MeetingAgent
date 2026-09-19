#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


BASE = Path("/home/fibo")
PROJECT_ROOT = BASE / "MeetingAgent"
LICENSE_DIR = BASE / "qcom_6490_license"
MODEL_ROOT = PROJECT_ROOT / "ai_models" / "llm_models"
DB_DIR = PROJECT_ROOT / "data" / "memory_db"
MEMORY_JSONL = DB_DIR / "memory.jsonl"
VECTORIZER_PATH = DB_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = DB_DIR / "tfidf_matrix.npz"
EMBEDDING_MATRIX_PATH = DB_DIR / "embedding_matrix.npy"
EMBEDDING_META_PATH = DB_DIR / "embedding_meta.json"
QWEN_MODEL = MODEL_ROOT / "Qwen3-0.6B" / "qwen3-0.6b_1.0.0_qcom_6490_qnn_2.28_dsp_537988b3eb4c6960d47c433c63fd32c2.fmodel"


def read_license_file(path):
    mode = "r" if str(path).endswith(".pem") else "rb"
    with open(path, mode) as f:
        return f.read()


def init_license():
    from fiboaisdk.api_aisdk_py import license_py as license_api

    ret = license_api.Init(
        read_license_file(LICENSE_DIR / "key1.pem"),
        read_license_file(LICENSE_DIR / "key2.pem"),
        read_license_file(LICENSE_DIR / "key3.pem"),
        read_license_file(LICENSE_DIR / "license.bin"),
    )
    print(f"[license] Init => {ret}")
    if ret != 0:
        raise RuntimeError(f"license init failed: {ret}")


def ensure_db_dir():
    DB_DIR.mkdir(parents=True, exist_ok=True)


def load_memories():
    if not MEMORY_JSONL.exists():
        return []
    memories = []
    with MEMORY_JSONL.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    memories.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"[memory] skip corrupt jsonl line {line_no}: {exc}")
    return memories


def append_memory(text, tags):
    ensure_db_dir()
    memories = load_memories()
    next_id = max([item.get("id", 0) for item in memories] + [0]) + 1
    item = {
        "id": next_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "text": text.strip(),
        "tags": tags or [],
    }
    with MEMORY_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return item


def build_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False, norm="l2")


def rebuild_tfidf_index(memories=None):
    ensure_db_dir()
    memories = load_memories() if memories is None else memories
    texts = [item["text"] for item in memories]
    if not texts:
        for path in [VECTORIZER_PATH, TFIDF_MATRIX_PATH]:
            if path.exists():
                path.unlink()
        return 0

    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    import joblib
    from scipy import sparse

    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(TFIDF_MATRIX_PATH, matrix)
    return len(memories)


def rebuild_embedding_index(memories=None):
    ensure_db_dir()
    memories = load_memories() if memories is None else memories
    texts = [item["text"] for item in memories]
    if not texts:
        for path in [EMBEDDING_MATRIX_PATH, EMBEDDING_META_PATH]:
            if path.exists():
                path.unlink()
        return 0

    from embedding_bge_small import FiboBgeSmallEmbedder

    with FiboBgeSmallEmbedder() as embedder:
        matrix = embedder.embed_many(texts)
    np.save(EMBEDDING_MATRIX_PATH, matrix)
    with EMBEDDING_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": "bge-small-zh-v1.5-dlc",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "memory_count": len(memories),
                "dim": int(matrix.shape[1]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return len(memories)


def rebuild_indexes(backend="auto"):
    memories = load_memories()
    tfidf_count = rebuild_tfidf_index(memories)
    embedding_count = None
    if backend in ("auto", "embedding"):
        try:
            embedding_count = rebuild_embedding_index(memories)
        except Exception as exc:
            if backend == "embedding":
                raise
            print(f"[memory] embedding rebuild skipped: {exc}")
    return tfidf_count, embedding_count


def load_tfidf_index():
    memories = load_memories()
    if not memories:
        return memories, None, None
    if not VECTORIZER_PATH.exists() or not TFIDF_MATRIX_PATH.exists():
        rebuild_tfidf_index(memories)
    import joblib
    from scipy import sparse

    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
    return memories, vectorizer, matrix


def search_tfidf(query, top_k):
    memories, vectorizer, matrix = load_tfidf_index()
    if not memories:
        return []
    query_vec = vectorizer.transform([query])
    scores = (matrix @ query_vec.T).toarray().ravel()
    ranked = np.argsort(scores)[::-1][:top_k]
    return _ranked_results(memories, scores, ranked, "tfidf")


def search_embedding(query, top_k):
    memories = load_memories()
    if not memories:
        return []
    if not EMBEDDING_MATRIX_PATH.exists():
        rebuild_embedding_index(memories)
    matrix = np.load(EMBEDDING_MATRIX_PATH)
    if matrix.shape[0] != len(memories):
        rebuild_embedding_index(memories)
        matrix = np.load(EMBEDDING_MATRIX_PATH)

    from embedding_bge_small import FiboBgeSmallEmbedder

    with FiboBgeSmallEmbedder() as embedder:
        query_vec = embedder.embed(query)
    scores = matrix @ query_vec
    ranked = np.argsort(scores)[::-1][:top_k]
    return _ranked_results(memories, scores, ranked, "embedding")


def _ranked_results(memories, scores, ranked, backend):
    results = []
    for index in ranked:
        item = dict(memories[int(index)])
        item["score"] = float(scores[int(index)])
        item["backend"] = backend
        results.append(item)
    return results


def search_memories(query, top_k, backend="auto"):
    if backend in ("embedding", "auto"):
        try:
            return search_embedding(query, top_k)
        except Exception as exc:
            if backend == "embedding":
                raise
            print(f"[memory] embedding search unavailable, fallback to TF-IDF: {exc}")
    return search_tfidf(query, top_k)


def chat_prompt(system, user):
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_llm_text(text):
    text = text or ""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start) + len("</think>")
        text = text[:start] + text[end:]
    for marker in ["<think>", "</think>", "<|im_end|>", "<|endoftext|>"]:
        text = text.replace(marker, "")
    text = text.strip()
    for prefix in ["Assistant:", "assistant:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip()


def ask_qwen(question, memories):
    from fiboaisdk.api_aisdk_py import api_nlp_py as nlp_api

    memory_text = "\n".join(
        f"- [{item['id']}] {item['text']} (score={item.get('score', 0):.3f}, backend={item.get('backend', 'unknown')})"
        for item in memories
        if item.get("score", 0) > 0
    )
    if not memory_text:
        memory_text = "没有检索到明显相关的长期记忆。"

    system = "你是一个运行在 SC171V3 开发板上的中文助手。你会优先参考长期记忆，但不要编造记忆里没有的信息。"
    user = (
        f"长期记忆：\n{memory_text}\n\n"
        f"用户问题：{question}\n\n"
        "请只根据长期记忆回答用户偏好或事实，最多两句话，不要输出思考过程。no_think"
    )
    prompt = chat_prompt(system, user)

    api = nlp_api.NLPAPI()
    ret = api.Init(str(QWEN_MODEL), "")
    print(f"[llm] Init => {ret}")
    if ret != 0:
        api.Release()
        raise RuntimeError(f"Qwen init failed: {ret}")

    result = nlp_api.ResultNlpText()
    ret = api.GenerateSync(prompt, result)
    print(f"[llm] GenerateSync => {ret}")
    text = clean_llm_text(getattr(result, "text", ""))
    api.Release()
    if ret != 0:
        raise RuntimeError(f"Qwen generate failed: {ret}")
    return text


def cmd_add(args):
    if not args.text.strip():
        raise ValueError("memory text cannot be empty")
    item = append_memory(args.text, args.tag)
    tfidf_count, embedding_count = rebuild_indexes(args.backend)
    print(json.dumps({"added": item, "tfidf_count": tfidf_count, "embedding_count": embedding_count}, ensure_ascii=False, indent=2))


def cmd_search(args):
    results = search_memories(args.query, args.top_k, args.backend)
    print(json.dumps({"query": args.query, "backend": args.backend, "results": results}, ensure_ascii=False, indent=2))


def cmd_ask(args):
    init_license()
    results = search_memories(args.query, args.top_k, args.backend)
    print("[memory] retrieved:")
    for item in results:
        print(f"- id={item['id']} score={item['score']:.3f} backend={item.get('backend')} text={item['text']}")
    answer = ask_qwen(args.query, results)
    print("[answer]")
    print(answer)


def cmd_list(args):
    memories = load_memories()
    for item in memories:
        tags = ",".join(item.get("tags", []))
        print(f"{item['id']}\t{item['created_at']}\t{tags}\t{item['text']}")
    print(f"[memory] count={len(memories)}")


def cmd_rebuild(args):
    tfidf_count, embedding_count = rebuild_indexes(args.backend)
    print(f"[memory] rebuilt TF-IDF index for {tfidf_count} memories")
    if embedding_count is not None:
        print(f"[memory] rebuilt embedding index for {embedding_count} memories")
    print(f"[memory] db => {DB_DIR}")


def cmd_embed_smoke(args):
    from embedding_bge_small import FiboBgeSmallEmbedder

    with FiboBgeSmallEmbedder() as embedder:
        vec = embedder.embed(args.text)
    print(json.dumps({"text": args.text, "dim": int(vec.shape[0]), "norm": float(np.linalg.norm(vec)), "preview": vec[:8].tolist()}, ensure_ascii=False, indent=2))


def add_backend_arg(parser):
    parser.add_argument("--backend", choices=["auto", "embedding", "tfidf"], default="auto")


def build_parser():
    parser = argparse.ArgumentParser(description="Local memory DB for SC171V3 with embedding and TF-IDF fallback.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add")
    add.add_argument("text")
    add.add_argument("--tag", action="append", default=[])
    add_backend_arg(add)
    add.set_defaults(func=cmd_add)

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    add_backend_arg(search)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask")
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int, default=5)
    add_backend_arg(ask)
    ask.set_defaults(func=cmd_ask)

    list_cmd = sub.add_parser("list")
    list_cmd.set_defaults(func=cmd_list)

    rebuild = sub.add_parser("rebuild")
    add_backend_arg(rebuild)
    rebuild.set_defaults(func=cmd_rebuild)

    smoke = sub.add_parser("embed-smoke")
    smoke.add_argument("text")
    smoke.set_defaults(func=cmd_embed_smoke)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
