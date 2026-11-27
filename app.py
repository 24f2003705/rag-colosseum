"""RAG Colosseum service: hybrid dense + lexical retrieval, extractive answers,
abstention, prompt-injection sanitising, multi-passage synthesis."""
import math, os, re, threading
from collections import Counter
from functools import lru_cache
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
PASSAGES = [l.strip() for l in open(os.path.join(HERE, "passages.txt"), encoding="utf-8") if l.strip()]
assert len(PASSAGES) == 50, len(PASSAGES)

# ----------------------------------------------------------------- text utils
STOP = set("""a an the of for in on at to by with from into onto over under about as is are was were be been being am
do does did done have has had having can could should would will shall may might must what which who whom whose when where why how
many much most more less least any some each every all both either neither this that these those it its they them their there here
and or but nor so than then also very just only not no yes i me my we our you your he she his her use uses used using work works
default value values tell give please explain describe state specify specified configured configure number amount level
difference differences differ differs compare compared comparison versus vs one two per via within without across during before after between among while whether if because since""".split())
CONNECT = set("and or vs versus compare compared comparison both difference differ differs between together combined while whereas".split())
NUMERIC_CUES = set("port timeout interval threshold size sizes quorum many much long often frequently time duration length bits bit limit count".split())

def stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "edly", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + ("y" if suf == "ies" else "")
            break
    if len(w) > 4 and w[-1] == w[-2] and w[-1] in "tnpdgmb":
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w

def toks(text: str) -> List[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]

def content_toks(text: str) -> List[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and stem(t) not in STOP]

def sentences(p: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]

# ----------------------------------------------------------------- index
P_TOKS = [set(content_toks(p)) for p in PASSAGES]
P_SENTS = [sentences(p) for p in PASSAGES]
TOPIC = [set(content_toks(sl[0])) for sl in P_SENTS]
VOCAB = set().union(*P_TOKS)
ALIASES = {"sgd": "stochastic gradient descent", "wal": "write-ahead logging", "lsm": "log-structured merge trees",
           "mvcc": "multi-version concurrency control", "mlp": "multi-layer perceptrons", "2pc": "two-phase commit",
           "3pc": "three-phase commit", "hlc": "hybrid logical clocks", "waf": "web application firewall",
           "ecc": "elliptic curve cryptography", "layernorm": "layer normalization", "batchnorm": "batch normalization",
           "http2": "HTTP/2", "http3": "HTTP/3", "tls1.3": "TLS 1.3", "tls1.2": "TLS 1.2", "dht": "consistent hashing",
           "hrw": "rendezvous hashing", "resnet": "ResNet residual convolutional networks", "gpt": "GPT decoder",
           "ssl": "TLS", "mdns": "Multicast DNS", "tcp": "TCP Transmission Control Protocol", "pkce": "PKCE",
           "jwt": "JWT JSON Web Tokens", "aes": "AES Advanced Encryption Standard"}
def expand_aliases(q: str) -> str:
    return re.sub(r"[A-Za-z0-9\.]+", lambda m: ALIASES.get(m.group(0).lower().rstrip("."), m.group(0)) + ("." if m.group(0).endswith(".") else ""), q)
DF = Counter(t for s in P_TOKS for t in s)
N = len(PASSAGES)
IDF = {t: math.log((N + 1) / (df + 0.5)) + 1 for t, df in DF.items()}
VOC_LIST = sorted(VOCAB)
VIDX = {t: i for i, t in enumerate(VOC_LIST)}

def tfidf_vec(tokens: List[str]) -> np.ndarray:
    v = np.zeros(len(VOC_LIST), dtype=np.float32)
    for t, c in Counter(tokens).items():
        if t in VIDX:
            v[VIDX[t]] = (1 + math.log(c)) * IDF[t]
    n = np.linalg.norm(v)
    return v / n if n else v

P_MAT = np.stack([tfidf_vec([t for t in content_toks(p)]) for p in PASSAGES])

# optional dense model
DENSE = None
P_EMB = None
_lock = threading.Lock()
if os.environ.get("DISABLE_DENSE") != "1":
    try:
        from sentence_transformers import SentenceTransformer
        DENSE = SentenceTransformer(os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        P_EMB = DENSE.encode(PASSAGES, normalize_embeddings=True, convert_to_numpy=True)
    except Exception as e:  # falls back to lexical only
        print("dense model unavailable, lexical only:", e)
        DENSE = None

@lru_cache(maxsize=512)
def dense_q(text: str):
    with _lock:
        return DENSE.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]

# ----------------------------------------------------------------- injection
INJ = re.compile(
    r"(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instruction|prompt|rule|context|above|previous|prior|system|guideline)"
    r"|system prompt|developer message|new instructions?|you are now|act as|pretend|jailbreak|\bDAN\b"
    r"|(reveal|print|show|leak|output|repeat)\b.{0,30}\b(prompt|instruction|secret|api key|password|system)"
    r"|(always|instead|just|only)\s+(answer|respond|reply|say|output|return)"
    r"|(answer|respond|reply|say|output|return|write)\s+(with\s+)?[\"'“‘]?(pwned|hacked|yes|no|null|none|i don'?t know|unknown)\b"
    r"|note to (the )?(system|assistant|model|ai)|<\|.*?\|>|\[/?(system|inst)\]|^\s*(system|assistant)\s*:|###",
    re.I,
)

def sanitize(q: str):
    parts = re.split(r"(?<=[.?!;\n])\s+|\n+", q)
    kept, injected = [], False
    for part in parts:
        m = INJ.search(part)
        if m:
            injected = True
            before = part[: m.start()].strip(" ,;:-—(")
            if len(content_toks(before)) >= 2:
                kept.append(before)
            continue
        kept.append(part)
    return " ".join(kept).strip(), injected

# ----------------------------------------------------------------- core
def spec_terms(question: str, qt: List[str]) -> List[str]:
    caps = set()
    for i, w in enumerate(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+\.#/]*", question)):
        if i > 0 and (w[0].isupper() or any(c.isdigit() for c in w) or w.isupper()):
            caps.update(stem(x) for x in re.findall(r"[a-z0-9]+", w.lower()))
    topics = set().union(*TOPIC)
    return [t for t in set(qt) if t in DF and DF[t] <= 3 and (t in topics or t in caps)]

def is_comparison(question: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", question.lower())) & CONNECT) or question.count("?") > 1

def covers(i: int, t: str, strict: bool = False) -> bool:
    if not strict:
        return t in P_TOKS[i]
    owners = [j for j in range(N) if t in TOPIC[j]]
    return (i in owners) if owners else (t in P_TOKS[i])

def retrieve(question: str):
    qt = content_toks(question)
    spec = spec_terms(question, qt)
    lex = P_MAT @ tfidf_vec(qt)
    if DENSE is not None:
        d = np.clip(P_EMB @ dense_q(question), 0, 1)
        base = 0.55 * d + 0.45 * np.clip(lex * 1.6, 0, 1)
    else:
        d = None
        base = np.clip(lex * 1.6, 0, 1)
    if spec:
        cov = np.array([sum(1.0 if covers(i, t, is_comparison(question)) else (0.3 if t in P_TOKS[i] else 0.0) for t in spec) / len(spec) for i in range(N)])
        base = base + 0.25 * cov
    return np.clip(base, 0, 1), qt, spec, d

def best_sentence(pidx: int, attr: List[str], numeric: bool, question: str) -> str:
    best, bs = None, -1.0
    for s in P_SENTS[pidx]:
        st = set(content_toks(s))
        sc = sum(IDF.get(t, 1) for t in attr if t in st)
        if numeric and re.search(r"\d", s):
            sc += 1.0
        if sc > bs:
            best, bs = s, sc
    if not attr:
        return P_SENTS[pidx][0]
    return best

ABSTAIN = {"answer": None, "sources": [], "scores": [], "confidence": 0.1}

def answer_question(question: str, top_k: int):
    clean, injected = sanitize(question)
    clean = expand_aliases(clean)
    qt = content_toks(clean)
    if not qt:
        return dict(ABSTAIN)
    # unknown-term check (off-topic / unknown entities)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+\.#/]*", clean)
    unknown_caps = 0
    for i, w in enumerate(words):
        lw = w.lower().strip(".-/")
        if lw in STOP or not lw:
            continue
        parts = [stem(x) for x in re.findall(r"[a-z0-9]+", lw)]
        if all(p in VOCAB or p in STOP for p in parts):
            continue
        if (i > 0 and w[0].isupper()) or any(c.isdigit() for c in w) and not w.isdigit() or w.isupper():
            unknown_caps += 1
    unknown = [t for t in qt if t not in VOCAB]
    if unknown_caps >= 1 or len(unknown) >= max(1, math.ceil(0.5 * len(qt))):
        return dict(ABSTAIN)

    scores, qt, spec, d = retrieve(clean)
    order = list(np.argsort(-scores))
    top = order[0]
    if scores[top] < 0.12:
        return dict(ABSTAIN)

    # pairing check: all rare entity terms must co-occur in one passage unless a comparison is asked
    comparison = is_comparison(clean)
    if len(spec) >= 2 and not comparison:
        if not any(all(t in P_TOKS[i] for t in spec) for i in range(N)):
            return dict(ABSTAIN)

    # choose passages that cover the rare terms (synthesis)
    chosen = [top]
    cmp_ = is_comparison(clean)
    missing = [t for t in spec if not covers(top, t, cmp_)]
    while missing and len(chosen) < 3:
        cand = [i for i in order if i not in chosen and any(covers(i, t, cmp_) for t in missing)]
        if not cand:
            break
        chosen.append(cand[0])
        missing = [t for t in missing if not covers(cand[0], t, cmp_)]
    if missing and not comparison:
        return dict(ABSTAIN)

    attr = [t for t in set(qt) if t in VOCAB and t not in spec]
    GENERIC = {"siz", "length", "limit", "count", "valu", "amount", "number", "type", "kind", "name", "mean", "purpos"}
    rare = [t for t in attr if DF[t] <= 3 and t not in GENERIC]
    if rare:
        pool = set().union(*[P_TOKS[i] for i in chosen])
        if sum(t in pool for t in rare) / len(rare) < 0.5:
            return dict(ABSTAIN)
    numeric = bool(set(re.findall(r"[a-z]+", clean.lower())) & NUMERIC_CUES)

    parts, matched_any = [], False
    for pi in chosen:
        sent = best_sentence(pi, attr, numeric, clean)
        st = set(content_toks(sent))
        hit = any(t in st for t in attr)
        if attr and not hit and DENSE is not None:
            hit = float(np.clip(dense_q(clean) @ DENSE.encode([sent], normalize_embeddings=True)[0], 0, 1)) >= 0.55
        matched_any = matched_any or hit or not attr
        if hit or not attr or len(chosen) > 1:
            parts.append(sent)
    if attr and not matched_any:
        return dict(ABSTAIN)  # entity known, requested attribute not in the KB

    answer = " ".join(dict.fromkeys(parts))
    k = max(1, min(int(top_k or 3), 10))
    ranked = list(order)
    for pi in reversed(chosen):  # make sure covering passages lead the list
        ranked.remove(pi); ranked.insert(0, pi)
    ranked = sorted(ranked, key=lambda i: -scores[i])
    for pi in chosen:  # keep chosen passages inside the top-k window
        if pi not in ranked[:k]:
            ranked = [pi] + [i for i in ranked if i != pi]
    ranked = sorted(ranked[:k], key=lambda i: -scores[i])
    sc = [round(float(min(1.0, max(0.0, scores[i]))), 4) for i in ranked]
    conf = round(float(min(0.95, 0.65 + 0.3 * min(1.0, scores[top]))), 3)
    return {"answer": answer, "sources": [int(i) for i in ranked], "scores": sc, "confidence": conf}

# ----------------------------------------------------------------- API
app = FastAPI(title="RAG Colosseum")
@app.middleware("http")
async def always_cors(request, call_next):
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str = ""
    top_k: Optional[int] = 3

@app.get("/")
def health():
    return {"status": "ok", "passages": N, "dense": DENSE is not None}

@app.post("/query")
def query(q: Query):
    try:
        return answer_question(q.question or "", q.top_k or 3)
    except Exception:
        return dict(ABSTAIN)
