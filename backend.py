from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os
import json
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Set allowed CORS origins based on environment
ENV = os.getenv("ENVIRONMENT", "development")
if ENV == "production":
    allowed_origins = ["https://translation-editor-frontend.vercel.app"]
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AlignedTranslation(BaseModel):
    original: str
    translated: str

class TranslationResponse(BaseModel):
    pairs: List[AlignedTranslation]

DEFAULT_USER_PROMPT = """
以下の英語の文章を日本語に翻訳してください。翻訳にあたっては、次の点に注意してください：
内容の忠実性：原文の情報や意味を一切変えず、取りこぼしがないようにしてください。
文脈の理解：文章全体の流れや背景を正確に理解し、適切な訳語を選んでください。
ニュアンスの再現：言葉の微妙なニュアンスや感情、文化的な要素を可能な限り日本語で再現してください。
自然な日本語表現：直訳にならないように注意し、日本語として自然で読みやすい文章に仕上げてください。
専門用語や固有名詞の統一性：専門用語や固有名詞が繰り返し出てくる場合、一貫した訳語を使用してください。
トーンとスタイルの維持：原文のトーン（フォーマル、インフォーマル、ユーモラスなど）やスタイルを維持してください。
文法と句読点の正確さ：日本語の文法規則や句読点の使い方に従ってください。
文化的背景の考慮：文化的な背景や慣用表現を日本の読者が理解しやすい形で翻訳してください。
読者の視点：日本語話者が理解しやすい表現を心がけ、必要に応じて補足説明を加えても構いません。
不明な箇所の処理：曖昧な表現や理解が難しい部分があれば、適切に解釈して翻訳するか、注釈を加えてください。
以上の点
"""

SYSTEM_PROMPT = """
You are an experienced English-to-Japanese translator.

System Instructions (must be strictly followed):
- Translate one English sentence into exactly one Japanese sentence.
- Do NOT split or merge English sentences.
- Translate '.' and '!' as '。'
- Do NOT use '。' for other punctuation like ':', '(', etc.
- Return ONLY a valid JSON array like:
[{"original": "Sentence.", "translated": "文。"}]
"""

def clean_text(raw_text):
    lines = raw_text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or line.isdigit():
            continue
        if line.isupper() and len(line.split()) < 6:
            continue
        if len(line) <= 2 and all(char in "'\"“”’‘-–—" for char in line):
            continue
        cleaned.append(line)
    return " ".join(cleaned)

def preprocess_sentences(text):
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text):
    return re.findall(r"[^\u3002\uff01\uff1f.?!]+[\u3002\uff01\uff1f.?!]", text)

def build_prompt(sentences, user_prompt):
    joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return f"""
User Style Guidance (follow if it does not conflict with the system instructions):
{user_prompt.strip()}

Translate the following English sentences into Japanese:
{joined}

Return ONLY the JSON array in this format:
[{{"original": "...", "translated": "..."}}]
""".strip()

def filter_skip_words(sentences, skip_words_str):
    skip_lines = [w.strip() for w in skip_words_str.splitlines() if w.strip()]
    if not skip_lines:
        return sentences
    return [s for s in sentences if not any(skip_word in s for skip_word in skip_lines)]

@app.post("/upload/", response_model=TranslationResponse)
async def upload_file(
    file: UploadFile,
user_prompt: str = Form(default=None),
skip_words: str = Form(default=None)

):
    if not user_prompt.strip():
        user_prompt = DEFAULT_USER_PROMPT

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    doc = fitz.open(stream=content, filetype="pdf")

    blocks = []
    for page in doc:
        for b in page.get_text("blocks"):
            blocks.append((page.number, b[1], b[4]))
    blocks = sorted(blocks, key=lambda b: (b[0], b[1]))
    raw_text = "\n".join(b[2] for b in blocks)

    lines = raw_text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if all(c in "'\"“”’‘-–—*•" for c in line.strip()):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    preprocessed = preprocess_sentences(cleaned)
    sentences = split_sentences(preprocessed)
    sentences = filter_skip_words(sentences, skip_words)

    print(f"📄 Total sentences extracted: {len(sentences)}")

    CHUNK_SIZE = 30
    all_pairs = []

    try:
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = sentences[i:i + CHUNK_SIZE]
            prompt = build_prompt(chunk, user_prompt)

            print(f"🧩 Sending chunk {i // CHUNK_SIZE + 1} with {len(chunk)} sentences...")

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            output = response.choices[0].message.content.strip()
            print("📥 GPT raw output (start):", output[:300])

            start = output.find("[")
            end = output.rfind("]")
            if start == -1 or end == -1:
                print("❌ GPT returned invalid JSON")
                print("❌ Full output:", output)
                raise ValueError("No JSON array found in GPT response.")

            json_str = output[start:end+1]
            raw_pairs = json.loads(json_str)

            pairs = [AlignedTranslation(**p) for p in raw_pairs]
            all_pairs.extend(pairs)

        print(f"✅ Finished! Translated {len(all_pairs)} sentence pairs.")
        return {"pairs": all_pairs}

    except Exception as e:
        print("🔥 Upload endpoint error:", str(e))
        raise HTTPException(status_code=500, detail=f"Upload translation failed: {str(e)}")