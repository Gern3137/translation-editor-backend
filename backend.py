from fastapi import FastAPI, UploadFile, Form, HTTPException, Request
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
<omitted for brevity, same as before>
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
    return f"{SYSTEM_PROMPT}\n\nUser Style Guidance (follow if it does not conflict with the above):\n{user_prompt.strip()}\n\nHere is the list of English sentences to translate:\n{joined}\n\nReturn ONLY the JSON array."

# Helper function to remove unwanted sentences
def filter_skip_words(sentences, skip_words_str):
    skip_lines = [w.strip() for w in skip_words_str.splitlines() if w.strip()]
    if not skip_lines:
        return sentences
    return [
        s for s in sentences
        if not any(skip_word in s for skip_word in skip_lines)
    ]

@app.post("/upload/", response_model=TranslationResponse)
async def upload_file(
    file: UploadFile,
    user_prompt: str = Form(DEFAULT_USER_PROMPT),
    skip_words: str = Form("")
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    doc = fitz.open(stream=content, filetype="pdf")
    raw_text = "\n".join([page.get_text() for page in doc])

    cleaned = clean_text(raw_text)
    preprocessed = preprocess_sentences(cleaned)
    sentences = split_sentences(preprocessed)

    # ✅ Apply skip word filter here
    sentences = filter_skip_words(sentences, skip_words)

    prompt = build_prompt(sentences, user_prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
        start = output.find("[")
        end = output.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in GPT response.")

        json_str = output[start:end+1]
        raw_pairs = json.loads(json_str)

        pairs = [AlignedTranslation(**p) for p in raw_pairs]
        return {"pairs": pairs}

    except Exception as e:
        print("Upload endpoint error:", str(e))
        raise HTTPException(status_code=500, detail=f"Upload translation failed: {str(e)}")


@app.post("/paste/", response_model=TranslationResponse)
async def paste_text(text: str = Form(...), user_prompt: str = Form(DEFAULT_USER_PROMPT)):
    cleaned = clean_text(text)
    preprocessed = preprocess_sentences(cleaned)
    sentences = split_sentences(preprocessed)

    prompt = build_prompt(sentences, user_prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        output = response.choices[0].message.content.strip()
        start = output.find("[")
        end = output.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in GPT response.")

        json_str = output[start:end+1]
        raw_pairs = json.loads(json_str)

        pairs = [AlignedTranslation(**p) for p in raw_pairs]
        return {"pairs": pairs}

    except Exception as e:
        print("Paste endpoint error:", str(e))
        raise HTTPException(status_code=500, detail=f"Paste translation failed: {str(e)}")

@app.post("/retranslate/")
async def retranslate_block(request: Request):
    try:
        data = await request.json()
        sentences = data.get("sentences")
        user_prompt = data.get("user_prompt", DEFAULT_USER_PROMPT)

        if not sentences or not isinstance(sentences, list):
            raise HTTPException(status_code=400, detail="Invalid input")

        prompt = build_prompt(sentences, user_prompt)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        output = response.choices[0].message.content.strip()
        start = output.find("[")
        end = output.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in GPT response.")

        json_str = output[start:end+1]
        raw_pairs = json.loads(json_str)

        return {"pairs": raw_pairs}

    except Exception as e:
        print("Retranslate endpoint error:", str(e))
        raise HTTPException(status_code=500, detail=f"Re-translation failed: {str(e)}")
