import os
import tempfile
import asyncio
import uuid
from typing import List, Optional
import warnings  # <-- add this

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # from qwen-vl-utils

from faster_whisper import WhisperModel
# from pyannote.audio import Pipeline

# import soundfile as sf  # <-- add this (you have soundfile in deps)


warnings.filterwarnings(
    "ignore",
    message="torchcodec is not installed correctly so built-in audio decoding will fail",
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------
# Load env / config
# ---------------------------------------------------------------------

load_dotenv()

QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH")
if not QWEN_MODEL_PATH:
    raise RuntimeError("QWEN_MODEL_PATH must be set in .env")

QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "512"))
QWEN_MAX_BATCH_SIZE = int(os.getenv("QWEN_MAX_BATCH_SIZE", "4"))
QWEN_BATCH_MAX_LATENCY_MS = int(os.getenv("QWEN_BATCH_MAX_LATENCY_MS", "20"))

VIDEO_DEFAULT_FPS = float(os.getenv("VIDEO_DEFAULT_FPS", "2.0"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Qwen batching engine
# ---------------------------------------------------------------------


class QwenRequest(BaseModel):
    id: str
    prompt_text: str
    system_prompt: Optional[str] = None
    # One of these is set depending on modality
    image_paths: Optional[List[str]] = None
    video_path: Optional[str] = None
    fps: Optional[float] = None  # per-request FPS for video
    max_new_tokens: int = QWEN_MAX_NEW_TOKENS
    temperature: float = 0.2
    top_p: float = 0.9


class QwenEngine:
    def __init__(self):
        # AutoProcessor handles both text + vision; use it instead of AutoTokenizer
        self.processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_PATH,
            use_fast=True,
        )

        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            tok = getattr(self.processor, "text_tokenizer", None)

        if tok is not None:
            # For decoder-only models you MUST left-pad
            tok.padding_side = "left"

            # Often pad_token is not set; reuse eos_token
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": DEVICE},
        )
        self.model.eval()

        self.queue: "asyncio.Queue[tuple[QwenRequest, asyncio.Future]]" = asyncio.Queue()
        self.max_batch_size = QWEN_MAX_BATCH_SIZE
        self.max_latency_ms = QWEN_BATCH_MAX_LATENCY_MS

        self._worker_task = asyncio.create_task(self._batch_loop())

    @staticmethod
    def _process_output(text: str) -> str:
        splitted = text.strip().split("assistant", maxsplit=1)

        if len(splitted) > 1:
            return splitted[1].strip()
        return splitted[0].strip()

    async def generate(self, req: QwenRequest) -> str:
        """Public API used by FastAPI handlers."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self.queue.put((req, fut))
        return await fut

    async def _batch_loop(self):
        """Continuously drain queue and run batched inference."""
        while True:
            req, fut = await self.queue.get()
            batch = [(req, fut)]

            # Small wait to accumulate more requests
            try:
                await asyncio.sleep(self.max_latency_ms / 1000.0)
            except asyncio.CancelledError:
                break

            while len(batch) < self.max_batch_size:
                try:
                    r, f = self.queue.get_nowait()
                    batch.append((r, f))
                except asyncio.QueueEmpty:
                    break

            try:
                outputs = await asyncio.get_running_loop().run_in_executor(
                    None, self._run_batch_sync, [r for r, _ in batch]
                )
                # outputs: List[str] aligned with batch
                for (_, fut), out in zip(batch, outputs):
                    if not fut.done():
                        fut.set_result(out)
            except Exception as e:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def _build_messages(self, req: QwenRequest):
        """Build Qwen chat messages format with optional image/video."""
        content = []
        if req.image_paths:
            for p in req.image_paths:
                # qwen-vl-utils expects "image" with file:// URI or PIL Image
                content.append({"type": "image", "image": f"file://{p}"})

        if req.video_path:
            # qwen-vl-utils expects "video" with file:// URI
            fps = req.fps if req.fps is not None else VIDEO_DEFAULT_FPS
            content.append(
                {
                    "type": "video",
                    "video": f"file://{req.video_path}",
                    "fps": float(fps),
                }
            )

        content.append({"type": "text", "text": req.prompt_text})

        messages = []
        if req.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": req.system_prompt}],
                }
            )
        messages.append({"role": "user", "content": content})
        return messages

    def _run_batch_sync(self, requests: List[QwenRequest]) -> List[str]:
        # 1) Build ChatML-style conversations
        conversations = [self._build_messages(r) for r in requests]

        # 2) Text prompts with vision placeholders
        texts = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        # apply_chat_template returns str for single conv, list[str] for batch
        if isinstance(texts, str):
            texts = [texts]

        # 3) Let qwen-vl-utils load/crop images & videos
        images, videos, video_kwargs = process_vision_info(
            conversations,
            return_video_kwargs=True,
        )

        # 4) Pack everything with AutoProcessor (handles padding & THW packing)
        inputs = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        inputs = inputs.to(DEVICE)

        # Use the maximum max_new_tokens across the batch, then trim per request
        max_new_tokens = max(r.max_new_tokens for r in requests)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=requests[0].temperature > 0,
                temperature=requests[0].temperature,
                top_p=requests[0].top_p,
            )

        # 5) Strip the prompt part per sample using attention_mask
        generated_token_seqs = []
        attn_mask = inputs["attention_mask"]

        for i, req in enumerate(requests):
            # true prompt length = number of non-pad tokens
            input_len = int(attn_mask[i].sum().item())
            gen_tokens = output_ids[i, input_len:]

            # respect per-request max_new_tokens
            if gen_tokens.shape[0] > req.max_new_tokens:
                gen_tokens = gen_tokens[: req.max_new_tokens]

            generated_token_seqs.append(gen_tokens)

        # 6) Decode batched
        texts_out = self.processor.batch_decode(
            generated_token_seqs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [self.__class__._process_output(t) for t in texts_out]


# ---------------------------------------------------------------------
# Audio engine: Whisper + pyannote diarization
# ---------------------------------------------------------------------


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    content: str


class AudioEngine:
    def __init__(self):
        # Whisper runs on GPU if available; CPU otherwise
        self.whisper = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=DEVICE,
            compute_type="bfloat16" if DEVICE == "cuda" else "float32",
            download_root=WHISPER_MODEL_PATH,
            local_files_only=True,
        )

        # Simple async queue (no batching yet – per file)
        self.queue: "asyncio.Queue[tuple[str, asyncio.Future]]" = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._loop())

    async def transcribe(self, audio_path: str) -> List[TranscriptionSegment]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self.queue.put((audio_path, fut))
        return await fut

    async def _loop(self):
        while True:
            path, fut = await self.queue.get()
            try:
                res = await asyncio.get_running_loop().run_in_executor(None, self._process_one, path)
                if not fut.done():
                    fut.set_result(res)
            except Exception as e:
                if not fut.done():
                    fut.set_exception(e)

    def _process_one(self, audio_path: str) -> List[TranscriptionSegment]:
        # faster-whisper uses ffmpeg internally, so mp3 is fine
        segments, info = self.whisper.transcribe(
            audio_path,
            beam_size=5,
            language=None,  # auto-detect language
            condition_on_previous_text=False,
            # task="translate",         # <- uncomment if you want EN translation
        )

        results: List[TranscriptionSegment] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            results.append(
                TranscriptionSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    content=text,
                )
            )
        return results


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(title="Internal Multimodal API")

qwen_engine: Optional[QwenEngine] = None
audio_engine: Optional[AudioEngine] = None


@app.on_event("startup")
async def _startup():
    global qwen_engine, audio_engine
    qwen_engine = QwenEngine()
    audio_engine = AudioEngine()


# ---------- Schemas ----------


class TextRequest(BaseModel):
    system: Optional[str] = None
    prompt: str
    max_new_tokens: int = QWEN_MAX_NEW_TOKENS
    temperature: float = 0.2
    top_p: float = 0.9


class TextResponse(BaseModel):
    id: str
    content: str


# class AudioResponse(BaseModel):
#     segments: List[SpeakerSegment]


class AudioResponse(BaseModel):
    segments: List[TranscriptionSegment]


# ---------- Text-only endpoint ----------


@app.post("/v1/qwen/text", response_model=TextResponse)
async def qwen_text(req: TextRequest):
    if qwen_engine is None:
        raise HTTPException(status_code=500, detail="Qwen engine not initialized")
    rid = str(uuid.uuid4())
    qreq = QwenRequest(
        id=rid,
        prompt_text=req.prompt,
        system_prompt=req.system,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    out = await qwen_engine.generate(qreq)
    return TextResponse(id=rid, content=out)


# ---------- Image endpoint ----------


@app.post("/v1/qwen/image", response_model=TextResponse)
async def qwen_image(
    image_file: UploadFile = File(...),
    prompt: str = Form(...),
    system: Optional[str] = Form(None),
    max_new_tokens: int = Form(QWEN_MAX_NEW_TOKENS),
    temperature: float = Form(0.2),
    top_p: float = Form(0.9),
):
    if qwen_engine is None:
        raise HTTPException(status_code=500, detail="Qwen engine not initialized")

    rid = str(uuid.uuid4())
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, image_file.filename)
        with open(img_path, "wb") as f:
            f.write(await image_file.read())

        qreq = QwenRequest(
            id=rid,
            prompt_text=prompt,
            system_prompt=system,
            image_paths=[img_path],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        out = await qwen_engine.generate(qreq)
        # tempdir auto-cleaned
    return TextResponse(id=rid, content=out)


# ---------- Video endpoint ----------


@app.post("/v1/qwen/video", response_model=TextResponse)
async def qwen_video(
    video_file: UploadFile = File(...),
    prompt: str = Form(...),
    system: Optional[str] = Form(None),
    fps: Optional[float] = Form(None),
    max_new_tokens: int = Form(QWEN_MAX_NEW_TOKENS),
    temperature: float = Form(0.2),
    top_p: float = Form(0.9),
):
    if qwen_engine is None:
        raise HTTPException(status_code=500, detail="Qwen engine not initialized")

    rid = str(uuid.uuid4())
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, video_file.filename)
        with open(vid_path, "wb") as f:
            f.write(await video_file.read())

        # You can control FPS via env. For now we just rely on VIDEO_DEFAULT_FPS
        qreq = QwenRequest(
            id=rid,
            prompt_text=prompt,
            system_prompt=system,
            video_path=vid_path,
            fps=fps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        out = await qwen_engine.generate(qreq)
        # tempdir auto-cleaned
    return TextResponse(id=rid, content=out)


# ---------- Audio (mp3) endpoint: transcription only ----------


@app.post("/v1/audio/transcribe", response_model=AudioResponse)
async def audio_transcribe(
    audio_file: UploadFile = File(...),
):
    if audio_engine is None:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, audio_file.filename)
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())

        segments = await audio_engine.transcribe(audio_path)
    return AudioResponse(segments=segments)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})
