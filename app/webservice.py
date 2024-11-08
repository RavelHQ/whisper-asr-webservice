from fastapi.security import APIKeyHeader
from fastapi import Security, HTTPException
from starlette.status import HTTP_403_FORBIDDEN

import importlib.metadata
import os
from os import path
from typing import Annotated, BinaryIO, Union
from urllib.parse import quote

import torch

import ffmpeg
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer
from app.utils import get_env_or_throw

from .faster_whisper.core import language_detection, transcribe, model_name, model_quantization

API_KEY = get_env_or_throw("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
workers = os.getenv("WORKERS")


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials")


ASR_ENGINE = os.getenv("ASR_ENGINE", "openai_whisper")
# if ASR_ENGINE == "faster_whisper":
#     from .faster_whisper.core import language_detection, transcribe
# else:
#     from .openai_whisper.core import language_detection, transcribe

SAMPLE_RATE = 16000
LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"] + "\n\nThis API requires an API key for authentication.",
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": projectMetadata["License"]},
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.get("/info", include_in_schema=False)
async def info():
    isCuda = torch.cuda.is_available()
    return {
        "ASR_ENGINE": ASR_ENGINE,
        "CUDA": isCuda,
        "model_name": model_name,
        "model_quantization": model_quantization,
        "workers": workers,
    }


@app.post("/asr", tags=["Endpoints"])
async def asr(
    api_key: str = Security(get_api_key),
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    temperature: Union[float, None] = Query(default=None, description="Sampling temperature"),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(default=False, description="Word level timestamps"),
    output: Union[str, None] = Query(default="twofold-json", enum=["txt", "vtt", "srt", "tsv", "json", "twofold-json"]),
):
    result = transcribe(
        load_audio(audio_file.file, encode),
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        temperature,
        output,
    )

    media_type = "application/json" if output == "json" else "text/plain"

    return StreamingResponse(
        result,
        media_type=media_type,
        headers={
            "Asr-Engine": ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    api_key: str = Security(get_api_key),
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code = language_detection(load_audio(audio_file.file, encode))
    return {"detected_language": tokenizer.LANGUAGES[detected_lang_code], "language_code": detected_lang_code}


def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
