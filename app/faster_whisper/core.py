import os
from io import StringIO
from threading import Lock
from typing import BinaryIO, Union

from app.utils import get_env_or_throw
import torch
import whisper
from faster_whisper import WhisperModel

from .utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT, WriteTwofoldJSON

model_name = get_env_or_throw("ASR_MODEL")
model_path = os.getenv("ASR_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))


if torch.cuda.is_available() == False:
    raise Exception("CUDA is not available. Please install CUDA and cuDNN.")

# More about available quantization levels is here:
#   https://opennmt.net/CTranslate2/quantization.html
device = "cuda"
model_quantization = get_env_or_throw("ASR_QUANTIZATION")

model = WhisperModel(
    model_size_or_path=model_name,
    device=device,
    compute_type=model_quantization,
    download_root=model_path,
)
model_lock = Lock()


def transcribe(
    audio,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None],
    vad_filter: Union[bool, None],
    word_timestamps: Union[bool, None],
    temperature: Union[float, None],
    output,
):
    options_dict = {"task": task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    if vad_filter:
        options_dict["vad_filter"] = True
    if word_timestamps:
        options_dict["word_timestamps"] = True
    if temperature:
        options_dict["temperature"] = temperature

    with model_lock:
        segments = []
        text = ""
        segment_generator, info = model.transcribe(audio, beam_size=5, **options_dict)
        for segment in segment_generator:
            segments.append(segment)
            text = text + segment.text
        result = {"language": info.language, "segments": segments, "text": text, "duration": info.duration}

    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file


def language_detection(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.pad_or_trim(audio)

    # detect the spoken language
    with model_lock:
        segments, info = model.transcribe(audio, beam_size=5)
        detected_lang_code = info.language

    return detected_lang_code


def write_result(result: dict, file: BinaryIO, output: Union[str, None]):
    if output == "srt":
        WriteSRT(ResultWriter).write_result(result, file=file)
    elif output == "vtt":
        WriteVTT(ResultWriter).write_result(result, file=file)
    elif output == "tsv":
        WriteTSV(ResultWriter).write_result(result, file=file)
    elif output == "json":
        WriteJSON(ResultWriter).write_result(result, file=file)
    elif output == "txt":
        WriteTXT(ResultWriter).write_result(result, file=file)
    elif output == "twofold-json":
        WriteTwofoldJSON(ResultWriter).write_result(result, file=file)
    else:
        return "Please select an output method!"
