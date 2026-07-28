"""Serve one OpenVLA LIBERO action per HTTP request on CPU.

This beta-only probe uses a LIBERO-specific frozen checkpoint outside its
training simulator. It is not a reproduction of the Harness VLA paper.
"""

import argparse
import base64
import io
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


app = Flask(__name__)
processor = None
model = None
unnorm_key = None


def center_crop(image, crop_scale=0.9):
    width, height = image.size
    scale = crop_scale ** 0.5
    crop_width = max(1, round(width * scale))
    crop_height = max(1, round(height * scale))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(
        (width, height), Image.Resampling.LANCZOS
    )


def decode_image(payload):
    if not isinstance(payload, dict):
        raise ValueError("image must be a structured object")
    if payload.get("encoding") != "base64" or payload.get("media_type") != "image/png":
        raise ValueError("image must be a base64 PNG")
    binary = base64.b64decode(payload.get("data", ""), validate=True)
    return Image.open(io.BytesIO(binary)).convert("RGB")


@app.get("/health")
def health():
    return jsonify({"status": "ready", "checkpoint": model.name_or_path, "device": "cpu"})


@app.post("/predict")
def predict():
    started = time.perf_counter()
    try:
        payload = request.get_json(force=True)
        image = center_crop(decode_image(payload.get("image")))
        instruction = str(payload.get("prompt", "")).strip()
        mode = str(payload.get("mode", "")).strip()
        if not instruction:
            raise ValueError("prompt must be non-empty")
        task = f"{instruction} Perform only the local {mode} contact phase."
        prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
        inputs = processor(prompt, image).to("cpu", dtype=torch.float32)
        with torch.inference_mode():
            action = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )
        return jsonify({
            "action": np.asarray(action, dtype=float).tolist(),
            "model_seconds": time.perf_counter() - started,
            "unnorm_key": unnorm_key,
        })
    except Exception as error:
        return jsonify({"error": f"{type(error).__name__}: {error}"}), 400


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="openvla/openvla-7b-finetuned-libero-object",
    )
    parser.add_argument("--unnorm-key", default="libero_object")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    global model, processor, unnorm_key
    args = parse_args()
    torch.set_num_threads(max(1, min(24, (torch.get_num_threads() or 1))))
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    unnorm_key = args.unnorm_key
    if unnorm_key not in model.norm_stats:
        raise ValueError(
            f"unnorm key {unnorm_key!r} absent; available: {sorted(model.norm_stats)}"
        )
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
