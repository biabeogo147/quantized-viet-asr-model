import argparse
import importlib
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

warnings.filterwarnings("ignore", message=".*flash attention.*")


ASSETS_DIR = os.path.join("../assets")
VIBERT_CAPU_DIR = os.path.join(ASSETS_DIR, "vibert_capu")
DENORM_DIR = os.path.join(ASSETS_DIR, "vietnamese-punc-cap-denorm-v1")

DEFAULT_TEXTS = [
    "hôm nay là buổi nhậm chức của tôi phước thành",
    "chào các bạn hôm nay chúng ta cùng nhau đến với bài học deep learning phần số mười ba đáng lý bài này đã học từ ngày hai mốt tháng mười hai năm hai nghìn không trăm hai mươi lăm nhưng vì nghỉ tết chúng ta dời lịch đến ngày hai mươi hai tháng hai năm hai nghìn không trăm hai mươi sáu",
]


def load_inputs(args: argparse.Namespace) -> List[str]:
    if args.text:
        return [args.text.strip()]

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    return DEFAULT_TEXTS


def patch_vibert_capu_runtime(model_dir: str) -> None:
    workspace_root = os.path.abspath("..")
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    original_resize = PreTrainedModel.resize_token_embeddings
    if not getattr(PreTrainedModel.resize_token_embeddings, "_codex_vibert_patch", False):
        def patched_resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
            return original_resize(
                self,
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                mean_resizing=False,
            )

        patched_resize_token_embeddings._codex_vibert_patch = True
        PreTrainedModel.resize_token_embeddings = patched_resize_token_embeddings

    modeling_seq2labels = importlib.import_module("assets.vibert_capu.modeling_seq2labels")
    sys.modules.setdefault("modeling_seq2labels", modeling_seq2labels)

    if not hasattr(modeling_seq2labels.Seq2LabelsOutput, "__dataclass_fields__"):
        modeling_seq2labels.Seq2LabelsOutput = dataclass(modeling_seq2labels.Seq2LabelsOutput)


class VietnamesePuncCapDenormModel:
    def __init__(self, model_dir: str, device: int | None = None):
        self.model_dir = model_dir
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
        self.torch_device = torch.device(
            "cuda" if device is not None and device >= 0 and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.torch_device).eval()

    def restore(self, text: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_length=max_length)

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


class VibertCapuModel:
    def __init__(self, model_dir: str, device: str | None = None):
        self.model_dir = os.path.abspath(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        patch_vibert_capu_runtime(self.model_dir)

        try:
            from assets.vibert_capu.gec_model import GecBERTModel
        except Exception as exc:
            raise RuntimeError(
                "Khong the import assets.vibert_capu.gec_model tu local assets."
            ) from exc

        try:
            self.model = GecBERTModel(
                vocab_path=os.path.join(self.model_dir, "vocabulary"),
                model_paths=[self.model_dir],
                device=self.device,
                split_chunk=True,
            )
        except Exception as exc:
            raise exc

    def restore(self, text: str) -> str:
        return self.model(text)[0].strip()


def print_result_block(title: str, original: str, restored: str, elapsed: float) -> None:
    print(f"\n========== {title} ==========")
    print("Input        :", original)
    print("Output       :", restored)
    print("Latency      :", round(elapsed, 3), "s")
    print("================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test 2 punctuation/capitalization models trong assets."
    )
    parser.add_argument("--text", help="Text dau vao can test.")
    parser.add_argument("--text-file", help="File txt, moi dong la 1 sample.")
    parser.add_argument(
        "--skip-vibert",
        action="store_true",
        help="Bo qua model vibert-capu.",
    )
    parser.add_argument(
        "--skip-denorm",
        action="store_true",
        help="Bo qua model vietnamese-punc-cap-denorm-v1.",
    )
    args = parser.parse_args()

    inputs = load_inputs(args)
    if not inputs:
        raise ValueError("Khong co text dau vao de test.")

    use_cuda = torch.cuda.is_available()
    hf_device = 0 if use_cuda else -1

    denorm_model = None
    vibert_model = None

    if not args.skip_denorm:
        denorm_model = VietnamesePuncCapDenormModel(
            model_dir=DENORM_DIR,
            device=hf_device,
        )

    if not args.skip_vibert:
        try:
            vibert_model = VibertCapuModel(model_dir=VIBERT_CAPU_DIR)
        except RuntimeError as exc:
            print("\n[WARN] Khong the load vibert-capu:", exc)

    for idx, text in enumerate(inputs, start=1):
        print(f"================ SAMPLE {idx} ================")

        if denorm_model is not None:
            t0 = time.time()
            denorm_output = denorm_model.restore(text)
            t1 = time.time()
            print_result_block(
                "VIETNAMESE PUNC CAP DENORM RESULT",
                text,
                denorm_output,
                t1 - t0,
            )

        if vibert_model is not None:
            t0 = time.time()
            vibert_output = vibert_model.restore(text)
            t1 = time.time()
            print_result_block(
                "VIBERT CAPU RESULT",
                text,
                vibert_output,
                t1 - t0,
            )


if __name__ == "__main__":
    main()
