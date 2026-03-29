import os, sys, io, warnings, re, time
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
import argparse
import dateparser


class ZipformerASR:
    def __init__(self, model_dir, use_int8=False):
        self.model_dir = model_dir
        self.use_int8 = use_int8
        self.token_path = os.path.join(model_dir, "tokens.txt")

        # --- Silence ONNX + warning ---
        os.environ["KMP_WARNINGS"] = "0"
        os.environ["ORT_LOG_SEVERITY_LEVEL"] = "4"
        os.environ["ORT_LOG_VERBOSITY_LEVEL"] = "0"
        os.environ["ORT_DISABLE_REDUCED_LOGGING"] = "1"
        os.environ["ORT_SUPPRESS_CONSOLE_OUTPUT"] = "1"
        warnings.filterwarnings("ignore")

        suffix = ".int8" if use_int8 else ""
        self.encoder_path = os.path.join(model_dir, f"encoder-epoch-20-avg-1{suffix}.onnx")
        self.decoder_path = os.path.join(model_dir, f"decoder-epoch-20-avg-1{suffix}.onnx")
        self.joiner_path  = os.path.join(model_dir, f"joiner-epoch-20-avg-1{suffix}.onnx")

        self.encoder_sess = self._silent_session(self.encoder_path)
        self.decoder_sess = self._silent_session(self.decoder_path)
        self.joiner_sess  = self._silent_session(self.joiner_path)

    def _silent_session(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()

        so = ort.SessionOptions()
        so.log_severity_level = 4
        so.log_verbosity_level = 0

        sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        sys.stdout, sys.stderr = old_stdout, old_stderr
        return sess

    def _load_features(self, wav_path):
        waveform, sr = torchaudio.load(wav_path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=80,
            f_min=20,
            f_max=8000,
            power=2.0
        )(waveform)

        log_mel = torch.clamp(mel, min=1e-10).log()
        return log_mel.squeeze(0).transpose(0, 1).numpy().astype(np.float32)

    def _decode_tokens(self, result):
        tokens = []
        with open(self.token_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                token = parts[0] if len(parts) == 2 else parts[-1]
                tokens.append(token)

        text = "".join(tokens[i] for i in result if i < len(tokens))
        text = text.replace("▁", " ").strip()
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def transcribe(self, wav_path):
        features = self._load_features(wav_path)
        encoder_input = features[None, ...]
        x_lens = np.array([features.shape[0]], dtype=np.int64)

        t0 = time.time()
        encoder_out, _ = self.encoder_sess.run(
            None, {"x": encoder_input, "x_lens": x_lens}
        )
        t1 = time.time()

        encoder_out = encoder_out[0]
        T = encoder_out.shape[0]

        y = [0, 0]
        result = []

        t2 = time.time()
        for t in range(T):
            enc_frame = encoder_out[t:t+1, :]

            while True:
                dec_in = np.array([y[-2:]], dtype=np.int64)
                dec_out = self.decoder_sess.run(None, {"y": dec_in})[0]

                join_out = self.joiner_sess.run(None, {
                    "encoder_out": enc_frame.astype(np.float32),
                    "decoder_out": dec_out.astype(np.float32)
                })[0]

                token = int(np.argmax(join_out, axis=-1)[0])
                if token == 0:
                    break

                result.append(token)
                y.append(token)
        t3 = time.time()

        return {
            "text": self._decode_tokens(result),
            "num_tokens": len(result),
            "encoder_time": round(t1 - t0, 3),
            "decoder_time": round(t3 - t2, 3),
            "use_int8": self.use_int8
        }


    def parse_vietnamese_date(self, text):
        """
        Tìm và chuyển ngày tiếng Việt sang dd/mm/yyyy
        Ví dụ:
        'ngày một tháng mười hai năm hai không hai lăm'
        -> '1/12/2025'
        """
        dt = dateparser.parse(
            text,
            languages=["vi"],
            settings={
                "DATE_ORDER": "DMY",
                "PREFER_DAY_OF_MONTH": "first"
            }
        )

        if dt is None:
            return None

        return dt.strftime("%-d/%-m/%Y")


class ZipformerASRWithBeamSearch(ZipformerASR):
    def __init__(
        self,
        model_dir: str,
        use_int8: bool = False,
        beam_size: int = 5,
        max_symbols_per_frame: int = 5,
        topk: int = None,
        blank_id: int = 0,
    ):
        super().__init__(model_dir=model_dir, use_int8=use_int8)
        self.beam_size = beam_size
        self.max_symbols_per_frame = max_symbols_per_frame
        self.topk = topk if topk is not None else max(beam_size, 8)
        self.blank_id = blank_id

    @staticmethod
    def _log_softmax_1d(logits_1d: np.ndarray) -> np.ndarray:
        m = np.max(logits_1d)
        x = logits_1d - m
        return x - np.log(np.sum(np.exp(x)))

    @staticmethod
    def _logaddexp(a: float, b: float) -> float:
        return float(np.logaddexp(a, b))

    def _rnnt_beam_search(self, encoder_out_2d: np.ndarray) -> list[int]:
        """
        encoder_out_2d: (T, Denc) float32
        returns: list token ids (không gồm 2 token start)
        """
        beam_size = self.beam_size
        max_symbols = self.max_symbols_per_frame
        topk = self.topk
        blank_id = self.blank_id

        T = int(encoder_out_2d.shape[0])

        # hypothesis: y_seq (tuple) -> logp
        hyps: dict[tuple[int, ...], float] = {(blank_id, blank_id): 0.0}

        for t in range(T):
            enc_frame = encoder_out_2d[t:t+1, :].astype(np.float32)  # (1, Denc)

            next_hyps: dict[tuple[int, ...], float] = {}
            active: dict[tuple[int, ...], float] = dict(hyps)

            for _ in range(max_symbols):
                expanded: dict[tuple[int, ...], float] = {}

                active_items = sorted(active.items(), key=lambda kv: kv[1], reverse=True)[:beam_size]
                for y_seq, score in active_items:
                    dec_in = np.array([y_seq[-2:]], dtype=np.int64)  # (1,2)
                    dec_out = self.decoder_sess.run(None, {"y": dec_in})[0].astype(np.float32)

                    join_out = self.joiner_sess.run(
                        None,
                        {"encoder_out": enc_frame, "decoder_out": dec_out},
                    )[0]
                    logits = join_out[0].astype(np.float32)  # (V,)
                    logp = self._log_softmax_1d(logits)

                    V = int(logp.shape[0])
                    k = min(topk, V)

                    # top-k indices (bao gồm cả blank nếu blank nằm trong topk; nếu không vẫn xử lý blank riêng)
                    idx = np.argpartition(-logp, k - 1)[:k]
                    idx = idx[np.argsort(-logp[idx])]

                    # blank: chuyển sang frame tiếp theo
                    blank_score = score + float(logp[blank_id])
                    old = next_hyps.get(y_seq)
                    next_hyps[y_seq] = blank_score if old is None else self._logaddexp(old, blank_score)

                    # non-blank: ở lại frame, append token
                    for tok in idx:
                        tok = int(tok)
                        if tok == blank_id:
                            continue
                        new_y = y_seq + (tok,)
                        new_score = score + float(logp[tok])
                        old2 = expanded.get(new_y)
                        expanded[new_y] = new_score if old2 is None else self._logaddexp(old2, new_score)

                if not expanded:
                    break

                # prune
                expanded = dict(sorted(expanded.items(), key=lambda kv: kv[1], reverse=True)[:beam_size])
                active = expanded

            if not next_hyps:
                next_hyps = hyps

            hyps = dict(sorted(next_hyps.items(), key=lambda kv: kv[1], reverse=True)[:beam_size])

        best_y = max(hyps.items(), key=lambda kv: kv[1])[0]
        return list(best_y[2:])  # bỏ 2 token start

    def transcribe(self, wav_path: str):
        features = self._load_features(wav_path)          # (T,80)
        encoder_input = features[None, ...]               # (1,T,80)
        x_lens = np.array([features.shape[0]], np.int64)  # (1,)

        t0 = time.time()
        encoder_out, _ = self.encoder_sess.run(None, {"x": encoder_input, "x_lens": x_lens})
        t1 = time.time()

        encoder_out = encoder_out[0].astype(np.float32)   # (T', Denc)

        t2 = time.time()
        result = self._rnnt_beam_search(encoder_out)
        t3 = time.time()

        return {
            "text": self._decode_tokens(result),
            "num_tokens": len(result),
            "encoder_time": round(t1 - t0, 3),
            "decoder_time": round(t3 - t2, 3),
            "use_int8": self.use_int8,
            "beam_size": self.beam_size,
        }


# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path wav file")
    args = parser.parse_args()

    asr = ZipformerASRWithBeamSearch(model_dir=os.path.join("../assets", "zipformer"))

    result = asr.transcribe(args.wav)

    print("\n========== ZIPFORMER RESULT ==========")
    print("Text         :", result["text"])
    print("Tokens       :", result["num_tokens"])
    print("Encoder time :", result["encoder_time"], "s")
    print("Decoder time :", result["decoder_time"], "s")
    print("INT8         :", result["use_int8"])
    print("================================\n")

    asr = ZipformerASRWithBeamSearch(model_dir=os.path.join("../assets", "vietasr-finetune"))

    result = asr.transcribe(args.wav)

    print("\n========== VIETASR RESULT ==========")
    print("Text         :", result["text"])
    print("Tokens       :", result["num_tokens"])
    print("Encoder time :", result["encoder_time"], "s")
    print("Decoder time :", result["decoder_time"], "s")
    print("INT8         :", result["use_int8"])
    print("================================\n")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--wav", required=True, help="Path wav file")
    # parser.add_argument("--model_dir", default="assets/models/vietasr_pretrain", help="Path to VietASR model directory")
    # args = parser.parse_args()
    #
    # asr = ZipformerASR(
    #     model_dir=args.model_dir,
    # )
    #
    # result = asr.transcribe(args.wav)
    #
    # print("\n========== VIETASR PRETRAIN RESULT ==========")
    # print("Text         :", result["text"])
    # print("Tokens       :", result["num_tokens"])
    # print("Encoder time :", result["encoder_time"], "s")
    # print("Decoder time :", result["decoder_time"], "s")
    # print("INT8         :", result["use_int8"])
    # print("================================\n")


if __name__ == "__main__":
    main()
