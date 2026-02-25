# -*- coding: utf-8 -*-
"""SenseVoice RKNN inference wrapper for RK3588 NPU."""

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import kaldi_native_fbank as knf
import numpy as np
import sentencepiece as spm
import soundfile as sf
import yaml
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from rknnlite.api.rknn_lite import RKNNLite

logger = logging.getLogger(__name__)

RKNN_INPUT_LEN = 171
SPEECH_SCALE = 1

FRAME_SAMPLE_LENGTH = 400
FRAME_SHIFT_LENGTH = 160
FRAME_DURATION_MS = 10

LANGUAGES = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}


class WavFrontend:
    def __init__(
        self,
        cmvn_file: Optional[str] = None,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 0,
    ):
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.mel_opts.num_bins = n_mels
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.cmvn: Optional[np.ndarray] = None
        if self.cmvn_file:
            self.cmvn = self._load_cmvn()

    def fbank(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform * (1 << 15)
        fbank_fn = knf.OnlineFbank(self.opts)
        fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frames = fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(frames):
            mat[i, :] = fbank_fn.get_frame(i)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def apply_lfr(self, inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        LFR_inputs = []
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n :].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)
        return np.vstack(LFR_inputs).astype(np.float32)

    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        if self.cmvn is None:
            return inputs
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars_ = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        return (inputs + means) * vars_

    def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)
        if self.cmvn_file:
            feat = self.apply_cmvn(feat)
        feat_len = np.array(feat.shape[0]).astype(np.int32)
        return feat, feat_len

    def get_features(self, inputs: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(inputs, str):
            inputs, _ = self._load_audio(inputs)
        fbank, _ = self.fbank(inputs)
        feats, _ = self.lfr_cmvn(fbank)
        return feats

    def _load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        data, sample_rate = sf.read(filename, always_2d=True, dtype="float32")
        assert sample_rate == 16000, f"Only 16000 Hz is supported, got {sample_rate}Hz"
        return data[:, 0], sample_rate

    def _load_cmvn(self) -> np.ndarray:
        with open(self.cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        means_list, vars_list = [], []
        for i, line in enumerate(lines):
            line_item = line.split()
            if not line_item:
                continue
            if line_item[0] == "<AddShift>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    means_list = line_item[3 : len(line_item) - 1]
            elif line_item[0] == "<Rescale>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    vars_list = line_item[3 : len(line_item) - 1]
        means = np.array(means_list).astype(np.float64)
        vars_ = np.array(vars_list).astype(np.float64)
        return np.array([means, vars_])


class VadSession:
    def __init__(self, model_path: str):
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = InferenceSession(
            model_path, sess_options=sess_opt, providers=["CPUExecutionProvider"]
        )

    def __call__(self, input_content: np.ndarray) -> List[np.ndarray]:
        input_dict = {"speech": input_content}
        return self.session.run(None, input_dict)


class FSMNVad:
    def __init__(self, config_dir: str):
        config_dir = Path(config_dir)
        with open(config_dir / "fsmn-config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        frontend_conf = self.config.get("WavFrontend", {}).get("frontend_conf", {})
        self.frontend = WavFrontend(
            cmvn_file=str(config_dir / "fsmn-am.mvn"),
            **frontend_conf
        )
        self.vad_session = VadSession(str(config_dir / "fsmnvad-offline.onnx"))
        self.speech_noise_thres = 0.6
        self.max_end_sil_frame_cnt_thresh = 650

    def _extract_feature(self, waveform: np.ndarray) -> np.ndarray:
        fbank, _ = self.frontend.fbank(waveform)
        feats, _ = self.frontend.lfr_cmvn(fbank)
        return feats.astype(np.float32)

    def _compute_decibel(self, waveform: np.ndarray) -> List[float]:
        decibel = []
        for offset in range(0, len(waveform) - FRAME_SAMPLE_LENGTH + 1, FRAME_SHIFT_LENGTH):
            energy = np.square(waveform[offset : offset + FRAME_SAMPLE_LENGTH]).sum()
            decibel.append(10 * math.log10(energy + 1e-6))
        return decibel

    def _compute_scores(self, feats: np.ndarray) -> Tuple[np.ndarray, int]:
        scores = self.vad_session(feats[None, ...])
        frm_cnt = scores[0].shape[1]
        return scores[0], frm_cnt

    def _get_frame_state(self, scores: np.ndarray, decibel: List[float], t: int) -> int:
        if decibel[t] < -100.0:
            return 0
        sil_pdf_score = scores[0][t][0]
        speech_prob = math.log(1.0 - sil_pdf_score) if sil_pdf_score < 1.0 else -100
        noise_prob = math.log(sil_pdf_score)
        if math.exp(speech_prob) >= math.exp(noise_prob) + self.speech_noise_thres:
            return 1
        return 0

    def segments_offline(self, waveform: np.ndarray) -> List[List[int]]:
        feats = self._extract_feature(waveform)
        scores, frm_cnt = self._compute_scores(feats)
        decibel = self._compute_decibel(waveform)

        segments = []
        in_speech = False
        start_ms = 0
        silence_count = 0

        for t in range(frm_cnt):
            state = self._get_frame_state(scores, decibel, t)
            if state == 1:
                if not in_speech:
                    start_ms = t * FRAME_DURATION_MS
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count > self.max_end_sil_frame_cnt_thresh:
                        segments.append([start_ms, (t - silence_count) * FRAME_DURATION_MS])
                        in_speech = False
                        silence_count = 0

        if in_speech:
            segments.append([start_ms, frm_cnt * FRAME_DURATION_MS])

        return segments


class SenseVoiceRKNN:
    PREFIX_LEN = 4
    MAX_SPEECH_LEN = RKNN_INPUT_LEN - PREFIX_LEN

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        self.model_dir = model_dir

        logger.info(f"Loading embedding from {model_dir / 'embedding.npy'}")
        self.embedding = np.load(model_dir / "embedding.npy")

        logger.info(f"Loading RKNN model from {model_dir / 'sense-voice-encoder.rknn'}")
        start = time.time()
        self.encoder = RKNNLite(verbose=False)
        self.encoder.load_rknn(str(model_dir / "sense-voice-encoder.rknn"))
        self.encoder.init_runtime()
        logger.info(f"RKNN model loaded in {time.time() - start:.2f}s")

        self.blank_id = 0
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"))

        logger.info("Loading frontend...")
        self.frontend = WavFrontend(str(model_dir / "am.mvn"))

        logger.info("Loading VAD model...")
        self.vad = FSMNVad(str(model_dir))

        logger.info("SenseVoiceRKNN initialized successfully")

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        use_itn: bool = False,
    ) -> Dict:
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        logger.info(f"Audio: {len(waveform) / sample_rate:.2f}s, {waveform.shape[1]} channel(s)")

        if waveform.shape[1] > 1:
            waveform = waveform[:, 0:1]

        channel_data = waveform[:, 0]
        segments = self.vad.segments_offline(channel_data)

        results = []
        full_text = ""

        for start_ms, end_ms in segments:
            audio_feats = self.frontend.get_features(channel_data[start_ms * 16 : end_ms * 16])
            lang_id = LANGUAGES.get(language, 0)
            text = self._inference(audio_feats[None, ...], lang_id, use_itn)

            results.append({
                "start_time_s": start_ms / 1000,
                "end_time_s": end_ms / 1000,
                "text": text,
            })
            full_text += text

        return {
            "segments": results,
            "full_transcription": full_text,
        }

    def _inference(self, speech: np.ndarray, language: int, use_itn: bool) -> str:
        language_query = self.embedding[[[language]]]
        text_norm_query = self.embedding[[[14 if use_itn else 15]]]
        event_emo_query = self.embedding[[[1, 2]]]

        speech = speech * SPEECH_SCALE

        if speech.shape[1] <= self.MAX_SPEECH_LEN:
            input_content = self._prepare_input(language_query, event_emo_query, text_norm_query, speech)
            input_content = self._pad_input(input_content)
            return self._run_inference(input_content)

        all_hypos = []
        for start in range(0, speech.shape[1], self.MAX_SPEECH_LEN):
            end = min(start + self.MAX_SPEECH_LEN, speech.shape[1])
            chunk = speech[:, start:end, :]
            chunk_input = self._prepare_input(language_query, event_emo_query, text_norm_query, chunk)
            chunk_input = self._pad_input(chunk_input)
            encoder_out = self.encoder.inference(inputs=[chunk_input])[0]
            hypos = self._unique_consecutive(encoder_out[0].argmax(axis=0))
            all_hypos.extend(hypos)

        return self.sp.DecodeIds(all_hypos)

    def _prepare_input(self, lang_q: np.ndarray, emo_q: np.ndarray, itn_q: np.ndarray, speech: np.ndarray) -> np.ndarray:
        return np.concatenate([lang_q, emo_q, itn_q, speech], axis=1).astype(np.float32)

    def _pad_input(self, input_content: np.ndarray) -> np.ndarray:
        if input_content.shape[1] >= RKNN_INPUT_LEN:
            return input_content
        return np.pad(input_content, ((0, 0), (0, RKNN_INPUT_LEN - input_content.shape[1]), (0, 0)))

    def _run_inference(self, input_content: np.ndarray) -> str:
        encoder_out = self.encoder.inference(inputs=[input_content])[0]
        hypos = self._unique_consecutive(encoder_out[0].argmax(axis=0))
        return self.sp.DecodeIds(hypos)

    def _unique_consecutive(self, arr: np.ndarray) -> List[int]:
        if len(arr) == 0:
            return []
        mask = np.append([True], arr[1:] != arr[:-1])
        out = arr[mask]
        out = out[out != self.blank_id]
        return out.tolist()
