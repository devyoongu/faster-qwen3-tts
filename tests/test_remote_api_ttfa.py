#!/usr/bin/env python3
"""
Remote OpenAI-compatible TTS API 서버 TTFA/RTF 측정 + 실시간 오디오 플레이 스크립트.

Usage:
    python tests/test_remote_api_ttfa.py
    python tests/test_remote_api_ttfa.py --host 172.31.79.202 --port 8000
    python tests/test_remote_api_ttfa.py --runs 5 --text "안녕하세요, 반갑습니다."
    python tests/test_remote_api_ttfa.py --play          # 실시간 재생 (측정 1회)
    python tests/test_remote_api_ttfa.py --play --runs 3 # 재생 + 3회 측정
"""
import argparse
import io
import struct
import time
import wave
from typing import Optional

import httpx
import numpy as np

DEFAULT_HOST = "172.31.79.202"
DEFAULT_PORT = 8000
DEFAULT_SENTENCES = [
    "안녕하세요. 경동나비엔 고객센터 에이아이 콜봇입니다.",
    "보다 정확한 상담을 위해 조용한 곳에서 상담사와 이야기 하듯이 편하게 말씀해 주세요.",
    "더 빠르게 도와 드릴 수 있습니다. 무엇을 도와 드릴까요?",
]
DEFAULT_VOICE = "alloy"
DEFAULT_RUNS = 1


def _parse_wav_header(data: bytes) -> Optional[int]:
    """WAV 헤더에서 sample_rate를 파싱. 헤더가 불완전하면 None 반환."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    # fmt chunk: offset 24 = sample_rate
    try:
        sample_rate = struct.unpack_from("<I", data, 24)[0]
        return sample_rate
    except struct.error:
        return None


def _count_pcm_samples(audio_bytes: bytes, sample_rate: int) -> float:
    """WAV 바이트에서 오디오 길이(초)를 계산."""
    try:
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        # WAV 헤더가 streaming용 unknown-length일 경우 raw PCM 크기로 추정
        # 16-bit mono: bytes / 2 / sample_rate
        pcm_bytes = len(audio_bytes) - 44  # WAV 헤더 제외
        if pcm_bytes > 0 and sample_rate > 0:
            return pcm_bytes / 2 / sample_rate
        return 0.0


def _pcm_chunk_to_float32(raw: bytes) -> np.ndarray:
    """16-bit PCM bytes → float32 numpy array [-1, 1]."""
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    arr /= 32768.0
    return arr


def measure_ttfa(url: str, text: str, voice: str, run_idx: int,
                 player=None, sample_rate: int = 24000) -> dict:
    """
    단일 요청의 TTFA, 총 소요시간, RTF를 측정.
    PCM 포맷으로 수신 (WAV 헤더 없음 → 파싱 오류/잡음 없음).
    player가 주어지면 수신 즉시 재생 (문장 간 동일 player 재사용으로 잡음 방지).
    """
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "pcm",  # WAV 헤더 없이 순수 16-bit PCM
    }

    ttfa_ms: Optional[float] = None
    total_bytes = 0
    pcm_carry = b""  # 홀수 바이트 carry (16-bit 샘플 경계 유지)

    t_start = time.perf_counter()
    t_last_byte = t_start

    with httpx.stream("POST", url, json=payload, timeout=60.0) as resp:
        resp.raise_for_status()

        for chunk in resp.iter_bytes(chunk_size=4096):
            now = time.perf_counter()

            # 첫 데이터 도착 시점 = TTFA
            if ttfa_ms is None and len(chunk) > 0:
                ttfa_ms = (now - t_start) * 1000

            # 실시간 재생 (16-bit 샘플 경계 정렬)
            if player is not None and len(chunk) > 0:
                raw = pcm_carry + chunk
                if len(raw) % 2 != 0:
                    pcm_carry = raw[-1:]
                    raw = raw[:-1]
                else:
                    pcm_carry = b""
                if len(raw) > 0:
                    player(_pcm_chunk_to_float32(raw), sample_rate)

            total_bytes += len(chunk)
            t_last_byte = now

    # RTF = 서버가 오디오를 생성한 속도 (네트워크 수신 완료 기준, 재생 시간 제외)
    elapsed_s = t_last_byte - t_start

    audio_duration_s = total_bytes / 2 / sample_rate if total_bytes > 0 else 0.0
    rtf = audio_duration_s / elapsed_s if elapsed_s > 0 else 0.0

    return {
        "run": run_idx,
        "ttfa_ms": ttfa_ms or 0.0,
        "total_ms": elapsed_s * 1000,
        "audio_duration_s": audio_duration_s,
        "rtf": rtf,
        "total_bytes": total_bytes,
        "sample_rate": sample_rate,
    }


def run_benchmark(host: str, port: int, sentences: list, voice: str, runs: int, save_wav: bool, play: bool = False):
    url = f"http://{host}:{port}/v1/audio/speech"
    print(f"\n{'='*60}")
    print(f"Remote TTS API TTFA Benchmark")
    print(f"  URL      : {url}")
    print(f"  Voice    : {voice}")
    print(f"  Runs     : {runs}")
    print(f"  Play     : {'ON' if play else 'OFF'}")
    print(f"  Sentences: {len(sentences)}")
    for i, s in enumerate(sentences, 1):
        print(f"    {i}. {s}")
    print(f"{'='*60}\n")

    # 서버 연결 확인
    try:
        httpx.get(f"http://{host}:{port}/", timeout=5.0)
    except httpx.ConnectError:
        print(f"ERROR: 서버에 연결할 수 없습니다 ({host}:{port})")
        print("  - 서버가 실행 중인지 확인하세요")
        print("  - 방화벽에서 포트가 열려 있는지 확인하세요")
        return
    except Exception:
        pass  # /는 404일 수 있으나 서버는 살아있음

    all_results = []

    for run_idx in range(1, runs + 1):
        print(f"\n--- Run {run_idx}/{runs} ---")
        run_results = []
        for s_idx, sentence in enumerate(sentences, 1):
            label = "재생 중" if play else "측정 중"
            print(f"  [{s_idx}/{len(sentences)}] {label}: {sentence[:40]}{'...' if len(sentence) > 40 else ''}", end=" ", flush=True)

            # 문장마다 새 player (PCM 포맷 + carry byte 수정으로 잡음 방지)
            player = None
            if play:
                try:
                    import sys as _sys, os as _os
                    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
                    from examples.audio import StreamPlayer
                    player = StreamPlayer()
                except ImportError:
                    print("  [경고] sounddevice 미설치. 재생 생략.")

            try:
                r = measure_ttfa(url, sentence, voice, run_idx,
                                 player=player, sample_rate=24000)
                run_results.append(r)
                print(f"→ TTFA={r['ttfa_ms']:.1f}ms  RTF={r['rtf']:.3f}")
            except httpx.HTTPStatusError as e:
                print(f"FAILED: HTTP {e.response.status_code}")
            except Exception as e:
                print(f"FAILED: {e}")
            finally:
                if player is not None:
                    player.close(wait=True)  # 재생 완료 후 다음 문장 시작

        all_results.append(run_results)

    # 통계: Run 1 제외 (warm-up), 문장별 평균
    stat_runs = all_results[1:] if len(all_results) > 1 else all_results
    if not stat_runs or not stat_runs[0]:
        print("\n측정 결과가 없습니다.")
        return

    print(f"\n{'='*60}")
    n = len(stat_runs)
    print(f"결과 요약 (Run 1 제외, n={n}회 평균)")
    print(f"  {'문장':<45} {'TTFA(avg)':>10} {'RTF(avg)':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10}")
    for s_idx, sentence in enumerate(sentences):
        ttfa_vals = [stat_runs[r][s_idx]["ttfa_ms"] for r in range(n) if s_idx < len(stat_runs[r])]
        rtf_vals  = [stat_runs[r][s_idx]["rtf"]     for r in range(n) if s_idx < len(stat_runs[r])]
        if ttfa_vals:
            label = sentence[:43] + ".." if len(sentence) > 45 else sentence
            print(f"  {label:<45} {sum(ttfa_vals)/len(ttfa_vals):>9.1f}ms {sum(rtf_vals)/len(rtf_vals):>10.3f}")
    print(f"{'='*60}\n")

    if save_wav:
        _save_last_wav(url, sentences[-1], voice)


def _save_last_wav(url: str, text: str, voice: str):
    """마지막 결과의 오디오를 파일로 저장."""
    out_path = "test_remote_output.wav"
    payload = {"model": "tts-1", "input": text, "voice": voice, "response_format": "wav"}
    try:
        with httpx.stream("POST", url, json=payload, timeout=60.0) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        print(f"오디오 저장 완료: {out_path}")
    except Exception as e:
        print(f"오디오 저장 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="Remote TTS API TTFA 측정")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--sentences", nargs="+", default=DEFAULT_SENTENCES, metavar="TEXT",
                        help="재생할 문장 목록 (공백으로 구분, 기본: 한국어 3문장)")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--save-wav", action="store_true", help="마지막 문장 WAV 저장")
    parser.add_argument("--no-play", action="store_true", help="오디오 재생 비활성화")
    args = parser.parse_args()

    run_benchmark(
        host=args.host,
        port=args.port,
        sentences=args.sentences,
        voice=args.voice,
        runs=args.runs,
        save_wav=args.save_wav,
        play=not args.no_play,
    )


if __name__ == "__main__":
    main()