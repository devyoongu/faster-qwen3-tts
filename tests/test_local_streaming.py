"""실시간 스트리밍 TTS 재생 테스트.

PCM 포맷으로 수신 → 청크 도착 즉시 스피커 재생 + WAV 저장 + TTFA 측정.

Requirements:
    pip install sounddevice
"""
import wave

import httpx
import sounddevice as sd
import time

# --- 설정 구간 ---
SERVER_IP = "172.31.79.202"
# SERVER_IP = "172.31.88.110"
SERVER_PORT = "8000"
# SERVER_PORT = "8001"
URL = f"http://{SERVER_IP}:{SERVER_PORT}/v1/audio/speech"

SAMPLE_RATE = 24000  # 서버 출력 샘플레이트 (24kHz 고정)
CHANNELS = 1         # 모노

PAYLOAD = {
    "input": "안녕하세요. 경동나비엔 고객센터 에이아이 콜봇입니다.",
    # "input": "안녕하세요. 경동나비엔 고객센터 에이아이 콜봇입니다. 보다 정확한 상담을 위해 조용한 곳에서 상담사와 이야기 하드시 편하게 말씀해 주시면 더 빠르게 도와 드릴 수 있습니다. 무엇을 도와 드릴까요?",
    "voice": "default",
    "response_format": "pcm",  # raw int16 LE PCM (WAV 헤더 없음 → 더 낮은 지연)
}

OUTPUT_PATH = "output.wav"


def stream_and_play():
    print(f"📡 서버 접속 시도: {URL}")
    print(f"📝 입력 텍스트: {PAYLOAD['input']}")
    print("-" * 50)

    timeout = httpx.Timeout(None)
    start_time = time.time()
    ttfa_measured = False
    pcm_frames = bytearray()

    try:
        with sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        ) as audio_stream:
            with httpx.stream("POST", URL, json=PAYLOAD, timeout=timeout) as response:
                if response.status_code != 200:
                    print(f"❌ 에러 발생: 상태 코드 {response.status_code}")
                    print(response.read().decode())
                    return

                for chunk in response.iter_bytes():
                    if not ttfa_measured:
                        ttfa_ms = (time.time() - start_time) * 1000
                        print(f"✅ 첫 번째 음성 조각 수신 성공!")
                        print(f"🚀 네트워크 포함 TTFA: {ttfa_ms:.2f} ms")
                        print("-" * 50)
                        ttfa_measured = True

                    # 청크 도착 즉시 스피커로 출력
                    audio_stream.write(chunk)
                    pcm_frames.extend(chunk)

    except httpx.ConnectError:
        print("❌ 서버 연결 실패: IP 주소와 포트를 확인해 주세요.")
        return
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return

    # 수신한 PCM → WAV 파일로 저장
    if pcm_frames:
        with wave.open(OUTPUT_PATH, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)       # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(pcm_frames))
        print(f"💾 WAV 파일 저장 완료: {OUTPUT_PATH}")


def main():
    stream_and_play()


if __name__ == "__main__":
    main()