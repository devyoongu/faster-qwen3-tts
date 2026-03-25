import httpx
import time
import json

# --- 설정 구간 ---
# 1. GPU 서버의 실제 IP 주소를 입력하세요.
SERVER_IP = "172.31.79.202"
SERVER_PORT = "8000"
URL = f"http://{SERVER_IP}:{SERVER_PORT}/v1/audio/speech"

# 2. 테스트할 텍스트와 모델 설정
PAYLOAD = {
    "model": "tts-1",
    "input": "안녕하세요. 경동나비엔 고객센터 에이아이 콜봇입니다. 무엇을 도와드릴까요?",
    "voice": "default",  # 서버 실행 시 --ref-audio로 지정한 목소리가 'default'가 됩니다.
    "response_format": "wav"
}

OUTPUT_PATH = "output.wav"

def check_remote_ttfa():
    print(f"📡 서버 접속 시도: {URL}")
    print(f"📝 입력 텍스트: {PAYLOAD['input']}")
    print("-" * 50)

    # 타임아웃을 None으로 설정해야 첫 실행(CUDA Graph 캡처) 시 에러가 나지 않습니다.
    timeout = httpx.Timeout(None)

    start_time = time.time()

    try:
        with httpx.stream("POST", URL, json=PAYLOAD, timeout=timeout) as response:
            if response.status_code != 200:
                print(f"❌ 에러 발생: 상태 코드 {response.status_code}")
                print(response.read().decode())
                return

            first_chunk = True
            with open(OUTPUT_PATH, "wb") as f:
                for chunk in response.iter_bytes():
                    if first_chunk:
                        # 첫 번째 청크(WAV 헤더 포함)가 도착한 시점 측정
                        end_time = time.time()
                        ttfa_ms = (end_time - start_time) * 1000

                        print(f"✅ 첫 번째 음성 조각 수신 성공!")
                        print(f"🚀 네트워크 포함 TTFA: {ttfa_ms:.2f} ms")
                        print("-" * 50)

                        first_chunk = False

                    f.write(chunk)

            print(f"💾 WAV 파일 저장 완료: {OUTPUT_PATH}")

    except httpx.ConnectError:
        print("❌ 서버 연결 실패: IP 주소와 포트를 확인해 주세요.")
    except Exception as e:
        print(f"❌ 예외 발생: {e}")

def main():
    # 실제 서비스 환경을 시뮬레이션하기 위해 실행
    check_remote_ttfa()

if __name__ == "__main__":
    main()