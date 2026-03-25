import time
import torch
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
text = "안녕하세요. 경동나비엔 고객센터입니다."

# Warm-up (최초 1회 캡처)
list(model.generate_custom_voice_streaming(text=text, speaker="aiden", language="Korean"))

# 실제 측정
start = time.time()
for chunk, sr, _ in model.generate_custom_voice_streaming(text=text, speaker="aiden", language="Korean", chunk_size=8):
    ttfa = (time.time() - start) * 1000
    print(f"\n🚀 [결과] TTFA: {ttfa:.2f} ms")
    break # 첫 청크만 확인하고 종료