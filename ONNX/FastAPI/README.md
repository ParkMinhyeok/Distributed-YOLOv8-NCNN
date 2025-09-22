## 율주행 로봇 테스트 베드를 실제 도로와 유사한 구현을 위한 코스 및 환경 구성

### 평가 지표 (Metrics)
- FPS (Frames Per Second) / Δt: 워밍업 제거 후의 평균 FPS와 95% 신뢰 구간(CI)을 함께 보여줍니다.
- 통신량 (Uplink/Downlink): 초당 평균(Mbps) 및 프레임당 바이트의 누적 통신량을 나타냅니다.
- 통신 횟수: 초당 요청 수 (ops/s)와 프레임당 요청 수를 나타냅니다.
- (선택) 지연 (Latency): 요청에 대한 평균/표준편차/95% 퍼센타일의 왕복 시간(round-trip)을 나타냅니다.
- (선택) 정확도 (Accuracy): 프레임별 라벨(ground-truth)이 있는 경우, mAP@0.5를 측정합니다 (동영상 실험에만 권장).

### 실험 스키마 (Scheme)
- RPI4 100% offline: 라즈베리 파이 4에서 모든 처리를 오프라인으로 수행합니다.
- RPI4-Edge Server adaptive split (auto): 라즈베리 파이 4와 엣지 서버 간에 작업을 자동으로 분할하는 방식입니다.
- RPI4-Edge Server raw image split: 라즈베리 파이 4에서 엣지 서버로 원본 이미지를 전송하여 분할 처리하는 방식입니다.
- RPI4-Server 75-layer split: 라즈베리 파이 4와 서버 간에 75개 레이어를 기준으로 작업을 분할하는 방식입니다.
  
### 채널 조건 (Channel Condition)
- trial 1, 2: Good (100Mbps)
- trial 3: Normal (20Mbps)
- trial 4: Bad (2Mbps)
