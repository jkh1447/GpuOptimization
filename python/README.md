# Description

## Deeplearning Model
1. 변수
- max_Clock : default로 딥러닝 진행지 peak Clock
- prevPerformance : 이전 에폭의 성능
- prevPowerDraw : 이전 에폭의 소모전력
- isUp : 이전 에폭에서 클럭을 올렸는지 유무
- isClockDone : 클럭을 결정했는지 유무
- sumPowerDraw : 누적 소모전력
- sumPerformance : 누적 성능(소모시간)
- shared_isStart : 파워소모 측정 시작을 위한 boolean
- shared_fepochd : 첫번째 에폭이 끝났는지 유무
- shared_fepochPer : 첫번째 에폭의 성능
- shared_fepochPow : 첫번째 에폭의 소모전력
- shared_powerDraw_1epoch : 한 에폭당 소모전력
- shared_isEpochDone : 한 에폭이 끝났는지 유무

## sendGpuInfo.py
