# ② Gripper / Hand Segmentation 가능 여부 — 완료

**요구**: 로봇 그리퍼·손 분할이 가능한지 확인, RobotSeg 검토, 활용 방안 구상.

**결론**: 현 스택(GroundingDINO+SAM2.1)으로 가능. Brainco 머리 캠에서 프레임당
4.15건 검출(로봇 팔·손), 마스크가 좌·우 팔을 정확히 분리
→ [검출률_실측.json](검출률_실측.json), [마스크예시](마스크예시_오레오_머리캠.png)

**RobotSeg**(CVPR 2026): arm/gripper/robot 3단계 마스크 자동 산출. 단 학습 로봇이
전부 팔+평행그리퍼·3인칭이라 휴머노이드 도메인 갭 존재 — '품질 향상' 관점에서
다음 단계 zero-shot 평가 예정 → [robotseg-review.md](robotseg-review.md)

**활용 구상**: 관측 정제(로봇 지우기), 파지 시점 자동 라벨, 가림 판정 강화, ghost 방지 필터.
**코드**: `tasks/도구/run_probe.py` (gripper 명령)
