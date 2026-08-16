"""파이프라인 실행 설정 — 대량 처리 시 조정하는 값들을 한곳에 모은다.

최종 목표는 다운로드된 전 데이터셋(Brainco 1,598 + HE g1 4,064 에피소드)에 3D box를
생성하는 것이다. 그때 조정해야 하는 것은 알고리즘이 아니라 **처리량과 산출물 구성**이다.

사용법:
    from config import RunConfig, PRESETS
    cfg = PRESETS["balanced"]                     # 미리 정의된 조합
    cfg = RunConfig(fps=3.0, save_video=False)    # 직접 지정
    cfg.estimate(n_episodes=4064, avg_sec=12.8)   # 예상 소요 시간
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class RunConfig:
    # ---------------- 처리량 (완료 시간을 결정하는 값) ----------------
    fps: float = 6.0
    """초당 몇 프레임을 처리할지. 원본 30fps에서 이 값으로 솎아낸다.
    처리 시간이 fps에 정비례한다. 3D box 어노테이션 용도로는 3~6fps면 충분하고,
    육안 검증용 영상은 10fps가 부드럽다."""

    max_frames_per_ep: int = 0
    """에피소드당 최대 프레임(0=제한 없음). 긴 에피소드가 전체 시간을 좌우하는 것을 막는다."""

    cameras: List[str] = field(default_factory=lambda: ["cam_left_high"])
    """Brainco에서 사용할 카메라. 처리 시간이 대수에 정비례한다.
    - 머리 2대: 안정적이고 크기가 정확하다(검증에서 대부분 90~100%)
    - 손목 2대: 조작 물체를 가까이 보지만 물체가 화면을 채우거나 시야를 벗어나 어렵다
    HE는 카메라가 1대(egocentric)라 이 설정과 무관하다."""

    # ---------------- 산출물 (용량과 후속 검증에 영향) ----------------
    amodal: bool = True
    """가려짐 보정(방식 B)을 함께 산출할지. 1~4단계는 공통이라 시간은 거의 늘지 않고
    영상 저장 시에만 2배가 된다. 끄면 '관측된 것만'(방식 A)만 남는다."""

    save_video: bool = False
    """오버레이 영상 저장 여부. 대량 실행에서는 끄는 것이 기본이다
    (에피소드당 1~3MB × 수천 개). 육안 검증 대상만 따로 켜서 다시 돌리면 된다."""

    save_frames_json: bool = True
    """프레임별 상세 기록(검출·마스크·게이트 진단). 이것이 실제 3D box 산출물이며,
    품질 점검과 재분석의 근거다. gzip으로 저장한다."""

    save_snapshots: bool = True
    """중간 프레임·가림 순간 정지 이미지. 용량이 작아 대량 실행에서도 켜둘 만하다."""

    # ---------------- 품질 ----------------
    stride_override: Optional[int] = None
    """프레임 간격을 직접 지정(설정 시 fps를 무시). 디버깅용."""

    align_dx_he: int = -20
    """HE의 RGB-depth 정렬 보정(px). STEP 0.5에서 실측한 값."""

    hfov_he: float = 70.0
    """HE intrinsics 가정(수평 화각). 메타에 intrinsics가 없어 필요하다.
    이 값이 틀리면 3D 크기가 상수배로 어긋난다 — 절대 크기가 중요하면 재검증이 필요하다."""

    # ---------------- 운영 ----------------
    resume: bool = True
    """이미 결과가 있는 유닛을 건너뛴다. 중단 후 재실행 시 이어서 진행된다."""

    # ---------------- 범위 (에피소드 샘플링) ----------------
    episodes_per_task: int = 0
    """태스크당 처리할 에피소드 수(0=전수). 에피소드는 같은 태스크의 반복 시도라
    전수 처리는 중복이 크다. robustness 검증에는 태스크당 3~5개면 충분하고,
    '모든 태스크 종류를 빠짐없이 덮는 것'이 커버리지의 핵심이다."""

    sample_seed: int = 20260803
    """에피소드 샘플링 seed. 고정하면 같은 표본이 재현된다."""

    robot_type: str = "g1"
    """HE는 g1(4,064)과 h1(4,885)이 섞여 있다. 본 과제 대상이 Unitree G1이므로 g1만 쓴다.
    빈 문자열이면 전체."""

    quality_report: bool = True
    """산출 후 품질 지표를 계산해 기록한다.
    주의: 이 지표는 **사람이 볼 표본을 고르는 용도**이며, 이것으로 파라미터를
    자동 조정하지 않는다. 지표 자체가 틀릴 수 있다(실측 사례: 화면 점유율 기준이
    손목 카메라에서 정상 물체를 오검출로 판정했다)."""

    def stride(self) -> int:
        if self.stride_override:
            return int(self.stride_override)
        return max(1, int(round(30.0 / max(self.fps, 0.1))))

    def estimate(self, n_episodes: int, avg_sec: float, n_cameras: Optional[int] = None,
                 sec_per_frame: float = 0.66, n_gpu: int = 1) -> dict:
        """예상 소요 시간. sec_per_frame은 실측 기본값(RTX 3090, depth 추정 포함 0.66s).
        HE는 실측 depth를 쓰므로 약 0.45s로 더 빠르다."""
        ncam = n_cameras if n_cameras is not None else max(len(self.cameras), 1)
        per_ep = avg_sec * self.fps
        if self.max_frames_per_ep:
            per_ep = min(per_ep, self.max_frames_per_ep)
        total_frames = n_episodes * per_ep * ncam
        hours = total_frames * sec_per_frame / 3600.0 / max(n_gpu, 1)
        return dict(frames=int(total_frames), hours=round(hours, 1),
                    days=round(hours / 24.0, 2), stride=self.stride())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


# 자주 쓰는 조합. 대량 실행 시 여기서 골라 쓰면 된다.
PRESETS = {
    "review": RunConfig(          # 육안 검증용 — 품질 우선, 영상 저장
        fps=10.0, max_frames_per_ep=220, save_video=True,
        cameras=["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]),

    "balanced": RunConfig(        # 기본 권장 — 머리 2대, 6fps
        fps=6.0, max_frames_per_ep=200, save_video=False,
        cameras=["cam_left_high", "cam_right_high"]),

    "fast": RunConfig(            # 대규모 전수 — 머리 1대, 3fps
        fps=3.0, max_frames_per_ep=120, save_video=False,
        cameras=["cam_left_high"]),

    "full": RunConfig(            # 최대 커버리지 — 4카메라, 6fps (시간이 4배)
        fps=6.0, max_frames_per_ep=200, save_video=False,
        cameras=["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]),

    "visible_only": RunConfig(    # 가림 보정 없이 관측된 것만
        fps=6.0, max_frames_per_ep=200, save_video=False, amodal=False,
        cameras=["cam_left_high", "cam_right_high"]),

    # robustness 검증 — 모든 태스크를 덮되 태스크당 소수 표본만
    "robust3": RunConfig(
        fps=6.0, max_frames_per_ep=200, save_video=False, episodes_per_task=3,
        cameras=["cam_left_high", "cam_right_high"]),
    "robust5": RunConfig(
        fps=6.0, max_frames_per_ep=200, save_video=False, episodes_per_task=5,
        cameras=["cam_left_high", "cam_right_high"]),
}


if __name__ == "__main__":
    # 전 데이터셋 적용 시 예상 시간 (실측 규모 기준)
    BC_EP, BC_SEC = 1598, 34.5      # Brainco 에피소드 수, 평균 길이
    HE_EP, HE_SEC = 4064, 12.8      # HE g1 에피소드 수, 평균 길이
    BC_TASKS, HE_TASKS = 8, 246
    print(f"{'프리셋':14s} {'에피소드':>14s} {'Brainco':>18s} {'HE':>18s} {'합계':>9s} {'GPU2':>8s}")
    print("-" * 88)
    for name, cfg in PRESETS.items():
        nb = BC_TASKS * cfg.episodes_per_task if cfg.episodes_per_task else BC_EP
        nh = HE_TASKS * cfg.episodes_per_task if cfg.episodes_per_task else HE_EP
        b = cfg.estimate(nb, BC_SEC, sec_per_frame=0.66)
        h = cfg.estimate(nh, HE_SEC, n_cameras=1, sec_per_frame=0.45)
        tot = b["hours"] + h["hours"]
        print(f"{name:14s} {f'{nb}+{nh}':>14s} {b['frames']:>9,}f {b['hours']:>6.1f}h "
              f"{h['frames']:>9,}f {h['hours']:>6.1f}h {tot:>7.1f}h {tot/2:>7.1f}h")
