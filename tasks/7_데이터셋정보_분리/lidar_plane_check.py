"""⑦-A  HE LiDAR로 화각 70° 가정을 검증할 수 있는가 (CPU 전용).

배경: HE 메타에 카메라 내부 파라미터가 없어 화각 70°를 가정하고 있다. 틀리면 3D 크기가
통째로 상수배 어긋나는데, 지금까지 이를 잴 기준이 없었다. parquet에 LiDAR 점군이
들어 있으니 이것을 절대 기준으로 쓸 수 있는지 확인한다.

방법: 같은 프레임에서 (1) LiDAR 점군의 지배 평면까지 거리와 (2) 실측 depth를 화각 f로
역투영한 점군의 지배 평면까지 거리를 각각 재고, 비율이 프레임 간 일정한지 본다.
일정해야만 그 비율을 스케일 보정 계수로 쓸 수 있다.

한계를 미리 적어 둔다: 카메라-LiDAR extrinsics가 없어 두 센서가 '같은 평면'을 보고
있다는 보장이 없다. LiDAR는 방 전체(중앙 약 4.3m)를, 깊이 카메라는 작업면(중앙 약 1.7m)을
본다. 그래서 이 스크립트의 결론은 '화각은 몇 도다'가 아니라
'이 방법이 성립하는가(비율이 일정한가)'까지다. 성립하지 않으면 그 자체가 결과다.
"""
import os, sys, json, glob
import numpy as np

ROOT = os.path.expanduser("~/task3")
HE = "/data2/humanoid_dataset_isangmin/humanoid-everyday"
OUT = f"{ROOT}/lidar_check"
FOV_LIST = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
N_EPISODES, N_FRAMES = 10, 6
LIDAR_MAX_M = 3.0          # 카메라 관측 범위에 맞춰 원거리 점을 자른다


def log(m):
    print(m, flush=True)


def ransac_plane(pts, iters=250, thr=0.03, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(pts) < 80:
        return None
    best_n, best_d, best_i = None, None, 0
    for _ in range(iters):
        p0, p1, p2 = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        d = -float(n @ p0)
        inl = int((np.abs(pts @ n + d) < thr).sum())
        if inl > best_i:
            best_n, best_d, best_i = n, d, inl
    if best_n is None:
        return None
    m = np.abs(pts @ best_n + best_d) < thr
    q = pts[m]
    if len(q) >= 80:
        c = q.mean(0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        best_n = vt[-1] / np.linalg.norm(vt[-1])
        best_d = -float(best_n @ c)
    return dict(dist=float(abs(best_d)), ratio=float(m.mean()), n=best_n.tolist())


def depth_points(D, fov, stride=6):
    H, W = D.shape
    f = (W / 2) / np.tan(np.deg2rad(fov) / 2)
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = D[::stride, ::stride]
    ok = np.isfinite(z) & (z > 0.2) & (z < 6.0)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    return np.stack([(xs - W / 2) / f * z, (ys - H / 2) / f * z, z], 1)


def main():
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{HE}/data/chunk-*/episode_*.parquet"))
    log(f"parquet {len(files)}개")
    rng = np.random.default_rng(0)
    rows, used = [], 0
    step = max(1, len(files) // (N_EPISODES * 4))
    for fp in files[::step]:
        if used >= N_EPISODES:
            break
        try:
            df = pd.read_parquet(fp, columns=["observation.lidar", "observation.depth.egocentric"])
        except Exception:
            continue
        n = len(df)
        got = 0
        for i in np.linspace(0, n - 1, min(N_FRAMES, n)).astype(int):
            try:
                L = np.stack([np.asarray(p, dtype=np.float64)
                              for p in df["observation.lidar"].iloc[i]])
                if np.nanpercentile(np.abs(L), 95) > 50:
                    L = L * 1e-3                       # mm -> m
                r = np.linalg.norm(L, axis=1)
                L = L[np.isfinite(r) & (r > 0.15) & (r < LIDAR_MAX_M)]
                if len(L) < 200:
                    continue
                pl = ransac_plane(L, rng=rng)
                if not pl or pl["ratio"] < 0.15:
                    continue
                D = np.stack([np.asarray(x) for x in
                              df["observation.depth.egocentric"].iloc[i]]).astype(np.float64) * 1e-3
                rec = dict(ep=os.path.basename(fp), frame=int(i), lidar_n=int(len(L)),
                           lidar_dist=round(pl["dist"], 4), lidar_inlier=round(pl["ratio"], 3))
                for fov in FOV_LIST:
                    P = depth_points(D, fov)
                    pc = ransac_plane(P, rng=rng)
                    if not pc or pc["ratio"] < 0.15:
                        continue
                    rec[f"cam_{int(fov)}"] = round(pc["dist"], 4)
                    rec[f"ratio_{int(fov)}"] = round(pl["dist"] / max(pc["dist"], 1e-6), 4)
                if any(k.startswith("ratio_") for k in rec):
                    rows.append(rec); got += 1
            except Exception:
                continue
        if got:
            used += 1
            log(f"  {os.path.basename(fp)}: {got}프레임 (누적 {used})")

    json.dump(rows, open(f"{OUT}/rows.json", "w"), ensure_ascii=False, indent=1)
    log(f"\n총 {len(rows)}프레임")
    if len(rows) < 3:
        log("!! 표본 부족 — LiDAR/depth 조합을 얻지 못함")
        return
    log("\n화각별 (LiDAR 평면거리 / 카메라 평면거리)")
    log(f"{'화각':<7}{'n':>5}{'중앙':>9}{'표준편차':>10}{'변동계수':>10}")
    res = {}
    for fov in FOV_LIST:
        v = np.array([r[f"ratio_{int(fov)}"] for r in rows if f"ratio_{int(fov)}" in r])
        if len(v) < 3:
            continue
        cv = float(v.std() / max(abs(v.mean()), 1e-9))
        res[int(fov)] = dict(n=len(v), median=round(float(np.median(v)), 3),
                             std=round(float(v.std()), 3), cv=round(cv, 3))
        log(f"{fov:<7.0f}{len(v):>5}{np.median(v):>9.3f}{v.std():>10.3f}{cv:>10.3f}")

    cvs = [d["cv"] for d in res.values()]
    log("")
    if cvs and min(cvs) > 0.30:
        log(f"변동계수가 전 화각에서 {min(cvs):.2f} 이상 — 프레임마다 비율이 흩어진다.")
        log("두 센서가 같은 평면을 보지 않는다는 뜻이며, LiDAR로 화각을 특정할 수 없다.")
        log("→ 결론: 이 경로는 성립하지 않음. 화각 검증은 다른 기준(UniDepth K_pred)으로 간다.")
        verdict = "불성립"
    else:
        best = min(res.items(), key=lambda kv: abs(kv[1]["median"] - 1.0))
        log(f"비율이 안정적이며 1에 가장 가까운 화각은 {best[0]}° (중앙 {best[1]['median']}).")
        verdict = f"성립 — 추정 화각 {best[0]}°"
    json.dump(dict(verdict=verdict, n_rows=len(rows), by_fov=res),
              open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1)
    log(f"\n판정: {verdict}")


if __name__ == "__main__":
    main()
