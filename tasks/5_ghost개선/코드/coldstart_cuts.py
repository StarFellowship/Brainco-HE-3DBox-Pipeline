"""cold-start 가드가 제거할 프레임을 r6 영상에서 뽑아 육안 검증용으로 저장 (CPU).
/data2 불필요 — A_visible.mp4 는 /home 에 있다."""
import os, json, cv2, glob
R6 = os.path.expanduser("~/task3/review/r6")
OUT = os.path.expanduser("~/task3/coldstart/cuts")
CONFIRM_N = 8
os.makedirs(OUT, exist_ok=True)

def simulate(unit, cn=CONFIRM_N):
    recs = json.load(open(f"{R6}/{unit}/frames.json"))
    stats = json.load(open(f"{R6}/{unit}/stats.json"))["stats"]
    out = {}
    for lb in stats:
        seen, est, rem = 0, None, []
        for i, r in enumerate(recs):
            for x in r["r"]:
                if x["label"] != lb or not x.get("accepted"): continue
                if x.get("src") in ("det","redet"):
                    seen += 1
                    if est is None and seen >= cn: est = i
                elif x.get("src") == "prop" and est is None:
                    rem.append(i)
        if rem: out[lb] = (rem, est, bool(stats[lb].get("is_target")))
    return out

def grab(v, i):
    c = cv2.VideoCapture(v); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0: c.release(); return None
    c.set(cv2.CAP_PROP_POS_FRAMES, min(i, n-1)); ok, im = c.read(); c.release()
    return im if ok else None

units = [u.replace(R6+"/","") for u in sorted(glob.glob(f"{R6}/brainco/*/cam_*")+glob.glob(f"{R6}/he/*_ep*"))
         if os.path.exists(f"{u}/frames.json")]
man = {}
for u in units:
    d = simulate(u)
    if not d: continue
    for lb,(rem,est,tgt) in d.items():
        key = f"{u.replace('/','__')}__{lb.replace(' ','')}"
        # 제거 구간의 시작/중간/끝 3장
        picks = sorted(set([rem[0], rem[len(rem)//2], rem[-1]]))
        saved = []
        for i in picks:
            im = grab(f"{R6}/{u}/A_visible.mp4", i)
            if im is None: continue
            fn = f"{OUT}/{key}_f{i}.png"
            cv2.imwrite(fn, im); saved.append(os.path.basename(fn))
        man[key] = dict(unit=u, label=lb, is_target=tgt, n_removed=len(rem),
                        established_at=est, frames=picks, files=saved)
        print(f"{u:<44} {lb:<13} {len(rem):>4}프레임 확립={est} -> {len(saved)}장")
json.dump(man, open(f"{OUT}/manifest.json","w"), ensure_ascii=False, indent=1)
print("완료", OUT)
