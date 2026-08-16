"""trajectory 단계별 스트립 합성 — 6패널(2행 3열) + 단계별 판정 배지.

한 장만 보고도 "어느 단계에서 처음 어긋났는지"가 읽히게 만드는 것이 목적이다.
각 패널 아래에 정상/이상 배지와 한 줄 근거를 붙인다.
"""
import json, os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f"{HERE}/단계별_스트립/원본패널"          # 렌더 원본 패널
DST = f"{HERE}/../../docs/assets/신규task보고_그림"  # 보고서용 사본
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
M = json.load(open(f"{SRC}/meta.json"))

OK, BAD, WARN = (22, 122, 61), (192, 0, 0), (176, 116, 12)
DARK, GRAY, LINE = (33, 33, 33), (110, 110, 110), (214, 214, 214)

f_hdr = ImageFont.truetype(FONT, 25, index=2)
f_sub = ImageFont.truetype(FONT, 19, index=0)
f_pt = ImageFont.truetype(FONT, 20, index=2)
f_bd = ImageFont.truetype(FONT, 18, index=2)
f_bt = ImageFont.truetype(FONT, 17, index=0)

PW, PH = 470, 353          # 패널 이미지 크기 (4:3)
TITLE_H, BADGE_H, GAP = 30, 56, 12

# 단계별 판정 — 렌더 육안 확인 후 확정한 값
V = json.load(open(os.path.join(os.path.dirname(__file__), "strip_verdicts.json"), encoding="utf-8"))


def panel(img_path, title, verdict, note):
    """제목 + 이미지 + 판정 배지 한 세트."""
    box = Image.new("RGB", (PW, TITLE_H + PH + BADGE_H), "white")
    d = ImageDraw.Draw(box)
    d.text((2, 4), title, font=f_pt, fill=DARK)
    if os.path.exists(img_path):
        im = Image.open(img_path).convert("RGB").resize((PW, PH), Image.LANCZOS)
    else:
        im = Image.new("RGB", (PW, PH), (238, 238, 238))
        ImageDraw.Draw(im).text((14, PH // 2 - 10), "(해당 없음)", font=f_sub, fill=GRAY)
    box.paste(im, (0, TITLE_H))
    col = {"정상": OK, "이상": BAD, "주의": WARN, "미진입": GRAY}[verdict]
    by = TITLE_H + PH + 6
    d.rectangle([0, by, PW, by + BADGE_H - 8], outline=col, width=2)
    d.rectangle([0, by, 6, by + BADGE_H - 8], fill=col)
    d.text((14, by + 4), verdict, font=f_bd, fill=col)
    # 근거는 최대 2줄로 접기
    words, lines, cur = note.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f_bt) > PW - 24 and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    lines.append(cur)
    for i, ln in enumerate(lines[:2]):
        d.text((14 + (58 if i == 0 else 0), by + 5 + i * 21), ln, font=f_bt, fill=DARK)
    return box


def build(key):
    v = V[key]
    m = M.get(key, {})
    stages = [
        ("원본 프레임", f"{SRC}/{key}_p0_orig.png", "정상", v["scene"]),
        ("① GroundingDINO — 2D 박스", f"{SRC}/{key}_p1_box2d.png", v["s1"][0], v["s1"][1]),
        ("② SAM 2.1 — 픽셀 마스크", f"{SRC}/{key}_p2_mask.png", v["s2"][0], v["s2"][1]),
        ("③ Depth", f"{SRC}/{key}_p3_depth.png", v["s3"][0], v["s3"][1]),
        ("④ 역투영 — 점군(위에서 본 모습)", f"{SRC}/{key}_p4_bev.png", v["s4"][0], v["s4"][1]),
        ("⑤ 3D 박스", f"{SRC}/{key}_p5_box3d.png", v["s5"][0], v["s5"][1]),
    ]
    ps = [panel(p, t, vd, nt) for t, p, vd, nt in stages]
    cw = PW * 3 + GAP * 2
    ch = ps[0].height * 2 + GAP
    HDR = 66
    out = Image.new("RGB", (cw, HDR + ch), "white")
    d = ImageDraw.Draw(out)
    d.text((2, 4), v["title"], font=f_hdr, fill=(31, 56, 100))
    d.text((2, 36), v["summary"], font=f_sub, fill=v["scolor"] == "bad" and BAD or DARK)
    d.line([(0, HDR - 6), (cw, HDR - 6)], fill=LINE, width=1)
    for i, p in enumerate(ps):
        out.paste(p, ((i % 3) * (PW + GAP), HDR + (i // 3) * (p.height + GAP)))
    fn = f"{DST}/traj_{key}.png"
    out.save(fn)
    print(f"{fn}  {out.size}  | {v['summary'][:52]}")
    return fn


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    for k in ("T1_oreo_head", "T2_laptop_he", "T3_charger_head",
              "T4_duster_he", "T5_cube_wrist", "T6_apple_head"):
        build(k)
