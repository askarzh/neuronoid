"""Cover images for the Neuronoid article, built from the project's own figures.

Usage: covers.py <img_dir> <out_dir> <fonts_dir>
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

IMG, OUT, FONTS = (Path(a) for a in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)

NAVY = (11, 13, 36)
INK = (240, 241, 250)
MUTED = (214, 217, 238)
ACCENT = (138, 150, 255)   # EGA-like blue, lightened for small text
SECOND = (229, 122, 228)   # EGA-like magenta
PAPER = (246, 246, 242)
DARK_INK = (22, 23, 43)


def plex(size, weight=b"Bold"):
    f = ImageFont.truetype(str(FONTS / "IBMPlexSans-Var.ttf"), size)
    f.set_variation_by_name(weight)
    return f


def pixel(size):
    return ImageFont.truetype(str(FONTS / "Silkscreen-Regular.ttf"), size)


def wrap(draw, text, font, width):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def text_block(draw, xy, text, font, width, fill, spacing=1.18):
    x, y = xy
    for line in wrap(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += int(font.size * spacing)
    return y


def cover_final(name, size, headline, kicker, footer, text_w):
    W, H = size
    BG = (0, 0, 0)
    im = Image.new("RGB", size, BG)
    # Crop the ship (lower part of the end screen) and scale it up, pixel-crisp.
    ship = Image.open(IMG / "final.png").crop((180, 230, 780, 560))
    scale = (H * 0.78) / ship.height if W > H else (W * 0.9) / ship.width
    ship = ship.resize((int(ship.width * scale), int(ship.height * scale)), Image.NEAREST)
    if W > H:
        sx, sy = W - ship.width - 10, (H - ship.height) // 2 + 20
    else:
        sx, sy = (W - ship.width) // 2, H - ship.height - 40
    im.paste(ship, (sx, sy))
    # Fade the ship into the text side.
    fade = Image.new("L", size, 0)
    fd = ImageDraw.Draw(fade)
    if W > H:
        for x in range(W):
            a = max(0, min(255, int(255 * (1 - (x - W * 0.46) / (W * 0.16)))))
            fd.line([(x, 0), (x, H)], fill=a)
    else:
        for y in range(H):
            a = max(0, min(255, int(255 * (1 - (y - H * 0.46) / (H * 0.14)))))
            fd.line([(0, y), (W, y)], fill=a)
    im = Image.composite(Image.new("RGB", size, BG), im, fade)
    d = ImageDraw.Draw(im)
    pad = 64
    d.text((pad, pad), kicker, font=pixel(28), fill=ACCENT)
    y = text_block(d, (pad, pad + 54), headline, plex(54 if W > H else 76), text_w, INK)
    fy = H - pad - 26 if W > H else y + 24
    d.text((pad, fy), footer, font=plex(26 if W > H else 32, b"Regular"), fill=MUTED)
    im.save(OUT / name)


def cover_split(name):
    W, H = 1200, 627
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 92], fill=NAVY)
    d.text((48, 28), "One game, two ways to build an AI player", font=plex(36), fill=INK)
    col = (W - 48 * 3) // 2
    # Left: RL plateau (left panel of the progress chart).
    rl = Image.open(IMG / "progress_en.png").convert("RGB").crop((0, 0, 825, 630))
    rl.thumbnail((col, 330))
    # Right: bounce zones measured by the researcher.
    bz = Image.open(IMG / "bounce_en.png").convert("RGB")
    bz.thumbnail((col, 330))
    for i, (label, pic, caption, color) in enumerate([
        ("SOFTWARE 2.0", rl, "About a day of neural network training. It never cleared all the bricks in round one.", SECOND),
        ("SOFTWARE 3.0", bz, "An LLM measured the bounce physics itself and, over a few tasks, wrote a player that won 10 of 10.", ACCENT),
    ]):
        x = 48 + i * (col + 48)
        d.text((x, 118), label, font=pixel(26), fill=color)
        im.paste(pic, (x + (col - pic.width) // 2, 160))
        text_block(d, (x, 160 + 330 + 18), caption, plex(25, b"SemiBold"), col, DARK_INK, 1.25)
    im.save(OUT / name)


def badge(d, center, text, font):
    w = d.textlength(text, font=font)
    cx, cy = center
    d.rounded_rectangle([cx - w / 2 - 16, cy - font.size * 0.75, cx + w / 2 + 16, cy + font.size * 0.75],
                        font.size, fill=ACCENT)
    d.text((cx - w / 2, cy - font.size * 0.62), text, font=font, fill=NAVY)


def cover_wrote(name, size=(1200, 627)):
    W, H = size
    sq = W == H
    im = Image.new("RGB", size, NAVY)
    d = ImageDraw.Draw(im)
    pad = 64
    y = text_block(d, (pad, 52 if not sq else 80), "Software 3.0 wrote Software 1.0",
                   plex(52 if not sq else 78), W - 2 * pad, INK, 1.12)
    # Page with the task.
    pw, ph = (300, 360) if not sq else (330, 420)
    px, py = pad, (170 if not sq else y + 70)
    d.rounded_rectangle([px, py, px + pw, py + ph], 10, fill=PAPER)
    d.text((px + 24, py + 22), "TASK.md", font=pixel(24), fill=(70, 84, 220))
    text_block(d, (px + 24, py + 64), "Figure out how to play and write a player program.",
               plex(24, b"SemiBold"), pw - 48, DARK_INK, 1.25)
    for i in range(7):
        ly = py + ph - 170 + i * 22
        d.line([(px + 24, ly), (px + pw - 24 - (i % 3) * 40, ly)], fill=(200, 202, 215), width=6)
    cap = plex(26 if not sq else 30, b"SemiBold")
    d.text((px, py + ph + 16), "1 page of text", font=cap, fill=MUTED)
    # Game frame (playfield only) played by the plain-code player.
    fh = ph
    fr = Image.open(IMG / "researcher_round18.png").crop((32, 0, 228, 200))
    fr = fr.resize((int(fr.width * fh / fr.height), fh), Image.NEAREST)
    fx, fy = W - pad - fr.width, py
    im.paste(fr, (fx, fy))
    d.rectangle([fx - 2, fy - 2, fx + fr.width + 2, fy + fr.height + 2], outline=(60, 64, 110), width=2)
    d.text((fx, fy + fr.height + 16), "plain code, no LLM:", font=cap, fill=MUTED)
    d.text((fx, fy + fr.height + 16 + int(cap.size * 1.25)), "10 wins of 10", font=cap, fill=INK)
    # Arrow with an LLM badge.
    ax0, ax1, ay = px + pw + 24, fx - 24, py + ph // 2
    d.line([(ax0, ay), (ax1 - 18, ay)], fill=ACCENT, width=6)
    d.polygon([(ax1, ay), (ax1 - 26, ay - 16), (ax1 - 26, ay + 16)], fill=ACCENT)
    badge(d, ((ax0 + ax1) // 2, ay - 48), "LLM", plex(28))
    if sq:
        text_block(d, (pad, H - 150), "I finally understood Andrej Karpathy, thanks to Arkanoid!",
                   plex(36, b"SemiBold"), W - 2 * pad, ACCENT, 1.2)
    im.save(OUT / name)


cover_final("cover-final-en.png", (1200, 627),
            "I finally understood Andrej Karpathy, thanks to Arkanoid!",
            "SOFTWARE 2.0 > 3.0", "A 1988 DOS game, beaten 10 out of 10 by an LLM-written player", 560)
cover_final("cover-final-ru.png", (1200, 627),
            "Наконец-то я понял Андрея Карпатого с\u00a0помощью Арканоида!",
            "SOFTWARE 2.0 > 3.0", "DOS-игра 1988 года, 10 побед из 10 у игрока, написанного LLM", 580)
cover_final("cover-final-en-square.png", (1080, 1080),
            "I finally understood Andrej Karpathy, thanks to Arkanoid!",
            "SOFTWARE 2.0 > 3.0", "A 1988 DOS game, beaten 10 out of 10", 950)
cover_split("cover-split-en.png")
cover_wrote("cover-wrote-en.png")
cover_wrote("cover-wrote-en-square.png", (1080, 1080))
print("done")
