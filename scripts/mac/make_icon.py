"""Draw the DriftSim app icon: a car seen from above, sliding sideways, with tire marks.

    python scripts/mac/make_icon.py out.png      (1024 x 1024 PNG; needs Pillow)

Used by make_app.sh, which turns the PNG into the app's .icns.
"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 2                    # draw at 2x and downsample for smooth edges
N = 1024 * SS


def s(v):
    return int(round(v * SS))


def to_px(x, y):
    """Icon coordinates (1024 grid, y up) to image pixels."""
    return (x * SS, (1024 - y) * SS)


def rounded_square():
    """macOS-style background tile: a vertical gradient in a rounded square."""
    top, bottom = (46, 50, 62), (18, 20, 26)
    grad = Image.new("RGB", (1, N))
    for j in range(N):
        t = j / (N - 1)
        grad.putpixel((0, j), tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    grad = grad.resize((N, N))
    mask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(mask).rounded_rectangle((s(100), s(100), s(924), s(924)), radius=s(185), fill=255)
    tile = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    tile.paste(grad, (0, 0), mask)
    return tile, mask


def car_sprite(length, width, steer_deg):
    """The car pointing right (+x), centred in its own transparent image."""
    pad = s(40)
    w, h = s(length) + 2 * pad, s(width) + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = w / 2, h / 2
    L, W = s(length), s(width)

    # wheels (drawn first, the body overlaps their inner edge)
    wl, ww = s(64), s(34)
    for fx, steer in ((0.31, steer_deg), (-0.31, 0.0)):
        for fy in (-1, 1):
            wheel = Image.new("RGBA", (wl + 8, wl + 8), (0, 0, 0, 0))
            ImageDraw.Draw(wheel).rounded_rectangle(
                ((wl + 8 - wl) / 2, (wl + 8 - ww) / 2, (wl + 8 + wl) / 2, (wl + 8 + ww) / 2),
                radius=s(8), fill=(12, 12, 14, 255))
            wheel = wheel.rotate(steer, resample=Image.BICUBIC)
            px, py = cx + fx * L, cy + fy * (W / 2 - s(6))
            img.alpha_composite(wheel, (int(px - wheel.width / 2), int(py - wheel.height / 2)))

    # body shell, cockpit glass and a stripe
    d.rounded_rectangle((cx - L / 2, cy - W / 2 + s(14), cx + L / 2, cy + W / 2 - s(14)),
                        radius=s(46), fill=(240, 242, 246, 255))
    d.rounded_rectangle((cx - L * 0.12, cy - W * 0.25, cx + L * 0.22, cy + W * 0.25),
                        radius=s(22), fill=(38, 44, 58, 255))
    d.rectangle((cx - L * 0.50 + s(12), cy - s(9), cx - L * 0.16, cy + s(9)), fill=(255, 122, 26, 255))
    d.rectangle((cx + L * 0.25, cy - s(9), cx + L * 0.50 - s(14), cy + s(9)), fill=(255, 122, 26, 255))
    return img


def main(out):
    tile, mask = rounded_square()

    # car pose: heading psi, drifting left with sideslip beta, on a turn of radius R
    C = (568.0, 548.0)
    psi, beta, R = 58.0, 36.0, 400.0
    length, width = 330.0, 190.0
    vel = math.radians(psi - beta)
    O = (C[0] - R * math.sin(vel), C[1] + R * math.cos(vel))   # centre of the turn, left of travel

    # tire marks from the rear wheels: arcs around the turn centre that fade out behind the car
    sweep, half_w = 1.25, 13.0 * SS                             # radians of arc; half stroke width (px)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    dx, dy = xx - O[0] * SS, (1024 * SS - yy) - O[1] * SS
    dist, ang = np.hypot(dx, dy), np.arctan2(dy, dx)
    fade = np.zeros((N, N))
    h = math.radians(psi)
    for side in (-1, 1):
        wx = C[0] - 0.31 * length * math.cos(h) - side * (width / 2 - 6) * math.sin(h)
        wy = C[1] - 0.31 * length * math.sin(h) + side * (width / 2 - 6) * math.cos(h)
        r = math.hypot(wx - O[0], wy - O[1]) * SS
        t = (math.atan2(wy - O[1], wx - O[0]) - ang) / sweep   # 0 at the wheel, 1 at the end
        edge = np.clip(half_w - np.abs(dist - r) + 0.5, 0.0, 1.0)
        fade = np.maximum(fade, edge * np.where((t >= 0) & (t <= 1), (1 - np.clip(t, 0, 1)) ** 1.4, 0.0))
    marks = Image.new("RGBA", (N, N), (255, 122, 26, 0))
    marks.putalpha(Image.fromarray((235 * fade).astype(np.uint8)))
    clipped = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    clipped.paste(marks, (0, 0), mask)
    tile.alpha_composite(clipped)

    # car with a soft shadow, counter-steering (front wheels point toward the direction of travel)
    car = car_sprite(length, width, steer_deg=-(beta - 6)).rotate(psi, resample=Image.BICUBIC, expand=True)
    shadow = Image.new("RGBA", car.size, (0, 0, 0, 0))
    shadow.putalpha(car.getchannel("A").point(lambda a: int(a * 0.55)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(s(14)))
    cx, cy = to_px(*C)
    tile.alpha_composite(shadow, (int(cx - car.width / 2 + s(10)), int(cy - car.height / 2 + s(16))))
    tile.alpha_composite(car, (int(cx - car.width / 2), int(cy - car.height / 2)))

    tile.resize((1024, 1024), Image.LANCZOS).save(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "icon.png")
