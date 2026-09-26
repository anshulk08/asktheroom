"""Printable ArUco markers (DICT_4X4_50) as an exact-size vector PDF on US Letter.

    python scripts/make_markers.py                       # table markers 0-3 -> markers_0-3.pdf
    python scripts/make_markers.py --ids 4 5 6 7         # floor markers (stretch D15)
    python scripts/make_markers.py --size-mm 70 --out x.pdf

Print at 100% / "Actual size" (never "fit to page"), then check the 10 cm ruler with a real ruler.
Matte paper: glossy paper reflects the lamp and breaks detection. Tape each marker flat with its centre on
the table position listed in config.yaml (table.markers, marker 0 = origin, clockwise from top-left).
"""
from __future__ import annotations

import argparse

import cv2

PT_PER_MM = 72 / 25.4
PAGE_W, PAGE_H = 612.0, 792.0            # US Letter in points
LABELS = {0: "top-left (origin)", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}


def cells(marker_id: int) -> list[list[bool]]:
    """6 x 6 grid (4 x 4 data + 1-cell black border); True = black."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.aruco.generateImageMarker(d, marker_id, 6, borderBits=1)
    return [[img[r, c] < 128 for c in range(6)] for r in range(6)]


def pdf(ids: list[int], size_mm: float) -> bytes:
    s = size_mm * PT_PER_MM                  # marker side in points
    quiet = 10 * PT_PER_MM                    # white margin kept around each marker (cut line)
    slot_w, slot_h = PAGE_W / 2, (PAGE_H - 90) / 2
    ops = ["0 g"]
    for k, mid in enumerate(ids[:4]):
        col, row = [(0, 0), (1, 0), (1, 1), (0, 1)][k]    # clockwise from top-left, as on the table
        x0 = col * slot_w + (slot_w - s) / 2
        y0 = PAGE_H - 50 - (row + 1) * slot_h + (slot_h - s) / 2 + 12
        cell = s / 6
        ops.append(f"0 g {x0:.3f} {y0:.3f} {s:.3f} {s:.3f} re f 1 g")   # one black square, white cells on
        for r, line in enumerate(cells(mid)):                              # top: no seams between black cells
            for c, black in enumerate(line):
                if not black:
                    ops.append(f"{x0 + c * cell:.3f} {y0 + (5 - r) * cell:.3f} {cell:.3f} {cell:.3f} re f")
        ops.append("0 g")
        ops.append(f"0.6 G 0.5 w [4 3] 0 d {x0 - quiet:.2f} {y0 - quiet:.2f} {s + 2 * quiet:.2f} "
                   f"{s + 2 * quiet:.2f} re S [] 0 d")
        label = f"ID {mid}" + (f"  {LABELS[mid]}" if mid in LABELS else "") + f"  ({size_mm:g} mm)"
        ops.append(f"BT /F1 11 Tf {x0 - quiet:.2f} {y0 - quiet - 14:.2f} Td ({label}) Tj ET")
    ruler = 100 * PT_PER_MM                   # 10 cm scale check
    ops.append(f"0 G 1 w 50 40 m {50 + ruler:.2f} 40 l S 50 35 m 50 45 l S {50 + ruler:.2f} 35 m "
               f"{50 + ruler:.2f} 45 l S")
    ops.append(f"BT /F1 10 Tf {60 + ruler:.2f} 37 Td (= 10 cm. Print at Actual size, matte paper.) Tj ET")
    stream = "\n".join(ops).encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:g} {PAGE_H:g}] "
        f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3], help="up to 4 marker ids")
    ap.add_argument("--size-mm", type=float, default=60.0, help="black square side")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = a.out or f"markers_{a.ids[0]}-{a.ids[-1]}.pdf"
    with open(out, "wb") as f:
        f.write(pdf(a.ids, a.size_mm))
    print(f"wrote {out}: ids {a.ids[:4]}, {a.size_mm:g} mm")


if __name__ == "__main__":
    main()
