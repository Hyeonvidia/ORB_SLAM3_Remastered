#!/usr/bin/env python3
"""
Converts an X11 `xwd` dump to PNG.

The container has OpenCV's C++ libraries but no cv2 module and no ImageMagick,
so screenshots taken with `xwd -root` need a converter that leans on nothing but
the standard library.

  xwd -root -silent -out shot.xwd && ./tools/xwd2png.py shot.xwd shot.png
"""
import struct
import sys
import zlib


def read_xwd(path):
    data = open(path, "rb").read()
    h = struct.unpack(">25I", data[:100])
    (header_size, _ver, _fmt, depth, width, height, _xoff, byte_order,
     _bunit, _border, _bpad, bpp, bytes_per_line, _vclass,
     red_mask, green_mask, blue_mask, _bits_rgb, _cmap_entries, ncolors,
     *_rest) = h
    pixels = data[header_size + ncolors * 12:]

    def shift(mask):
        if not mask:
            return 0, 0
        s = 0
        while not (mask >> s) & 1:
            s += 1
        bits = 0
        m = mask >> s
        while (m >> bits) & 1:
            bits += 1
        return s, bits

    rs, _ = shift(red_mask)
    gs, _ = shift(green_mask)
    bs, _ = shift(blue_mask)

    stride = bpp // 8
    if stride not in (3, 4):
        sys.exit(f"unsupported bits_per_pixel: {bpp}")

    rows = []
    for y in range(height):
        base = y * bytes_per_line
        row = bytearray(width * 3)
        for x in range(width):
            off = base + x * stride
            chunk = pixels[off:off + stride]
            if len(chunk) < stride:
                break
            value = int.from_bytes(chunk, "big" if byte_order else "little")
            row[x * 3 + 0] = (value >> rs) & 0xFF
            row[x * 3 + 1] = (value >> gs) & 0xFF
            row[x * 3 + 2] = (value >> bs) & 0xFF
        rows.append(bytes(row))
    return width, height, rows


def write_png(path, width, height, rows):
    raw = b"".join(b"\x00" + r for r in rows)   # filter type 0 per scanline

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    open(path, "wb").write(png)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <in.xwd> <out.png>")
    w, h, rows = read_xwd(sys.argv[1])
    write_png(sys.argv[2], w, h, rows)
    print(f"{sys.argv[2]}  {w}x{h}")
