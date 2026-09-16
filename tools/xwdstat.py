# Reads an XWD dump and reports whether anything was actually drawn.
# XWD header is 100 big-endian uint32s; pixels follow the colormap.
import struct, sys, collections
data = open(sys.argv[1], 'rb').read()
hdr = struct.unpack('>25I', data[:100])
(header_size, file_version, pixmap_format, pixmap_depth, pixmap_width,
 pixmap_height, xoffset, byte_order, bitmap_unit, bitmap_bit_order,
 bitmap_pad, bits_per_pixel, bytes_per_line, visual_class, red_mask,
 green_mask, blue_mask, bits_per_rgb, colormap_entries, ncolors,
 window_width, window_height, window_x, window_y, window_bdrwidth) = hdr
pix_start = header_size + ncolors * 12
pix = data[pix_start:]
print(f"size        : {pixmap_width}x{pixmap_height} depth={pixmap_depth} bpp={bits_per_pixel}")
print(f"pixel bytes : {len(pix)}")
# Sample every 4th pixel word and count distinct values.
words = [pix[i:i+4] for i in range(0, min(len(pix), bytes_per_line*pixmap_height), 16)]
counts = collections.Counter(words)
print(f"distinct sampled colours: {len(counts)}")
top = counts.most_common(4)
total = sum(counts.values())
for w, c in top:
    print(f"  {w.hex()}  {100*c/total:5.1f}%")
print("VERDICT: " + ("blank (single colour)" if len(counts) <= 1 else "content rendered"))
