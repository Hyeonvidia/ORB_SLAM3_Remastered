static int __attribute__((noinline)) sb(int x, int y) { int s = 1; for (int i = 0; i < x; ++i) s *= (i + y) | 1; return s; }
int fb(int x) { double d = x * 1.25; if (d > 10.5) return sb(x, 3) + sa(x, 3); return (int)d + sb(x, 7); }
