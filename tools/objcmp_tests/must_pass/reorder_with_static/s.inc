static int __attribute__((noinline)) sa(int x, int y) { int s = 0; for (int i = 0; i < x; ++i) s += i * y ^ (s >> 2); return s; }
