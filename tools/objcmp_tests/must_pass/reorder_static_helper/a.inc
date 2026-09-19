static int __attribute__((noinline)) sa(int x, int y) { int s = 0; for (int i = 0; i < x; ++i) s += i * y ^ (s >> 2); return s; }
int fa(int x) { switch (x & 7) { case 0: return sa(x, 3); case 1: return 9; case 2: return x*x; case 3: return sa(x, 5) + 1; case 4: return 100; case 5: return -x; default: return 0; } }
