__attribute__((noipa)) static int h1(int x) { return x + 1; }
__attribute__((noipa)) static int h2(int x) { return x * 3; }
int f(int x) { return h2(x); }
int g(int x) { return h1(x) + h2(x); }
