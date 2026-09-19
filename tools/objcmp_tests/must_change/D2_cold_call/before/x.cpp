#define HOT __attribute__((noipa, section(".text.helpers")))
HOT static int h1(int x) { return x + 1; }
HOT static int h2(int x) { return x * 3; }
int f(int x) { return h1(x) + 7; }
int g(int x) { return h1(x) + h2(x); }
