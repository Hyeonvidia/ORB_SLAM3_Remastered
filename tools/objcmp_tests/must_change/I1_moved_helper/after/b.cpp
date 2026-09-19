int keep_b() { return 2; }
__attribute__((noipa)) static int helper(int x) { return x * 5; }
int api(int x) { return helper(x) + 1; }
