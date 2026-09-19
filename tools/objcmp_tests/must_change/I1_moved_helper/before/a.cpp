__attribute__((noipa)) static int helper(int x) { return x * 3; }
int api(int x) { return helper(x) + 1; }
int keep_a() { return 1; }
