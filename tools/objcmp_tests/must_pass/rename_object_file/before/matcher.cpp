#include <vector>
template <class T> __attribute__((noinline)) T twice(T x) { return x * 2; }
static int helper(const std::vector<int>& v) { int s = 0; for (int x : v) s += twice(x); return s; }
int search(int n) { std::vector<int> v(n, 2); return helper(v) + twice((int)v.size()); }
