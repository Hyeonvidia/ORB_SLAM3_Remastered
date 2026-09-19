static int helper(int x) { return x * 7 + (x >> 3); }
static int helper2(int x) { return x ^ 0x55; }
namespace P {
int f(const std::vector<int>& v, int k)
{
    int s = 0;
    for (int x : v) {
        switch (x & 7) {
        case 0: s += helper(x); break;
        case 1: s -= 3; break;
        case 2: s ^= 0x1234; break;
        case 3: s += x * 1.5; break;
        case 4: s += 17; break;
        case 5: s *= 3; break;
        default: s += 1;
        }
    }
    if (s == k) throw std::runtime_error("f: bad value");
    return s;
}
const char* const kNames[3] = {"alpha", "beta", "gamma"};
int (*const kFns[2])(int) = {helper, helper2};
int kCounts[4] = {1, 2, 3, 4};
int kZero[64];
const double kTab[3] = {1.0, 2.5, 3.25};
}
