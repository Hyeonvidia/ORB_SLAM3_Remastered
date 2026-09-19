#include <algorithm>
#include <vector>
namespace ORB_SLAM3 {
struct ByAsc  { bool operator()(int a, int b) const { return a < b; } };
struct ByDesc { bool operator()(int a, int b) const { return a > b; } };
void order(std::vector<int>& v) { std::sort(v.begin(), v.end(), ByDesc()); }
void other(std::vector<int>& v) { std::sort(v.begin(), v.end(), ByDesc()); }
}
