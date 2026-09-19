#include <algorithm>
#include <vector>
namespace ORB_SLAM3 {
struct ByAsc  { const float* k; bool operator()(int a, int b) const { return k[a] < k[b]; } };
struct ByDesc { const float* k; bool operator()(int a, int b) const { return k[a] > k[b]; } };
void order(std::vector<int>& v, const float* k) { std::stable_sort(v.begin(), v.end(), ByDesc{k}); }
void other(std::vector<int>& v, const float* k) { std::stable_sort(v.begin(), v.end(), ByDesc{k}); }
void other2(std::vector<int>& v, const float* k) { std::stable_sort(v.begin(), v.end(), ByAsc{k}); }
}
