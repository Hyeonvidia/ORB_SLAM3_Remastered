#include <stdexcept>
#include <string>
#include <vector>
namespace P {
int f(const std::vector<int>& v, int k);
int g(int);
int h(int);
extern const char* const kNames[3];
extern int (*const kFns[2])(int);
extern int kCounts[4];
extern int kZero[64];
extern const double kTab[3];
}
