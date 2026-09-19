#include "../common.hpp"
static const char* msg(int i) { static const char* m[] = {"one", "two", "three", "four"}; return m[i & 3]; }
namespace P {
int h(int a) { return std::string(msg(a)).size() + a; }
}
