#include "../common.hpp"
namespace P {
int g(int a) { if (a > 3) throw std::logic_error("g"); return a + 1; }
}
