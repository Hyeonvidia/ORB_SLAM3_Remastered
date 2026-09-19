#include <istream>
namespace P {
struct Base { virtual ~Base(); virtual int read(std::istream&) = 0; };
struct A : Base { int read(std::istream& is) override; int x = 0; };
struct B : Base { int read(std::istream& is) override; int y = 0; };
int u(std::istream&);
static int __attribute__((noinline)) helper(std::istream& is) { int v = 0; is >> v; return v; }
}
