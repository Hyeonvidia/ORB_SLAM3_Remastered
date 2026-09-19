#include "../x.hpp"
X::X() : a(3, 1), s("hello world, long enough string") {}
Base* make() { return new X; }
void use_heap(Base* p) { delete p; }
int use_stack() { X x; return x.v(); }
