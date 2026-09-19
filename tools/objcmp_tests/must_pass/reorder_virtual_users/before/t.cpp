#include "../x.hpp"
X::X() : a(3, 1), s("hello world, long enough string") {}
int use_stack() { X x; return x.v(); }
void use_heap(Base* p) { delete p; }
Base* make() { return new X; }
