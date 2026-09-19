namespace { int gstate; }
namespace P { int g(int v) { gstate += v; return gstate; } }
