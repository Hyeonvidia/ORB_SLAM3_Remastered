namespace P {
Base::~Base() {}
int A::read(std::istream& is) { int s = 0; for (int i = helper(is); i > 0; i -= 3) { s += i * i; if (s > 1000) break; } return s; }
int B::read(std::istream& is) { int s = 0; for (int i = helper(is); i > 0; i -= 3) { s += i * i; if (s > 1000) break; } return s; }
}
