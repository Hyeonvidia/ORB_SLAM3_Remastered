struct A { virtual int f(); virtual int g(); virtual ~A(); };
int A::f() { return 1; }
int A::g() { return 2; }
A::~A() {}
