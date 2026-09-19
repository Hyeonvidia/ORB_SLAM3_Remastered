struct A { virtual int g(); virtual int f(); virtual ~A(); };
int A::f() { return 1; }
int A::g() { return 2; }
A::~A() {}
