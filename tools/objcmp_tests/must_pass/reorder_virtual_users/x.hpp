#include <string>
#include <vector>
struct Base { virtual ~Base() {} virtual int v() const = 0; };
struct X : Base {
    std::vector<int> a; std::string s;
    X();
    ~X() override { a.clear(); }
    int v() const override { return (int)a.size() + (int)s.size(); }
};
