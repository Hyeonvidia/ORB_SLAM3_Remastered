#include <string>
#include <vector>
namespace P {
struct Item { std::string name; int id; };
void f1(std::vector<Item>& v, const Item& it);
void f2(std::vector<Item>& v, const Item& it);
int h(int);
}
