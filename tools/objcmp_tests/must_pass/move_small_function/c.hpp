#include <map>
#include <memory>
#include <string>
#include <vector>
namespace P {
struct Node { std::vector<double> v; std::string n; };
int f1(std::map<std::string, int>& m, const std::string& k, std::shared_ptr<Node> p);
int f2(std::map<std::string, int>& m, const std::string& k, std::shared_ptr<Node> p);
int f3(std::map<std::string, int>& m, const std::string& k, std::shared_ptr<Node> p);
int h(int);
}
