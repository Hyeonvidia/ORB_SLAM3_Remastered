#include <string>
std::string make_name();
static const std::string kA = "alpha";
static const std::string kB = make_name();
const std::string& a() { return kA; }
