#include <stdexcept>
void work();
int f() { try { work(); } catch (const std::runtime_error&) { return 1; } return 0; }
int g() { try { work(); } catch (const std::logic_error&) { return 1; } return 0; }
