#include <stdexcept>
void work();
int guarded() { try { work(); } catch (const std::runtime_error&) { return 1; } return 0; }
