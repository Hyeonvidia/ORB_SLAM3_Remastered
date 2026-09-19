#include <stdexcept>
void work();
int guarded() { try { work(); } catch (const std::logic_error&) { return 1; } return 0; }
