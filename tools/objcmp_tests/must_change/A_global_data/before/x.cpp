int g_threshold = 50;
extern const int kTable[4] = {1, 2, 3, 4};
int over(int d) { return d > g_threshold; }
int at(int i) { return kTable[i]; }
