static int a = 1, b = 2;
static const double t1[4] = {1.5, 2.5, 3.5, 4.5}, t2[4] = {9.5, 8.5, 7.5, 6.5};
void set(int v, int w) { a = v; b = w; }
int sum() { return a + b; }
int get() { return a; }
double both(int i) { return t1[i] + t2[i]; }
double pick(int i) { return t1[i]; }
