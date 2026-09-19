namespace P {
int f(const int* p, int n, int k)
{
    int s = 0;
    for (int i = 0; i < n; ++i) {
        if (p[i] == k) break;
        s += p[i] * (i ^ k);
    }
    return s;
}
}
