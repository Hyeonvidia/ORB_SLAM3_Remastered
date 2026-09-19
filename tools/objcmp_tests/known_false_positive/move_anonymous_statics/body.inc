namespace {
int hits;
int misses;
double total;
}
namespace P {
int record(bool hit, double v)
{
    if (hit) ++hits; else ++misses;
    total += v;
    return hits - misses;
}
}
