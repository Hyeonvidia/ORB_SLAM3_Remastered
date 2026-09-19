namespace { long ycount; long ymax; }
namespace P { long h(long v) { ycount += v; if (v > ymax) ymax = v; return ycount; } }
