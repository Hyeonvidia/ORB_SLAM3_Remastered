namespace { int fa; int fb; }
namespace P { int f(bool hit) { if (hit) ++fa; else ++fb; return fa - fb; } }
