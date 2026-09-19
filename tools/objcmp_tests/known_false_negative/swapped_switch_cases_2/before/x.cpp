int pick(int k, int x) {
  switch (k) {
    case 0: return x + 11; case 1: return x * 37; case 2: return x - 7; case 3: return x ^ 5;
    case 4: return x << 2; case 5: return x / 3; case 6: return x % 9; case 7: return ~x; }
  return 0;
}
