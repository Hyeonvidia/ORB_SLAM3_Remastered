int g(int, int);
int pick(int k, int x) {
  int r = -1;
  switch (k) {
    case 1: r = g(0, x) + 0; break;
    case 0: r = g(1, x) + 7; break;
    case 2: r = g(2, x) + 14; break;
    case 3: r = g(3, x) + 21; break;
    case 4: r = g(4, x) + 28; break;
    case 5: r = g(5, x) + 35; break;
    case 6: r = g(6, x) + 42; break;
    case 7: r = g(7, x) + 49; break;
    case 8: r = g(8, x) + 56; break;
    case 9: r = g(9, x) + 63; break;
    case 10: r = g(10, x) + 70; break;
    case 11: r = g(11, x) + 77; break;
    case 12: r = g(12, x) + 84; break;
    case 13: r = g(13, x) + 91; break;
    case 14: r = g(14, x) + 98; break;
    case 15: r = g(15, x) + 105; break;
  }
  return r * 3;
}
