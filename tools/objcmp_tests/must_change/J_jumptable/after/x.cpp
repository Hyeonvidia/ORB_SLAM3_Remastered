int a(); int b(); int c(); int d(); int e();
int dispatch(int k) { switch (k) { case 1: return a(); case 0: return b(); case 2: return c(); case 3: return d(); case 4: return e(); } return -1; }
