# Generated modification reports

Everything in this directory is **generated**. Do not edit it by hand.

```bash
./tools/upstream_delta.py --project g2o    --out docs/modifications
./tools/upstream_delta.py --project DBoW2  --out docs/modifications
./tools/upstream_delta.py --project Sophus --out docs/modifications
```

Each `<project>/<path>.diff` is the difference between the file ORB-SLAM3
vendored and its **closest ancestor anywhere in upstream history** — not
upstream's current tip.

That distinction is the whole point. ORB-SLAM3 froze these libraries years ago,
so diffing its fork against a 2024 release reports every upstream commit since
as though ORB-SLAM3 had written it: about 15,000 lines for g2o, which buries the
roughly 1,100 that are actually theirs. Searching history for the nearest blob
and reporting only the residual isolates their work. CRLF is folded to LF first,
because several of these files were committed with Windows line endings and
would otherwise show every line as changed.

A companion tool answers the coarser question — was this file edited at all?

```bash
./tools/classify_vendored.sh all
```

It tests blob identity: if the exact file exists somewhere in upstream history,
ORB-SLAM3 only froze an older revision and changed nothing. That is how
`Sophus: 21 untouched, 0 edited` was established.
