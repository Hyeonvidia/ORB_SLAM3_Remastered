#!/usr/bin/env python3
"""
Adds back the includes the compiler says are missing, after
tools/prune_includes.py has removed the redundant ones.

The pruner cannot know whether a header needs a complete type -- base classes,
by-value members and inline bodies all do -- so it removes and lets the build
decide. This reads the build log and puts an include where the error is:

  - an error inside a header means that header really does need the definition,
    so the include goes back there
  - an error inside a .cpp means that file had been relying on the definition
    arriving through some other header, so the include belongs to it now

Run it between builds until the build is clean.

  ./tools/fix_includes.py build/compile.log
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Types that do not live in a header named after them. Angle-bracket entries
# are emitted as system includes.
EXTRA = {
    "Verbose": "Verbose.hpp",
    "Settings": "Settings.hpp",
    "EIGEN_MAKE_ALIGNED_OPERATOR_NEW": "<Eigen/Core>",
    "Pinhole": "CameraModels/Pinhole.hpp",
    "KannalaBrandt8": "CameraModels/KannalaBrandt8.hpp",
}

ERROR_PATTERNS = [
    re.compile(r"^(?P<file>[^:]+):\d+:\d+: error: invalid use of incomplete type "
               r"'(?:class |struct )?(?:ORB_SLAM3::)?(?P<type>[A-Za-z_]\w*)'"),
    re.compile(r"^(?P<file>[^:]+):\d+:\d+: error: '(?P<type>[A-Za-z_]\w*)' "
               r"(?:was not declared|has not been declared|does not name a type)"),
    re.compile(r"^(?P<file>[^:]+):\d+:\d+: error: forward declaration of "
               r"'(?:class |struct )?(?:ORB_SLAM3::)?(?P<type>[A-Za-z_]\w*)'"),
    re.compile(r"^(?P<file>[^:]+):\d+:\d+: error: incomplete type "
               r"'(?:ORB_SLAM3::)?(?P<type>[A-Za-z_]\w*)' used"),
]


def header_for(type_name, known):
    if type_name in EXTRA:
        return EXTRA[type_name]
    if type_name in known:
        return f"{type_name}.hpp"
    return None


def insert_include(path, header):
    text = path.read_text(errors="replace")
    system = header.startswith("<")
    # Match at a path boundary: "KeyFrame.hpp" must not look like "Frame.hpp".
    needle = re.escape(header.strip("<>"))
    if re.search(r'#include\s*[<"](?:[^>"]*/)?' + needle + r'[>"]', text):
        return False
    includes = list(re.finditer(r'^#include[ \t]*[<"][^>"]+[>"][ \t]*\r?\n', text, re.M))
    line = f"#include {header}\n" if system else f'#include "{header}"\n'
    if includes:
        at = includes[-1].end()
    else:
        # No includes yet: land after the include guard or #pragma once.
        guard = re.search(r"^#pragma once[ \t]*\r?\n|^#define[ \t]+\w+[ \t]*\r?\n", text, re.M)
        at = guard.end() if guard else 0
    path.write_text(text[:at] + line + text[at:])
    return True


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <compile.log>")
    log = pathlib.Path(sys.argv[1]).read_text(errors="replace")
    known = {h.stem for h in (ROOT / "include").rglob("*.hpp")}

    wanted = {}
    for line in log.splitlines():
        for pattern in ERROR_PATTERNS:
            m = pattern.match(line)
            if not m:
                continue
            header = header_for(m.group("type"), known)
            if not header:
                continue
            # Build logs carry container paths.
            f = m.group("file").replace("/workspace/", "")
            path = ROOT / f
            if path.is_file() and path.suffix in {".hpp", ".cpp"} and path.name != header:
                wanted.setdefault(path, set()).add(header)
            break

    if not wanted:
        print("nothing to add")
        return 0

    added = 0
    for path in sorted(wanted):
        done = [h for h in sorted(wanted[path]) if insert_include(path, h)]
        if done:
            added += len(done)
            print(f"{path.relative_to(ROOT)}: + {', '.join(done)}")
    print(f"\nadded {added} includes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
