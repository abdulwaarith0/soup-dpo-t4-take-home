"""Prefix every line of stdin with a UTC timestamp. Usage: cmd 2>&1 | python scripts/ts.py LOGFILE

Progress bars redraw with carriage returns; each redraw is kept as its own line
so the log is raw, not a tidied summary.
"""

import sys
from datetime import datetime, timezone


def main() -> None:
    out = open(sys.argv[1], "a", encoding="utf-8", newline="\n")
    buf = ""
    while True:
        chunk = sys.stdin.read(1)
        if not chunk:
            break
        if chunk in "\r\n":
            if buf:
                line = f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} {buf}"
                print(line, flush=True)
                out.write(line + "\n")
                out.flush()
            buf = ""
        else:
            buf += chunk
    if buf:
        line = f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} {buf}"
        print(line, flush=True)
        out.write(line + "\n")
    out.close()


if __name__ == "__main__":
    main()
