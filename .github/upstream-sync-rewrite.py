# Minimal reproduction of the original extraction's message rewriting, kept
# byte-identical to scripts/ so the workflow file is self-contained.
import re
from datetime import datetime, timezone
T1 = datetime(2026, 1, 19, 19, 18, tzinfo=timezone.utc).timestamp()
T2 = datetime(2026, 7, 13, 16, 43, tzinfo=timezone.utc).timestamp()
msg = commit.message.decode("utf-8", "replace")
date = int(commit.author_date.split()[0])
if date < T1:
    new = re.sub(r"\(#(\d+)\)", r"(sgl-project/sglang#\1)", msg)
elif date <= T2:
    new = re.sub(r"\(#(\d+)\)", r"(lightseekorg/smg#\1)", msg)
else:
    new = msg
if new != msg:
    commit.message = new.encode("utf-8")
