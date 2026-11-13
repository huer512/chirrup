#!/usr/bin/env python3
import os,sys
from pathlib import Path
root=Path('/workspace')
if not Path.cwd().resolve().is_relative_to(root): raise SystemExit("Run this mailbox tool inside its registered project.")
for key in ("MAIL4AGENT_TOKEN", "MAIL4AGENT_URL", "MAIL4AGENT_MAILBOX"): os.environ.pop(key, None)
os.execv('/home/ubuntu/.local/share/mail4agent/runtimes/8fc9d7251d1921438c9860fec0348648456632af8ca07638c7d7789bf5316688/bin/python', ['/home/ubuntu/.local/share/mail4agent/runtimes/8fc9d7251d1921438c9860fec0348648456632af8ca07638c7d7789bf5316688/bin/python', "-I", "-m", "mail4agent", "project", "mail", "--project", '/workspace', "--", *sys.argv[1:]])
