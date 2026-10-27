#!/bin/bash
# Double-click on a Mac to build and start Neural Anatomy (also runs on Linux).
exec "$(dirname "$0")/desktop/build-and-run.sh" "$@"
