#!/bin/bash
# Builds and starts Neural Anatomy on macOS or Linux.
# Installs Rust for this user first if it is missing (rustup, about 300 MB).
set -u
cd "$(dirname "$0")" || exit 1
LOG="$PWD/build-log.txt"
echo "Neural Anatomy $(date)" > "$LOG"
export PATH="$HOME/.cargo/bin:$PATH"

pause() { if [ -t 0 ]; then read -r -p "Press Return to close this window." _; fi; }

if [ "$(uname)" = "Darwin" ] && ! xcode-select -p >/dev/null 2>&1; then
  echo "Apple's command line tools are needed to build the app (they provide the linker)."
  echo "A window will ask you to install them. When that has finished, open this again."
  xcode-select --install
  pause
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust is not installed. Installing it for this user (about 300 MB, a few minutes)..."
  if ! curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >> "$LOG" 2>&1; then
    echo "Rust could not be installed. See desktop/build-log.txt"
    pause
    exit 1
  fi
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi

echo "Building the app (the first build takes a minute or two)..."
if ! cargo build --release >> "$LOG" 2>&1; then
  echo "The build failed. See desktop/build-log.txt"
  pause
  exit 1
fi

echo
echo "Starting Neural Anatomy. Keep this window open while you use the app;"
echo "closing the app's window stops it."
echo
./target/release/neural-anatomy "$@"
status=$?
echo
echo "Neural Anatomy has stopped (exit code $status)."
pause
