@echo off
rem Builds and starts Neural Anatomy. Installs Rust for this user first if it is missing.
setlocal
cd /d "%~dp0"
title Neural Anatomy
set "LOG=%~dp0build-log.txt"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
echo Neural Anatomy > "%LOG%"

where cargo >nul 2>&1
if errorlevel 1 (
  echo Rust is not installed. Downloading the Rust installer...
  curl.exe -sSfL -o "%TEMP%\rustup-init.exe" https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-gnu/rustup-init.exe >> "%LOG%" 2>&1
  if errorlevel 1 ( echo Could not download the Rust installer. See build-log.txt & pause & exit /b 1 )
  echo Installing Rust for this user, about 300 MB. This takes a few minutes...
  "%TEMP%\rustup-init.exe" -y --default-host x86_64-pc-windows-gnu --default-toolchain stable --profile minimal >> "%LOG%" 2>&1
  if errorlevel 1 ( echo Rust could not be installed. See build-log.txt & pause & exit /b 1 )
)

echo Building the app...
cargo build --release >> "%LOG%" 2>&1
if not errorlevel 1 goto built
rem A Rust installed earlier for Visual Studio needs its C++ tools to link. If they
rem are missing, build with the GNU toolchain instead, which brings its own linker.
findstr /c:"link.exe" "%LOG%" >nul 2>&1
if errorlevel 1 ( echo The build failed. See build-log.txt & pause & exit /b 1 )
echo Your Rust needs Visual Studio's C++ tools; using the GNU toolchain instead...
rustup toolchain install stable-x86_64-pc-windows-gnu --profile minimal >> "%LOG%" 2>&1
cargo +stable-x86_64-pc-windows-gnu build --release >> "%LOG%" 2>&1
if errorlevel 1 ( echo The build failed. See build-log.txt & pause & exit /b 1 )
:built

echo.
echo Starting Neural Anatomy. Keep this window open while you use the app;
echo closing this window also closes the app.
echo.
target\release\neural-anatomy.exe
echo.
echo Neural Anatomy has stopped (exit code %errorlevel%).
pause
