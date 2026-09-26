#!/bin/zsh
# Build, install and launch on a simulator, then screenshot. Output goes to build/ (git-ignored).
#   ./run.sh [shot-name] [launch args...]     e.g. ./run.sh sample -mock YES -mockPaused YES
set -e
cd "${0:A:h}"
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
SIM=${SIM:-A2B80560-30EC-41A8-B479-FBB9D1B10974}
NAME=${1:-shot}; shift || true
mkdir -p build/shots
xcodebuild build -project AskTheRoom/AskTheRoom.xcodeproj -scheme AskTheRoom -destination "id=$SIM" \
  -derivedDataPath build/dd -quiet 2>&1 | grep -E "error|warning: " | grep -v "appintents" || true
xcrun simctl terminate $SIM com.adrian.asktheroom 2>/dev/null || true
xcrun simctl install $SIM build/dd/Build/Products/Debug-iphonesimulator/AskTheRoom.app
xcrun simctl launch $SIM com.adrian.asktheroom "$@" >/dev/null
sleep ${WAIT:-2.5}
xcrun simctl io $SIM screenshot build/shots/$NAME.png >/dev/null 2>&1
echo build/shots/$NAME.png
