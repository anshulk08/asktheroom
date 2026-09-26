#!/bin/bash
# grab a live frame every 15 s for ~2 h
cd "$(dirname "$0")/frames"
for i in $(seq 1 480); do
  ts=$(date +%H%M%S)
  curl -s -m 5 -o live_$ts.jpg http://localhost:8080/frame.jpg || rm -f live_$ts.jpg
  sleep 15
done
