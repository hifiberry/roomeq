#!/bin/bash

# RoomEQ Optimization Example - Complete curl command with realistic measurement data
# This example shows a typical room measurement with common issues:
# - Bass boost around 80-125Hz (room modes)
# - Mid dip around 500-1000Hz (typical room acoustics)
# - High frequency rolloff above 8kHz (typical speaker/room response)

echo "Running RoomEQ optimization with realistic measurement data..."
echo "Server endpoint: http://localhost:10315/eq/optimize"
echo ""

curl -X POST "http://localhost:10315/eq/optimize" \
  -H "Content-Type: application/json" \
  -H "Accept: text/plain" \
  -N \
  -d '{
    "frequencies": [
      20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
      630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 
      10000, 12500, 16000, 20000
    ],
    "magnitudes": [
      -12.5, -8.2, -4.8, -2.1, 0.5, 2.8, 5.2, 6.8, 7.2, 5.9, 3.1, 1.2, 
      -0.5, -2.8, -4.5, -3.8, -2.1, -1.2, 0.8, 2.1, 1.5, 0.2, -1.8, -3.2, 
      -4.8, -6.2, -8.5, -11.2, -14.8, -18.5, -22.0
    ],
    "target_curve": "weighted_flat",
    "optimizer_preset": "default",
    "filter_count": 8,
    "sample_rate": 48000,
    "add_highpass": true
  }'

echo ""
echo "Optimization completed. Check the streaming output above for:"
echo "- Real-time progress updates"
echo "- Filter details as they are added"
echo "- Complete filter set after each step"
echo "- Frequency response calculations"
echo "- Final optimization results"
