## Enhanced Hand-Controlled Visuals (OpenCV + MediaPipe)
Interactive visualizer that uses real-time hand tracking to drive four effects: Kaleidoscope Particles, Reactive Aurora, Ripple Rings, and an animated EKG. Finger openness (per hand) controls each mode’s primary/secondary parameters. HUD overlays show FPS, mode, and live slider values. Designed for smooth 60 FPS with exponential smoothing and particle optimizations.

## Why
A compact playground for gesture-driven visuals you can extend into XR/installation art, VJ sets, or accessibility-oriented UIs.

## Features

Real-time hand tracking (MediaPipe Hands) with per-hand “openness” signals

4 visual modes with parameter mapping + camera preview HUD

Stable smoothing filters for jitter-free control

Clean architecture: tracker → normalized sliders → visual modules

## Controls

- 1/2/3/4 switch modes

- C toggle camera preview

- H toggle HUD

- Q quit
Left hand = primary parameter, Right hand = secondary parameter.

Quick start
```
pip install mediapipe opencv-python numpy
python hand_visuals2.py
```
Requires a webcam (defaults to 1200×700 @ 60 FPS).
