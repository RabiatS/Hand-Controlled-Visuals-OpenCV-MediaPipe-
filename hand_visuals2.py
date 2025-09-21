import sys
import time
import math
import random
from collections import deque

import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    print("This script requires MediaPipe. Install with: pip install mediapipe opencv-python numpy")
    sys.exit(1)


# -------------- Enhanced Utility: smoothing + lerp --------------
def lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))

class SmoothFilter:
    """Enhanced smoothing filter with multiple methods"""
    def __init__(self, alpha=0.85, history_size=10):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
        self.history = deque(maxlen=history_size)
        
    def update(self, x):
        if x is None:
            return self.value
            
        if not self.initialized:
            self.value = x
            self.initialized = True
            
        # Exponential smoothing
        self.value = self.alpha * self.value + (1 - self.alpha) * x
        
        # Add to history for trend analysis
        self.history.append(x)
        
        return self.value
    
    def get_trend(self):
        """Returns trend direction (-1, 0, 1)"""
        if len(self.history) < 3:
            return 0
        recent = list(self.history)[-3:]
        if recent[-1] > recent[0] + 0.1:
            return 1
        elif recent[-1] < recent[0] - 0.1:
            return -1
        return 0


# -------------- Enhanced Hand + Face tracking --------------
class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.6):
        # Hands only
        self.mp_hands = mp.solutions.hands

        # Some MediaPipe versions don't accept model_complexity for Hands; try it, fall back if needed.
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                model_complexity=1
            )
        except TypeError:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Smoothing filters
        self.left_filter = SmoothFilter(0.88, 15)
        self.right_filter = SmoothFilter(0.88, 15)

        # Hand state tracking (kept in case you expand later)
        self.hand_states = {'left': {}, 'right': {}}

    def compute_hand_openness(self, hand_landmarks):
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
        fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_bases = [3, 6, 10, 14, 18]

        extensions = []
        for tip_idx, base_idx in zip(fingertips, finger_bases):
            tip = np.array([hand_landmarks.landmark[tip_idx].x, hand_landmarks.landmark[tip_idx].y])
            base = np.array([hand_landmarks.landmark[base_idx].x, hand_landmarks.landmark[base_idx].y])
            tip_dist = np.linalg.norm(tip - wrist)
            base_dist = np.linalg.norm(base - wrist)
            if base_dist > 0:
                extensions.append(tip_dist / base_dist)

        avg_extension = np.mean(extensions) if extensions else 0
        openness = np.clip((avg_extension - 0.8) / 0.6, 0.0, 1.0)
        return openness

    def draw_hand_overlay(self, frame, hand_landmarks, handedness, openness):
        h, w = frame.shape[:2]
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

        landmarks = hand_landmarks.landmark
        center_x = int(np.mean([lm.x for lm in landmarks]) * w)
        center_y = int(np.mean([lm.y for lm in landmarks]) * h)

        label = handedness.classification[0].label
        color = (0, 255, 100) if label == 'Left' else (255, 100, 0)

        radius = int(30 + openness * 20)
        cv2.circle(frame, (center_x, center_y), radius, color, 3)
        cv2.circle(frame, (center_x, center_y), int(radius * openness), color, -1, cv2.LINE_AA)

        text = f"{label}: {openness:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + radius + 25
        cv2.rectangle(frame, (text_x - 5, text_y - 20), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return center_x, center_y

    def update(self, frame_bgr):
        """Process hands only; return (left_smooth, right_smooth, hands_detected_bool)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        left_openness, right_openness = None, None
        hands_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            hands_detected = True
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right' (from the user's perspective)
                openness = self.compute_hand_openness(hand_landmarks)
                self.draw_hand_overlay(frame_bgr, hand_landmarks, handedness, openness)
                if label == 'Left':
                    left_openness = openness
                elif label == 'Right':
                    right_openness = openness

        left_smooth = self.left_filter.update(left_openness)
        right_smooth = self.right_filter.update(right_openness)

        return left_smooth, right_smooth, hands_detected

# -------------- Enhanced Visual 1: Improved Kaleido Particles --------------
class EnhancedKaleidoParticles:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.center = np.array([self.w // 2, self.h // 2], dtype=np.float32)
        self.particles = []
        self.max_particles = 8000
        self.trail_particles = []  # For particle trails
        
    def spawn_burst(self, count, speed=120.0):
        """Enhanced particle spawning with brighter colors"""        
        for _ in range(int(count)):
            ang = random.uniform(0, 2*np.pi)                
            r = random.uniform(0, 15.0)
            pos = self.center + r * np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
            
            vel_magnitude = random.uniform(0.4, 1.5) * speed / 60.0
            vel = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32) * vel_magnitude
            
            life = random.uniform(1.0, 4.0)
            size = random.uniform(2.0, 6.0)  # Larger particles
            color_seed = random.random()
            
            self.particles.append([pos, vel, life, life, size, color_seed])

        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]

    def update_particles(self, dt, gravity=0.0):
        """Enhanced particle physics"""
        alive = []
        for pos, vel, life, max_life, size, color_seed in self.particles:
            life -= dt
            if life > 0:
                # Apply physics
                pos += vel * dt * 60.0
                vel *= (1.0 - 0.008 * dt * 60.0)  # air resistance
                vel[1] += gravity * dt * 60.0  # gravity
                
                # Add slight noise for organic movement
                noise = np.random.normal(0, 5.0, 2) * dt
                pos += noise
                
                alive.append([pos, vel, life, max_life, size, color_seed])
                
                # Create trails for bright particles
                if life > max_life * 0.7 and random.random() < 0.1:
                    self.trail_particles.append([pos.copy(), life * 0.3, size * 0.5])
        
        self.particles = alive
        
        # Update trails
        alive_trails = []
        for pos, trail_life, trail_size in self.trail_particles:
            trail_life -= dt * 2.0
            if trail_life > 0:
                alive_trails.append([pos, trail_life, trail_size])
        self.trail_particles = alive_trails

    def render_enhanced(self):
        """Enhanced rendering with brighter, more vibrant colors"""
        base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Render main particles with enhanced brightness
        for pos, vel, life, max_life, size, color_seed in self.particles:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.w and 0 <= y < self.h:
                life_ratio = life / max_life
                alpha = np.clip(life_ratio * 1.5, 0.0, 1.0)  # Brighter particles
                
                # More vibrant color calculation
                hue = (color_seed * 360 + life * 100) % 360
                saturation = 0.9 + 0.1 * math.sin(life * 10)  # High saturation
                brightness = 0.8 + 0.2 * life_ratio  # High brightness
                
                # Convert HSV to RGB with enhanced values
                r = math.sin(hue * math.pi / 180) * saturation + (1 - saturation)
                g = math.sin((hue + 120) * math.pi / 180) * saturation + (1 - saturation)
                b = math.sin((hue + 240) * math.pi / 180) * saturation + (1 - saturation)
                
                # Boost colors significantly
                color = (
                    int(np.clip(b * alpha * brightness * 255 * 1.8, 0, 255)), 
                    int(np.clip(g * alpha * brightness * 255 * 1.8, 0, 255)), 
                    int(np.clip(r * alpha * brightness * 255 * 1.8, 0, 255))
                )
                
                # Draw with size and glow effect
                radius = max(2, int(size * life_ratio))
                cv2.circle(base, (x, y), radius, color, -1)
                # Add glow
                if radius > 2:
                    glow_color = tuple(int(c * 0.3) for c in color)
                    cv2.circle(base, (x, y), radius + 2, glow_color, 1)
        
        # Render brighter trails
        for pos, trail_life, trail_size in self.trail_particles:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.w and 0 <= y < self.h:
                alpha = np.clip(trail_life * 1.5, 0.0, 1.0)
                color = (int(180 * alpha), int(220 * alpha), int(255 * alpha))  # Brighter blue trails
                cv2.circle(base, (x, y), max(1, int(trail_size)), color, -1)
        
        # Enhanced bloom effect for more brightness
        blurred = cv2.GaussianBlur(base, (0, 0), 3.0)
        enhanced = cv2.addWeighted(base, 0.6, blurred, 0.4, 10)  # Added brightness offset
        
        return enhanced

    def create_kaleidoscope(self, img, slices):
        """Improved kaleidoscope effect"""
        if slices < 2:
            return img
            
        out = np.zeros_like(img, dtype=np.float32)
        angle_step = 360.0 / slices
        
        for i in range(int(slices)):
            angle = angle_step * i
            M = cv2.getRotationMatrix2D((self.w/2, self.h/2), angle, 1.0)
            rotated = cv2.warpAffine(img.astype(np.float32), M, (self.w, self.h), 
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # Blend with varying opacity for depth
            blend_factor = 0.8 + 0.2 * math.sin(i * math.pi / slices)
            out += rotated * blend_factor
        
        # Normalize and enhance
        out = np.clip(out / slices * 1.2, 0, 255).astype(np.uint8)
        return out

    def draw(self, left_slider, right_slider, hands_detected, dt):
        """Only show particles when hands are detected"""
        if not hands_detected:
            # Return black canvas when no hands detected
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)
            
        symmetry = int(np.clip(lerp(2, 24, left_slider), 2, 24))  # More symmetry range
        energy = np.clip(right_slider, 0.0, 1.0)

        spawn_rate = lerp(5, 150, energy)  # Higher spawn rate for more control
        speed = lerp(100, 400, energy)  # Faster particles

        self.spawn_burst(spawn_rate, speed=speed)
        self.update_particles(dt, gravity=0)  # No gravity

        base = self.render_enhanced()
        result = self.create_kaleidoscope(base, symmetry)
        
        return result


# -------------- Enhanced Visual 2: Reactive Aurora --------------
class ReactiveAurora:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.t = 0.0
        self.wave_offset = 0.0
        
    def draw(self, left_slider, right_slider, hands_detected, dt):
        self.t += dt
        self.wave_offset += dt * lerp(0.1, 2.0, right_slider)
        
        # Enhanced parameters
        turbulence = lerp(0.3, 6.0, left_slider)
        intensity = lerp(0.3, 1.5, right_slider)
        hue_shift = int(lerp(0, 179, self.t * 0.1))  # Slowly cycling hue
        
        # Create more complex wave patterns
        x = np.linspace(0, 1, self.w, dtype=np.float32)
        y = np.linspace(0, 1, self.h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        
        # Multiple wave layers for aurora effect
        wave1 = np.sin((xv * turbulence + self.wave_offset) * 2 * np.pi)
        wave2 = np.sin((yv * turbulence + self.wave_offset * 1.3) * 2 * np.pi)
        wave3 = np.sin(((xv + yv) * turbulence * 0.7 + self.wave_offset * 0.8) * 2 * np.pi)
        wave4 = np.sin(((xv - yv) * turbulence * 1.2 + self.wave_offset * 0.6) * 2 * np.pi)
        
        # Combine waves
        combined = (wave1 + wave2 + wave3 + wave4) * 0.25 + 0.5
        
        # Apply intensity
        combined = np.clip(combined * intensity, 0, 1)
        
        # Enhanced blur for aurora softness
        img = (combined * 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 4.0)
        
        # Create colorful aurora
        hsv = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        hsv[..., 0] = (img // 2 + hue_shift) % 180
        hsv[..., 1] = np.clip(combined * 255 * 1.4, 100, 255).astype(np.uint8)
        hsv[..., 2] = np.clip(combined * 255 * 1.6, 80, 255).astype(np.uint8)
        
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr


# -------------- Enhanced Visual 3: Ripple Rings --------------
class EnhancedRippleRings:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.rings = []  # each: (x, y, radius, alpha, color_hue)
        self.t = 0.0
        self.spawn_acc = 0.0

    def draw(self, left_slider, right_slider, hands_detected, dt):
        self.t += dt
        # left_slider: spawn rate (0..50 Hz)
        # right_slider: damping/ring speed (0.88..0.998)
        spawn_rate = lerp(0.0, 50.0, left_slider)
        damping = lerp(0.88, 0.998, right_slider)
        ring_speed = lerp(80.0, 280.0, right_slider)

        self.spawn_acc += spawn_rate * dt
        while self.spawn_acc >= 1.0:
            # Random spawn locations
            cx = random.randint(int(self.w*0.1), int(self.w*0.9))
            cy = random.randint(int(self.h*0.1), int(self.h*0.9))
            
            # Random colors for each ring
            color_hue = random.randint(0, 179)
            self.rings.append([cx, cy, 2.0, 1.0, color_hue])
            self.spawn_acc -= 1.0

        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        alive = []
        for cx, cy, r, a, hue in self.rings:
            r += ring_speed * dt
            a *= damping
            if a > 0.02 and r < max(self.w, self.h):
                # Create HSV color with random hue
                color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0,0]
                color = tuple(int(c * a) for c in color_bgr)
                
                thickness = max(1, int(r * 0.015))
                cv2.circle(canvas, (int(cx), int(cy)), int(r), color, thickness)
                alive.append([cx, cy, r, a, hue])
        self.rings = alive

        # Enhanced post-processing
        canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5)
        return canvas


# -------------- Enhanced Visual 4: EKG Visualizer --------------
class EnhancedEKGVisualizer:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.x = 0
        self.t = 0.0
        self.prev_val = self.h // 2
        self.scroll_speed = 2

    def ekg_value(self, t, bpm, noise_level, amplitude, face_influence):
        # Enhanced EKG with face tilt influence
        period = 60.0 / max(30.0, bpm)
        phase = (t % period) / period

        v = 0.0
        # P wave
        v += 0.25 * math.exp(-((phase - 0.15) ** 2) / 0.0012)
        # QRS complex
        v -= 0.7 * math.exp(-((phase - 0.28) ** 2) / 0.0003)  # Q
        v += 2.2 * math.exp(-((phase - 0.30) ** 2) / 0.00008) # R (enhanced)
        v -= 1.1 * math.exp(-((phase - 0.32) ** 2) / 0.0003)  # S
        # T wave
        v += 0.5 * math.exp(-((phase - 0.55) ** 2) / 0.004)

        # Face tilt adds baseline drift
        baseline_drift = (face_influence - 0.5) * 0.3
        v += baseline_drift
        
        # Enhanced noise with occasional artifacts
        if random.random() < 0.02:  # 2% chance of artifact
            v += random.uniform(-0.5, 0.5)
        v += (np.random.randn() * noise_level)

        return v * amplitude

    def draw(self, left_slider, right_slider, hands_detected, dt):
        self.t += dt
        # left_slider: heart rate (BPM 28..200) - Now goes as low as 28
        # right_slider: amplitude and noise
        bpm = lerp(28.0, 200.0, left_slider)  # Lower minimum BPM
        amplitude = lerp(15.0, 90.0, right_slider)
        noise_level = lerp(0.01, 0.08, right_slider)
        
        # Fixed scroll speed for smoother animation
        scroll_multiplier = 2.0
        
        val = self.ekg_value(self.t, bpm, noise_level, amplitude, 0.5)
        y = int(self.h // 2 - val)
        y = np.clip(y, 10, self.h - 10)

        # Variable scroll speed
        pixels_to_scroll = max(1, int(self.scroll_speed * scroll_multiplier))
        
        # Scroll canvas left
        if pixels_to_scroll < self.w:
            self.canvas[:, :-pixels_to_scroll] = self.canvas[:, pixels_to_scroll:]
            self.canvas[:, -pixels_to_scroll:] = (0, 0, 0)

        # Draw new line segment
        for i in range(pixels_to_scroll):
            x_pos = self.w - pixels_to_scroll + i
            if x_pos > 0:
                interp_y = int(self.prev_val + (y - self.prev_val) * (i / pixels_to_scroll))
                cv2.line(self.canvas, (x_pos - 1, self.prev_val), (x_pos, interp_y), (0, 255, 100), 3)

        # Draw grid FIRST, then EKG line on top
        grid = np.zeros_like(self.canvas)
        grid_alpha = 0.4 + 0.1 * math.sin(self.t * 0.5)  # Pulsing grid
        grid_color = (int(50 * grid_alpha), int(50 * grid_alpha), int(50 * grid_alpha))
        
        # Major grid lines
        for x in range(0, self.w, 50):
            cv2.line(grid, (x, 0), (x, self.h), grid_color, 1)
        for y_grid in range(0, self.h, 50):
            cv2.line(grid, (0, y_grid), (self.w, y_grid), grid_color, 1)
            
        # Minor grid lines
        minor_color = tuple(int(c * 0.6) for c in grid_color)
        for x in range(25, self.w, 50):
            cv2.line(grid, (x, 0), (x, self.h), minor_color, 1)

        # Combine: grid first, then EKG line on top
        result = cv2.addWeighted(grid, 1.0, self.canvas, 1.0, 0)

        # Heart rate indicator on top
        cv2.putText(result, f"{int(bpm)} BPM", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2, cv2.LINE_AA)

        self.prev_val = y
        return result


# -------------- Main Application --------------
def main():
    W, H = 1200, 700  # Larger window for better experience
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Increase FPS to 60

    if not cap.isOpened():
        print("Could not open webcam. Make sure a webcam is connected.")
        return

    # Initialize trackers and visualizations
    tracker = HandTracker()
    vis_kaleido = EnhancedKaleidoParticles(W, H)
    vis_aurora = ReactiveAurora(W, H)
    vis_ripples = EnhancedRippleRings(W, H)
    vis_ekg = EnhancedEKGVisualizer(W, H)

    mode = 1  # Start with kaleidoscope
    prev_time = time.time()
    fps_counter = deque(maxlen=30)
    
    # UI Configuration
    show_controls = True
    show_camera = True
    camera_size = 300  # Larger camera preview
    
    print("Enhanced Hand-Controlled Visuals Started!")
    print("Controls:")
    print("  [1/2/3/4] - Switch visual modes")
    print("  [C] - Toggle camera preview")
    print("  [H] - Toggle help/controls")
    print("  [Q] - Quit")
    print("\nHand Controls:")
    print("  Left Hand - Primary parameter")
    print("  Right Hand - Secondary parameter")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        # Calculate FPS
        now = time.time()
        dt = max(1e-3, min(0.05, now - prev_time))
        fps_counter.append(1.0 / dt)
        avg_fps = np.mean(fps_counter)
        prev_time = now

        # Mirror frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Get tracking data with visual overlays
        left, right, hands_detected = tracker.update(frame)
        
        # Generate visuals
        if mode == 1:
            visual = vis_kaleido.draw(left, right, hands_detected, dt)
            mode_name = "Kaleidoscope Particles"
            controls = "L:Symmetry | R:Energy | (Hands Required)"
        elif mode == 2:
            visual = vis_aurora.draw(left, right, hands_detected, dt)
            mode_name = "Reactive Aurora"
            controls = "L:Turbulence | R:Intensity"
        elif mode == 3:
            visual = vis_ripples.draw(left, right, hands_detected, dt)
            mode_name = "Ripple Rings"
            controls = "L:Spawn Rate | R:Ring Speed"
        else:  # mode == 4
            visual = vis_ekg.draw(left, right, hands_detected, dt)
            mode_name = "EKG Monitor"
            controls = "L:Heart Rate (28-200) | R:Amplitude"

        # Create camera preview if enabled
        if show_camera:
            small_camera = cv2.resize(frame, (camera_size, int(camera_size * frame.shape[0] / frame.shape[1])))
            # Position camera preview in top-right corner
            y1, y2 = 10, 10 + small_camera.shape[0]
            x1, x2 = W - camera_size - 10, W - 10
            
            # Add border
            cv2.rectangle(visual, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 2)
            visual[y1:y2, x1:x2] = small_camera

        # Enhanced HUD
        if show_controls:
            # Semi-transparent background for text
            overlay = visual.copy()
            
            # Main info
            info_lines = [
                f"Mode: {mode_name}",
                f"FPS: {avg_fps:.1f}",
                f"Left: {left:.2f} | Right: {right:.2f} | Hands: {'✓' if hands_detected else '✗'}",
                controls,
                "",
                "Controls: [1/2/3/4]Mode [C]Camera [H]Help [Q]Quit"
            ]
            
            y_start = 20
            for i, line in enumerate(info_lines):
                if line:  # Skip empty lines
                    # Background rectangle
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(overlay, (10, y_start + i*30 - 5), 
                                (20 + text_size[0], y_start + i*30 + 20), (0, 0, 0), -1)
                    
                    # Text
                    color = (0, 255, 255) if i == 0 else (255, 255, 255)
                    cv2.putText(overlay, line, (15, y_start + i*30 + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # Blend overlay
            visual = cv2.addWeighted(visual, 0.8, overlay, 0.2, 0)

        cv2.imshow("Enhanced Hand-Controlled Visuals", visual)

        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            mode = 1
            print("Switched to Kaleidoscope Particles")
        elif key == ord('2'):
            mode = 2  
            print("Switched to Reactive Aurora")
        elif key == ord('3'):
            mode = 3
            print("Switched to Ripple Rings")
        elif key == ord('4'):
            mode = 4
            print("Switched to EKG Monitor")
        elif key == ord('c'):
            show_camera = not show_camera
            print(f"Camera preview: {'ON' if show_camera else 'OFF'}")
        elif key == ord('h'):
            show_controls = not show_controls
            print(f"Controls display: {'ON' if show_controls else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully!")

if __name__ == "__main__":
    main() 
    #eof
    
