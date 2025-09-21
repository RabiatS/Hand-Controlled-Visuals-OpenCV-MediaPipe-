
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


# -------------- Utility: smoothing + lerp --------------
def lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))

class LowPass:
    def __init__(self, alpha=0.8, init=0.0):
        self.alpha = alpha
        self.y = init
        self.initialized = False

    def update(self, x):
        if not self.initialized:
            self.y = x
            self.initialized = True
            return self.y
        self.y = self.alpha * self.y + (1 - self.alpha) * x
        return self.y


# -------------- Hand tracking + openness --------------
class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.6, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=0
        )
        self.left_filter = LowPass(0.85, 0.0)
        self.right_filter = LowPass(0.85, 0.0)

    def compute_openness(self, hand_landmarks):
        # Openness = avg distance (in normalized coords) from wrist to fingertips
        # Use WRIST = 0, fingertips = [4, 8, 12, 16, 20]
        idxs = [4, 8, 12, 16, 20]
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
        dists = []
        for i in idxs:
            p = np.array([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y])
            dists.append(np.linalg.norm(p - wrist))
        # Normalize roughly: observed hand sizes ~ 0.1 - 0.4 in normalized space
        raw = np.clip((np.mean(dists) - 0.10) / (0.30), 0.0, 1.0)
        return raw

    def update(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        left = None
        right = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                open_raw = self.compute_openness(hand_landmarks)
                if label == 'Left':
                    left = open_raw
                else:
                    right = open_raw

        left_s = self.left_filter.update(left if left is not None else self.left_filter.y)
        right_s = self.right_filter.update(right if right is not None else self.right_filter.y)
        return left_s, right_s


# -------------- Visual 1: Kaleido Particles --------------
class KaleidoParticles:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.center = np.array([self.w // 2, self.h // 2], dtype=np.float32)
        self.particles = []
        self.max_particles = 5000

    def spawn(self, count, speed=120.0):
        for _ in range(int(count)):
            ang = random.uniform(0, 2*np.pi)
            r = random.uniform(0, 10.0)
            pos = self.center + r * np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
            vel = (np.array([math.cos(ang), math.sin(ang)], dtype=np.float32) *
                   random.uniform(0.2, 1.0) * speed / 60.0)
            life = random.uniform(0.8, 2.0)
            self.particles.append([pos, vel, life])

        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]

    def step(self, dt):
        alive = []
        for pos, vel, life in self.particles:
            life -= dt
            if life > 0:
                pos += vel * dt * 60.0
                vel *= (1.0 - 0.01 * dt * 60.0)  # light drag
                alive.append([pos, vel, life])
        self.particles = alive

    def render_base(self):
        # Draw to a base grayscale canvas
        base = np.zeros((self.h, self.w), dtype=np.uint8)
        for pos, _, life in self.particles:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.w and 0 <= y < self.h:
                val = int(np.clip(200 * life, 0, 255))
                base[y, x] = max(base[y, x], val)
        # slight bloom
        base = cv2.GaussianBlur(base, (0, 0), 2.0)
        return base

    def kaleido(self, gray, slices):
        # Create kaleidoscopic effect by rotating & blending
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out = np.zeros_like(color)
        angle_step = 360.0 / max(1, slices)
        for i in range(int(slices)):
            M = cv2.getRotationMatrix2D((self.w/2, self.h/2), angle_step*i, 1.0)
            rotated = cv2.warpAffine(color, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            out = cv2.max(out, rotated)
        return out

    def draw(self, left_slider, right_slider, dt):
        # left_slider -> symmetry 3..16
        # right_slider -> energy 0..1 controls spawn rate and speed
        symmetry = int(round(lerp(3, 16, left_slider)))
        energy = float(np.clip(right_slider, 0.0, 1.0))

        spawn = lerp(2, 80, energy)
        speed = lerp(60, 240, energy)

        self.spawn(spawn, speed=speed)
        self.step(dt)

        base = self.render_base()
        img = self.kaleido(base, symmetry)

        # hue sparkle by tiny noise on energy
        hue_shift = int(lerp(0, 180, energy))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img


# -------------- Visual 2: Aurora Noise Field --------------
class AuroraNoise:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.t = 0.0

    def draw(self, left_slider, right_slider, dt):
        self.t += dt
        # left_slider: turbulence scale 0.5..4.0
        # right_slider: hue rotation 0..180 (OpenCV HSV hue)
        scale = lerp(0.5, 4.0, left_slider)
        hue_rot = int(lerp(0, 179, right_slider))

        # Create procedural "plasma" / noise-like field using trig blends
        x = np.linspace(0, 1, self.w, dtype=np.float32)
        y = np.linspace(0, 1, self.h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        t = self.t

        f = (
            np.sin((xv*scale + t*0.10)*2*np.pi) +
            np.sin((yv*scale + t*0.13)*2*np.pi) +
            np.sin(((xv+yv)*scale*0.7 + t*0.07)*2*np.pi) +
            np.sin(((xv-yv)*scale*1.3 + t*0.05)*2*np.pi)
        ) * 0.25 + 0.5

        # Smooth with Gaussian for aurora feel
        img = (f * 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 3.0)

        hsv = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        hsv[..., 0] = (img + hue_rot) % 180
        hsv[..., 1] = np.clip((f*255*1.2), 80, 255).astype(np.uint8)  # saturation
        hsv[..., 2] = np.clip((f*255*1.5), 60, 255).astype(np.uint8)  # value
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr


# -------------- Visual 3: Ripple Rings --------------
class RippleRings:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.rings = []  # each: (x, y, radius, alpha)
        self.t = 0.0
        self.spawn_acc = 0.0

    def draw(self, left_slider, right_slider, dt):
        self.t += dt
        # left_slider: spawn rate (0..30 Hz)
        # right_slider: damping (0.85..0.995)
        spawn_rate = lerp(0.0, 30.0, left_slider)
        damping = lerp(0.85, 0.995, right_slider)

        self.spawn_acc += spawn_rate * dt
        while self.spawn_acc >= 1.0:
            cx = random.randint(int(self.w*0.1), int(self.w*0.9))
            cy = random.randint(int(self.h*0.1), int(self.h*0.9))
            self.rings.append([cx, cy, 1.0, 1.0])  # start radius, alpha
            self.spawn_acc -= 1.0

        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        alive = []
        for cx, cy, r, a in self.rings:
            r += 180.0 * dt  # expand speed
            a *= damping
            if a > 0.03:
                # draw ring
                thickness = int(max(1, r * 0.02))
                color = (255, 255, 255)
                cv2.circle(canvas, (int(cx), int(cy)), int(r), color, thickness)
                alive.append([cx, cy, r, a])
        self.rings = alive

        # fade with alpha map
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return color


# -------------- Visual 4: EKG Visualizer --------------
class EKGVisualizer:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.x = 0
        self.t = 0.0
        self.prev_val = self.h // 2

    def ekg_value(self, t, bpm, noise_level, amplitude):
        # period in seconds
        period = 60.0 / max(40.0, bpm)
        # normalized phase 0..1
        phase = (t % period) / period

        # Simple synthetic ECG: baseline + P wave + QRS + T wave
        v = 0.0
        # P wave (gentle bump around 0.15)
        v += 0.2 * math.exp(-((phase - 0.15) ** 2) / 0.0009)
        # Q dip then R spike then S dip centered near ~0.30
        v -= 0.6 * math.exp(-((phase - 0.28) ** 2) / 0.0002)  # Q
        v += 1.8 * math.exp(-((phase - 0.30) ** 2) / 0.00005) # R
        v -= 0.9 * math.exp(-((phase - 0.32) ** 2) / 0.0002)  # S
        # T wave (broad bump around ~0.55)
        v += 0.4 * math.exp(-((phase - 0.55) ** 2) / 0.003)

        # Add noise
        v += (np.random.randn() * noise_level)

        return v * amplitude

    def draw(self, left_slider, right_slider, dt):
        # left_slider: heart rate (BPM 40..180)
        # right_slider: amplitude/noise balance
        bpm = lerp(40.0, 180.0, left_slider)
        amplitude = lerp(10.0, 70.0, right_slider)  # px
        noise_level = lerp(0.00, 0.03, right_slider)

        self.t += dt

        # Scroll effect: draw one vertical column per frame
        val = self.ekg_value(self.t, bpm, noise_level, amplitude)
        y = int(self.h // 2 - val)

        # shift canvas left by 1 px
        self.canvas[:, :-1] = self.canvas[:, 1:]
        # draw new column (rightmost)
        self.canvas[:, -1] = (0, 0, 0)
        cv2.line(self.canvas, (self.w - 2, self.prev_val), (self.w - 1, y), (0, 255, 0), 2)

        # grid lines
        grid = self.canvas.copy()
        for x in range(0, self.w, 40):
            cv2.line(grid, (x, 0), (x, self.h), (30, 30, 30), 1)
        for y2 in range(0, self.h, 40):
            cv2.line(grid, (0, y2), (self.w, y2), (30, 30, 30), 1)

        self.prev_val = y
        return grid


# -------------- Main loop --------------
def main():
    W, H = 960, 540
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    if not cap.isOpened():
        print("Could not open webcam. Make sure a webcam is connected and accessible.")
        return

    tracker = HandTracker()

    vis_kaleido = KaleidoParticles(W, H)
    vis_aurora = AuroraNoise(W, H)
    vis_ripples = RippleRings(W, H)
    vis_ekg = EKGVisualizer(W, H)

    mode = 1  # 1=kaleido, 2=aurora, 3=ripples, 4=ekg
    prev_time = time.time()

    help_text = [
        "Controls:",
        "[1] Kaleido-Particles  | L: symmetry  R: energy",
        "[2] Aurora Noise Field | L: turbulence  R: hue",
        "[3] Ripple Rings       | L: spawn rate R: damping",
        "[4] EKG Visual         | L: BPM (40-180) R: amplitude/noise",
        "[q] Quit"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = max(1e-3, min(0.05, now - prev_time))
        prev_time = now

        frame = cv2.flip(frame, 1)  # mirror for UX
        left, right = tracker.update(frame)

        if mode == 1:
            vis = vis_kaleido.draw(left, right, dt)
        elif mode == 2:
            vis = vis_aurora.draw(left, right, dt)
        elif mode == 3:
            vis = vis_ripples.draw(left, right, dt)
        else:
            vis = vis_ekg.draw(left, right, dt)

        # HUD
        hud = vis.copy()
        cv2.putText(hud, f"Mode: {mode}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(hud, f"L={left:.2f}  R={right:.2f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        y0 = 86
        for i, t in enumerate(help_text):
            cv2.putText(hud, t, (12, y0 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220,220,220), 1, cv2.LINE_AA)

        cv2.imshow("Hand-Controlled Visuals", hud)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
        elif key == ord('4'):
            mode = 4

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
