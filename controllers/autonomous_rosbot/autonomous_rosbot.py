from controller import Supervisor, Keyboard
import numpy as np
import cv2
import math
import time

TIME_STEP = 64
SCALE = 100  # pixels per meter
MAP_SIZE = 700
CENTER = MAP_SIZE // 2

supervisor = Supervisor()
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

motors = [supervisor.getDevice(name) for name in [
    'front left wheel motor', 'front right wheel motor',
    'rear left wheel motor', 'rear right wheel motor']]
for m in motors:
    m.setPosition(float('inf'))

lidar = supervisor.getDevice('lidar')
lidar.enable(TIME_STEP)

trajectory = []
last_lidar_time = time.time()
last_lidar_points = []

def set_speed(left, right):
    motors[0].setVelocity(left)
    motors[2].setVelocity(left)
    motors[1].setVelocity(right)
    motors[3].setVelocity(right)

def get_pose():
    node = supervisor.getSelf()
    pos = node.getPosition()
    ori = node.getOrientation()
    theta = math.atan2(ori[3], ori[0])
    return (pos[0], pos[1], theta)

while supervisor.step(TIME_STEP) != -1:
    key = keyboard.getKey()
    speed = 0.0
    turn = 0.0
    if key == keyboard.UP:
        speed = 3.0
    elif key == keyboard.DOWN:
        speed = -3.0
    elif key == keyboard.LEFT:
        turn = 1.3
    elif key == keyboard.RIGHT:
        turn = -1.3
    set_speed(speed - turn, speed + turn)
    
    x, y, theta = get_pose()
    trajectory.append((x, y))
    
    grid = np.full((MAP_SIZE, MAP_SIZE, 3), 255, np.uint8)
    cv2.line(grid, (CENTER, 0), (CENTER, MAP_SIZE), (210,210,210), 1)
    cv2.line(grid, (0, CENTER), (MAP_SIZE, CENTER), (210,210,210), 1)
    # Draw trajectory
    for i in range(1, len(trajectory)):
        p1 = (int(CENTER + trajectory[i-1][0] * SCALE), int(CENTER - trajectory[i-1][1] * SCALE))
        p2 = (int(CENTER + trajectory[i][0] * SCALE), int(CENTER - trajectory[i][1] * SCALE))
        cv2.line(grid, p1, p2, (0, 200, 0), 2)

    # --- Top-Down Lidar in World Frame every 1s
    if time.time() - last_lidar_time > 1.0:
        # Get absolute lidar, world coordinates only!
        last_lidar_points = []
        lidar_data = lidar.getRangeImage()
        n = len(lidar_data)
        angle_step = 2 * math.pi / n
        for i, d in enumerate(lidar_data):
            if 0.15 < d < 8.0:
                # angle of this beam in robot frame, but we use supervisor's pose (top-view) for world
                wx = x + d * math.cos(theta + i * angle_step)
                wy = y + d * math.sin(theta + i * angle_step)
                px = int(CENTER + wx * SCALE)
                py = int(CENTER - wy * SCALE)
                if 0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE:
                    last_lidar_points.append((px, py))
        last_lidar_time = time.time()
    # Draw world-centric lidar points (no rotation in map!)
    for px, py in last_lidar_points:
        cv2.circle(grid, (px, py), 2, (220,0,0), -1)
    
    px = int(CENTER + x * SCALE)
    py = int(CENTER - y * SCALE)
    cv2.circle(grid, (px, py), 7, (0,0,230), -1)
    arrow_len = 25
    arrow_x = int(px + arrow_len * math.cos(theta))
    arrow_y = int(py - arrow_len * math.sin(theta))
    cv2.arrowedLine(grid, (px, py), (arrow_x, arrow_y), (0,0,230), 2, tipLength=0.4)
    info = f"({x:.2f},{y:.2f},{math.degrees(theta):.0f}°)"
    cv2.putText(grid, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (33,33,33), 2)
    cv2.putText(grid, 'ARROWS=move; ESC=exit', (10, MAP_SIZE-18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,120,120), 1)
    cv2.imshow('Supervisor/Lidar Top-Down (World Frame)', grid)
    if cv2.waitKey(2) & 0xFF == 27:
        break
cv2.destroyAllWindows()
