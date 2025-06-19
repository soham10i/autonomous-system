from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Main loop
while robot.step(timestep) != -1:
    print("Simulation running... (hook up A* here)")
