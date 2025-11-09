import os
import time
import random
import keyboard

# --- Game Configuration ---
FIELD_WIDTH = 16 
FIELD_HEIGHT = 20

# Tetromino shapes and their colors
SHAPES = {
    'T': ([[0, 1, 0], [1, 1, 1]], 1),
    'S': ([[0, 2, 2], [2, 2, 0]], 2),
    'Z': ([[3, 3, 0], [0, 3, 3]], 3),
    'J': ([[4, 0, 0], [4, 4, 4]], 4),
    'L': ([[0, 0, 5], [5, 5, 5]], 5),
    'I': ([[6, 6, 6, 6]], 6),
    'O': ([[7, 7], [7, 7]], 7)
}

# ANSI color codes for the blocks
COLORS = [
    "\033[0m",        # 0 - Reset
    "\033[95m",       # 1 - T (Magenta)
    "\033[92m",       # 2 - S (Green)
    "\033[91m",       # 3 - Z (Red)
    "\033[94m",       # 4 - J (Blue)
    "\033[38;5;208m", # 5 - L (Orange)
    "\033[96m",       # 6 - I (Cyan)
    "\033[93m"        # 7 - O (Yellow)
]

# --- Game State Variables ---
field = [[0 for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]
score = 0
game_over = False
current_piece = None
next_piece = None 
piece_x, piece_y = 0, 0
fall_speed = 0.5

# --- Helper Functions ---

# --- REMOVED the clear_screen() function as it's no longer needed ---

def draw_screen(field, piece, px, py, next_p):
    """Draws the entire game screen without flickering by repositioning the cursor."""
    screen_buffer = [list(row) for row in field]

    # Overlay the current piece onto the buffer
    if piece:
        shape, color = SHAPES[piece]
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell and (0 <= py + y < FIELD_HEIGHT and 0 <= px + x < FIELD_WIDTH):
                    screen_buffer[py + y][px + x] = color
    
    # --- FLICKER FIX V2: Build the screen string starting with a cursor reset code ---
    output_buffer = "\033[H" # This ANSI escape code moves the cursor to the top-left (home) position.
    output_buffer += "--- Python CMD Tetris (Flicker-Free) ---\n"
    output_buffer += f"Score: {score}\n"

    # Top border
    top_border = "╔" + "═" * (FIELD_WIDTH * 2) + "╗"
    next_box_title = " NEXT PIECE "
    output_buffer += f"{top_border}  {next_box_title}\n"

    # Prepare the 'Next Piece' shape for drawing
    next_shape, _ = SHAPES[next_p] if next_p else ([], 0)
    
    for y in range(FIELD_HEIGHT):
        # Draw game field row
        line = "║"
        for x in range(FIELD_WIDTH):
            cell_color_index = screen_buffer[y][x]
            if cell_color_index > 0:
                line += f"{COLORS[cell_color_index]}▓▓{COLORS[0]}"
            else:
                line += "  "
        line += "║"

        # Draw the 'Next Piece' box content next to the game field
        if y == 0:
            line += " ╔══════════╗"
        elif y < len(next_shape) + 2 and y > 1:
            line += " ║ "
            next_row = next_shape[y-2]
            for cell in next_row:
                if cell > 0:
                    line += f"{COLORS[cell]}▓▓{COLORS[0]}"
                else:
                    line += "  "
            padding = 10 - (len(next_row) * 2)
            line += " " * padding + "║"
        elif y == 5:
             line += " ╚══════════╝"
        
        output_buffer += line + "\n"

    # Bottom border
    bottom_border = "╚" + "═" * (FIELD_WIDTH * 2) + "╝"
    output_buffer += bottom_border + "\n"
    output_buffer += "Controls: A(Left), D(Right), S(Down), W(Rotate)\n"

    # --- FLICKER FIX V2: Print the entire buffer at once. The cursor is already reset. ---
    print(output_buffer, end="")


def new_piece():
    """Generates a new random Tetromino and manages the next piece."""
    global current_piece, next_piece, piece_x, piece_y
    
    if next_piece is None:
        current_piece = random.choice(list(SHAPES.keys()))
        next_piece = random.choice(list(SHAPES.keys()))
    else:
        current_piece = next_piece
        next_piece = random.choice(list(SHAPES.keys()))

    piece_shape = SHAPES[current_piece][0]
    piece_x = FIELD_WIDTH // 2 - len(piece_shape[0]) // 2
    piece_y = 0
    
    if check_collision(piece_shape, piece_x, piece_y):
        return True # Game Over
    return False # Continue

def check_collision(shape, px, py):
    """Checks if the piece at the given position collides with the field."""
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                field_x, field_y = px + x, py + y
                if not (0 <= field_x < FIELD_WIDTH and 0 <= field_y < FIELD_HEIGHT and field[field_y][field_x] == 0):
                    return True
    return False

def lock_piece(shape, px, py, color):
    """Locks the current piece onto the field."""
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                if 0 <= py + y < FIELD_HEIGHT and 0 <= px + x < FIELD_WIDTH:
                    field[py + y][px + x] = color

def clear_lines():
    """Checks for and clears completed lines, updating the score."""
    global score, fall_speed
    lines_to_clear = []
    for i, row in enumerate(field):
        if all(cell > 0 for cell in row):
            lines_to_clear.append(i)

    if lines_to_clear:
        for line_index in sorted(lines_to_clear, reverse=True):
            del field[line_index]
            field.insert(0, [0 for _ in range(FIELD_WIDTH)])
        
        score += len(lines_to_clear) * 100 * len(lines_to_clear) 
        fall_speed = max(0.1, fall_speed * 0.9)


def rotate_piece(shape):
    """Rotates a piece (matrix) 90 degrees clockwise."""
    return [list(reversed(col)) for col in zip(*shape)]

# --- Main Game Loop ---

def main():
    global game_over, piece_x, piece_y, fall_speed

    last_fall_time = time.time()
    game_over = new_piece()

    while not game_over:
        # --- Handle User Input ---
        move_x = 0
        
        if keyboard.is_pressed('a') or keyboard.is_pressed('left'):
            move_x = -1
            time.sleep(0.1) 
        elif keyboard.is_pressed('d') or keyboard.is_pressed('right'):
            move_x = 1
            time.sleep(0.1) 
        elif keyboard.is_pressed('w') or keyboard.is_pressed('up'):
            rotated_shape = rotate_piece(SHAPES[current_piece][0])
            if not check_collision(rotated_shape, piece_x, piece_y):
                SHAPES[current_piece] = (rotated_shape, SHAPES[current_piece][1])
            time.sleep(0.15)
        
        if keyboard.is_pressed('s') or keyboard.is_pressed('down'):
             current_fall_speed = 0.05
        else:
             current_fall_speed = fall_speed

        # Move horizontally if no collision
        if not check_collision(SHAPES[current_piece][0], piece_x + move_x, piece_y):
            piece_x += move_x

        # --- Game Logic (Gravity) ---
        if time.time() - last_fall_time > current_fall_speed:
            shape, color = SHAPES[current_piece]
            if not check_collision(shape, piece_x, piece_y + 1):
                piece_y += 1
            else:
                lock_piece(shape, piece_x, piece_y, color)
                clear_lines()
                if new_piece():
                    game_over = True
            last_fall_time = time.time()

        # --- Drawing ---
        draw_screen(field, current_piece, piece_x, piece_y, next_piece)
        time.sleep(0.01) 

    print("\n--- GAME OVER ---\n")
    print(f"Final Score: {score}")

if __name__ == "__main__":
    main()
