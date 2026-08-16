import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

# ============================================================
# GRID WORLD SETUP
# ============================================================

# 0 = free path, 1 = wall/obstacle
GRID = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

START = (0, 0)   # Top-left corner
GOAL  = (9, 9)   # Bottom-right corner

ROWS, COLS = GRID.shape

# ============================================================
# A* ALGORITHM
# ============================================================

def heuristic(a, b):
    # Manhattan distance - estimates cost from current cell to goal
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    # Priority queue: (cost, position)
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}           # Tracks the path
    g_score = {start: 0}    # Cost from start to current node
    nodes_explored = 0

    while open_set:
        current_cost, current = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            # Reconstruct path by backtracking
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, nodes_explored

        # Check all 4 neighbors (up, down, left, right)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0] + dr, current[1] + dc)

            # Skip if out of bounds or wall
            if not (0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS):
                continue
            if grid[neighbor[0]][neighbor[1]] == 1:
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None, nodes_explored  # No path found

# ============================================================
# Q-LEARNING ALGORITHM
# ============================================================

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
NUM_ACTIONS = 4

# Q-table: rows x cols x actions
Q_table = np.zeros((ROWS, COLS, NUM_ACTIONS))

# Hyperparameters
LEARNING_RATE  = 0.1    # How fast the robot learns
DISCOUNT       = 0.95   # How much future rewards matter
EPSILON        = 1.0    # Exploration rate (starts high)
EPSILON_DECAY  = 0.995  # Epsilon reduces each episode
MIN_EPSILON    = 0.01   # Minimum exploration rate
EPISODES       = 500    # Number of training rounds
MAX_STEPS      = 200    # Max steps per episode

def get_reward(state):
    # Reward system
    if state == GOAL:
        return 100    # Big reward for reaching goal
    elif GRID[state[0]][state[1]] == 1:
        return -100   # Big penalty for hitting wall
    else:
        return -1     # Small penalty for each step (encourages shortest path)

def q_learning():
    global EPSILON
    episode_rewards = []   # Track total reward per episode
    success_count = 0      # Count successful episodes

    for episode in range(EPISODES):
        state = START
        total_reward = 0

        for step in range(MAX_STEPS):
            # Epsilon-greedy: explore randomly or use Q-table
            if np.random.random() < EPSILON:
                action = np.random.randint(NUM_ACTIONS)  # Random action
            else:
                action = np.argmax(Q_table[state[0], state[1]])  # Best known action

            # Take action
            dr, dc = ACTIONS[action]
            next_state = (state[0] + dr, state[1] + dc)

            # Check if next state is valid
            if not (0 <= next_state[0] < ROWS and 0 <= next_state[1] < COLS):
                next_state = state  # Stay in place if out of bounds

            if GRID[next_state[0]][next_state[1]] == 1:
                next_state = state  # Stay in place if wall

            reward = get_reward(next_state)
            total_reward += reward

            # Q-Learning update formula
            best_next = np.max(Q_table[next_state[0], next_state[1]])
            Q_table[state[0], state[1], action] += LEARNING_RATE * (
                reward + DISCOUNT * best_next - Q_table[state[0], state[1], action]
            )

            state = next_state

            if state == GOAL:
                success_count += 1
                break

        # Decay epsilon after each episode
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        episode_rewards.append(total_reward)

    success_rate = (success_count / EPISODES) * 100
    return episode_rewards, success_rate

def get_q_path():
    # Extract the learned path from Q-table
    state = START
    path = [state]
    visited = set()

    for _ in range(MAX_STEPS):
        if state == GOAL:
            break
        if state in visited:
            break  # Avoid infinite loops
        visited.add(state)
        action = np.argmax(Q_table[state[0], state[1]])
        dr, dc = ACTIONS[action]
        next_state = (state[0] + dr, state[1] + dc)
        if not (0 <= next_state[0] < ROWS and 0 <= next_state[1] < COLS):
            break
        if GRID[next_state[0]][next_state[1]] == 1:
            break
        state = next_state
        path.append(state)

    return path

# ============================================================
# VISUALIZATION
# ============================================================

def visualize(astar_path, q_path, episode_rewards, astar_time, q_time, success_rate):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Robot Path Planning: A* vs Q-Learning', fontsize=16, fontweight='bold')

    # --- Plot 1: A* Path ---
    ax1 = axes[0]
    grid_display = np.copy(GRID).astype(float)
    ax1.imshow(grid_display, cmap='Greys', vmin=0, vmax=1)

    if astar_path:
        for r, c in astar_path:
            if (r, c) != START and (r, c) != GOAL:
                ax1.plot(c, r, 's', color='dodgerblue', markersize=14)

    ax1.plot(START[1], START[0], 's', color='green', markersize=16, label='Start')
    ax1.plot(GOAL[1],  GOAL[0],  's', color='red',   markersize=16, label='Goal')
    ax1.set_title(f'A* Algorithm\nPath Length: {len(astar_path)} steps | Nodes: {astar_nodes}\nTime: {astar_time:.4f}s', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.set_xticks(range(COLS))
    ax1.set_yticks(range(ROWS))
    ax1.grid(True, color='gray', linewidth=0.5)

    # --- Plot 2: Q-Learning Path ---
    ax2 = axes[1]
    ax2.imshow(grid_display, cmap='Greys', vmin=0, vmax=1)

    for r, c in q_path:
        if (r, c) != START and (r, c) != GOAL:
            ax2.plot(c, r, 's', color='orange', markersize=14)

    ax2.plot(START[1], START[0], 's', color='green', markersize=16, label='Start')
    ax2.plot(GOAL[1],  GOAL[0],  's', color='red',   markersize=16, label='Goal')
    ax2.set_title(f'Q-Learning (after {EPISODES} episodes)\nPath Length: {len(q_path)} steps | Success Rate: {success_rate:.1f}%\nTime: {q_time:.4f}s', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.set_xticks(range(COLS))
    ax2.set_yticks(range(ROWS))
    ax2.grid(True, color='gray', linewidth=0.5)

    # --- Plot 3: Learning Curve ---
    ax3 = axes[2]
    smoothed = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
    ax3.plot(episode_rewards, alpha=0.3, color='orange', label='Raw reward')
    ax3.plot(smoothed, color='red', linewidth=2, label='Smoothed (avg 20)')
    ax3.set_title('Q-Learning: Learning Curve\n(Reward improves over episodes)', fontsize=11)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# MAIN - RUN EVERYTHING
# ============================================================

print("=" * 50)
print("   ROBOT PATH PLANNING: A* vs Q-LEARNING")
print("=" * 50)

# Run A*
print("\n[1] Running A* Algorithm...")
start_time = time.time()
astar_path, astar_nodes = astar(GRID, START, GOAL)
astar_time = time.time() - start_time

if astar_path:
    print(f"    Path found! Length: {len(astar_path)} steps")
    print(f"    Nodes explored: {astar_nodes}")
    print(f"    Time: {astar_time:.4f} seconds")
else:
    print("    No path found!")

# Run Q-Learning
print("\n[2] Training Q-Learning Agent (500 episodes)...")
start_time = time.time()
episode_rewards, success_rate = q_learning()
q_time = time.time() - start_time
q_path = get_q_path()

print(f"    Training complete!")
print(f"    Path length: {len(q_path)} steps")
print(f"    Success rate: {success_rate:.1f}%")
print(f"    Training time: {q_time:.4f} seconds")

# Performance Comparison
print("\n" + "=" * 50)
print("   PERFORMANCE COMPARISON")
print("=" * 50)
print(f"{'Method':<15} {'Path Length':<15} {'Success Rate':<15} {'Time'}")
print("-" * 55)
print(f"{'A* (optimal)':<15} {len(astar_path):<15} {'100%':<15} {astar_time:.4f}s")
print(f"{'Q-Learning':<15} {len(q_path):<15} {success_rate:.1f}%{'':<10} {q_time:.4f}s")

if len(astar_path) > 0:
    efficiency = (len(astar_path) / len(q_path)) * 100
    print(f"\n    Q-Learning path efficiency: {efficiency:.1f}% of optimal")

print("\n[3] Generating visualization...")
visualize(astar_path, q_path, episode_rewards, astar_time, q_time, success_rate)
print("    Visualization saved as 'results.png'")
print("\nDone! Check the graphs that appeared on your screen.")