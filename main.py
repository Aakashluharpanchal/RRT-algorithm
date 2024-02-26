import numpy as np
import matplotlib.pyplot as plt
import random


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


def generate_random_map(width, height, obstacle_prob):
    # Generate a random map with obstacles
    map = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if random.random() < obstacle_prob:
                map[i, j] = 1
    return map


def is_valid_node(map, node):
    # Check if the node is within the map boundaries and not in an obstacle
    if node.x < 0 or node.x >= map.shape[1] or node.y < 0 or node.y >= map.shape[0]:
        return False
    if map[node.y, node.x] == 1:
        return False
    return True


def euclidean_distance(node1, node2):
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


def nearest_node(nodes, target):
    # Find the nearest node in terms of Euclidean distance
    min_dist = float('inf')
    nearest = None
    for node in nodes:
        dist = euclidean_distance(node, target)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest


def steer(from_node, to_node, max_step):
    # Steer towards the target node with a maximum step size
    if euclidean_distance(from_node, to_node) <= max_step:
        return to_node
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    theta = np.arctan2(dy, dx)
    x = int(from_node.x + max_step * np.cos(theta))
    y = int(from_node.y + max_step * np.sin(theta))
    return Node(x, y)


def rrt(map, start, goal, max_iter, max_step):
    # RRT algorithm to find a path from start to goal in the given map
    nodes = [start]
    for _ in range(max_iter):
        if random.random() < 0.5:
            rand_node = Node(random.randint(0, map.shape[1] - 1), random.randint(0, map.shape[0] - 1))
        else:
            rand_node = goal
        nearest = nearest_node(nodes, rand_node)
        new_node = steer(nearest, rand_node, max_step)
        if is_valid_node(map, new_node):
            new_node.parent = nearest
            nodes.append(new_node)
            if euclidean_distance(new_node, goal) <= max_step:
                goal.parent = new_node
                nodes.append(goal)
                return nodes
    return None


def plot_path(map, nodes, start, goal):
    plt.imshow(map, cmap='Greys', origin='lower')
    plt.plot(start.x, start.y, 'ro', markersize=10)
    plt.plot(goal.x, goal.y, 'go', markersize=10)
    for node in nodes:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'k-', linewidth=2)
    plt.title('RRT Path Planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def main():
    width = 100
    height = 100
    obstacle_prob = 0.3
    max_iter = 1000
    max_step = 10

    # Generate random map

    map = generate_random_map(width, height, obstacle_prob)

    # Define start and goal nodes
    start = Node(10, 10)
    goal = Node(90, 90)

    # Run RRT algorithm
    path = rrt(map, start, goal, max_iter, max_step)

    if path:
        print("Path found!")
        plot_path(map, path, start, goal)
    else:
        print("Path not found.")


if __name__ == "__main__":
    main()
