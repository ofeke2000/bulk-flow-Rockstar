import random


def pick_dot(box_size, cube_length):
    """
    Picks a random point within a box of size 'box_size' such that
    the point is within (cube_length, box_size - cube_length) on each axis.

    Parameters:
        box_size (float): The size of the box.
        cube_length (float): The length that determines the boundary offset.

    Returns:
        tuple: A random point (x, y, z) within the constrained range.
    """
    if cube_length >= box_size / 2:
        raise ValueError("cube_length must be smaller than half of the box_size.")

    x = random.uniform(cube_length, box_size - cube_length)
    y = random.uniform(cube_length, box_size - cube_length)
    z = random.uniform(cube_length, box_size - cube_length)

    return (x, y, z)


# Example usage:
# box_size = 10
# cube_length = 2
# random_point = pick_dot(box_size, cube_length)
# print(f"Random point in a box of size {box_size} with boundary offset {cube_length}: {random_point}")
