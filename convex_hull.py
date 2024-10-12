import math
import sys
from typing import List, Tuple

EPSILON = sys.float_info.epsilon
Point = Tuple[int, int]


def y_intercept(p1: Point, p2: Point, x: int) -> float:
    """
    Given two points, p1 and p2, an x coordinate from a vertical line,
    compute and return the y-intercept of the line segment p1->p2
    with the vertical line passing through x.
    """
    x1, y1 = p1
    x2, y2 = p2    
    slope = (y2 - y1) / (x2 - x1)
    return y1 + (x - x1) * slope


def triangle_area(a: Point, b: Point, c: Point) -> float:
    """
    Given three points a,b,c, computes and returns the area defined by the triangle a,b,c.
    Note that this area will be negative if a,b,c represents a clockwise sequence,
    positive if it is counter-clockwise, and zero if the points are collinear.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    return ((cx - bx) * (by - ay) - (bx - ax) * (cy - by)) / 2


def is_clockwise(a: Point, b: Point, c: Point) -> bool:
    """ Returns True if a, b, c form a clockwise sequence. """
    return triangle_area(a, b, c) < -EPSILON


def is_counter_clockwise(a: Point, b: Point, c: Point) -> bool:
    """ Returns True if a, b, c form a counter-clockwise sequence. """
    return triangle_area(a, b, c) > EPSILON


def collinear(a: Point, b: Point, c: Point) -> bool:
    """ Returns True if a, b, c are collinear. """
    return abs(triangle_area(a, b, c)) <= EPSILON


def clockwise_sort(points: List[Point]):
    """
    Sort points by ascending clockwise angle from +x about the centroid.
    """
    if len(points) < 2:
        return

    centroid_x = sum(p[0] for p in points) / len(points)
    centroid_y = sum(p[1] for p in points) / len(points)

    def sort_key(point: Point):
        angle = math.atan2(point[1] - centroid_y, point[0] - centroid_x)
        return (angle + math.tau) % math.tau, point[0], point[1]

    points.sort(key=sort_key)

def split_in_two(points: List[Point]):
    """ 
    Split the list of points into two halves based on their x-values. 
    Points with x-values less than or equal to the middle are placed in the left list, 
    and points with larger x-values go to the right list.
    """
    left_points = []
    right_points = []
    x_values = [p[0] for p in points]  # Get all x-values from the points
    middle_x = sum(x_values) / len(points)  # Find the middle x-value

    for p in points:
        if p[0] <= middle_x:  # If the point is on the left side
            left_points.append(p)
        else:  # If the point is on the right side
            right_points.append(p)

    return left_points, right_points  # Return the two lists


def findHullsRightMostPoint(hull: List[Point]) -> Point:
    """ 
    Find and return the point with the largest x-value in the hull (the rightmost point). 
    """
    return max(hull, key=lambda p: p[0])  # Return the point with the highest x


def findHullsLeftMostPoint(hull: List[Point]) -> Point:
    """ 
    Find and return the point with the smallest x-value in the hull (the leftmost point). 
    """
    return min(hull, key=lambda p: p[0])  # Return the point with the lowest x


def find_top_connector_line_segment(left_hull: List[Point], right_hull: List[Point], left_hull_rightmost_point: Point, right_hull_leftmost_point: Point, midpoint_line: float) -> Tuple[Point, Point]:
    """ 
    Find the top connecting line between the left and right hulls. 
    This finds the highest line segment that connects the two hulls.
    """
    best_case_right_hull_point = right_hull_leftmost_point
    best_case_left_hull_point = left_hull_rightmost_point
    best_case_y_intercept = y_intercept(best_case_left_hull_point, best_case_right_hull_point, midpoint_line)

    right_length = len(right_hull)
    left_length = len(left_hull)

    # Keep searching for the best line by checking if we can get a better y-intercept
    while (y_intercept(right_hull[(right_hull.index(best_case_right_hull_point) + 1) % right_length], best_case_left_hull_point, midpoint_line) < best_case_y_intercept or
           y_intercept(left_hull[(left_hull.index(best_case_left_hull_point) - 1) % left_length], best_case_right_hull_point, midpoint_line) < best_case_y_intercept):

        if y_intercept(best_case_left_hull_point, right_hull[(right_hull.index(best_case_right_hull_point) + 1) % right_length], midpoint_line) > best_case_y_intercept:
            best_case_left_hull_point = left_hull[(left_hull.index(best_case_left_hull_point) - 1) % left_length]
        else:
            best_case_right_hull_point = right_hull[(right_hull.index(best_case_right_hull_point) + 1) % right_length]

        best_case_y_intercept = y_intercept(best_case_left_hull_point, best_case_right_hull_point, midpoint_line)

    return best_case_left_hull_point, best_case_right_hull_point  # Return the two points that form the top line


def find_bottom_connector_line_segment(left_hull: List[Point], right_hull: List[Point], left_hull_rightmost_point: Point, right_hull_leftmost_point: Point, midpoint_line: float):
    """ 
    Find the bottom connecting line between the left and right hulls. 
    This finds the lowest line segment that connects the two hulls.
    """
    best_case_right_hull_point = right_hull_leftmost_point
    best_case_left_hull_point = left_hull_rightmost_point
    best_case_y_intercept = y_intercept(best_case_left_hull_point, best_case_right_hull_point, midpoint_line)

    right_length = len(right_hull)
    left_length = len(left_hull)

    # Keep searching for the best line by checking if we can get a better y-intercept
    while (y_intercept(right_hull[(right_hull.index(best_case_right_hull_point) + 1) % right_length], best_case_left_hull_point, midpoint_line) > best_case_y_intercept or
           y_intercept(left_hull[(left_hull.index(best_case_left_hull_point) - 1) % left_length], best_case_right_hull_point, midpoint_line) > best_case_y_intercept):

        if y_intercept(best_case_left_hull_point, right_hull[(right_hull.index(best_case_right_hull_point) + 1) % right_length], midpoint_line) < best_case_y_intercept:
            best_case_left_hull_point = left_hull[(left_hull.index(best_case_left_hull_point) - 1) % left_length]
        else:
            best_case_right_hull_point = right_hull[(right_hull.index(best_case_right_hull_point) + 1) % right_length]

        best_case_y_intercept = y_intercept(best_case_left_hull_point, best_case_right_hull_point, midpoint_line)

    return best_case_left_hull_point, best_case_right_hull_point  # Return the two points that form the bottom line


def combine(left_hull: List[Point], right_hull: List[Point]) -> List[Point]:
    """ 
    Combine the left and right hulls into one convex hull. 
    First, find the top and bottom connecting lines and then merge the points.
    """
    # Sort both hulls in clockwise order
    clockwise_sort(left_hull)
    clockwise_sort(right_hull)

    # Find the rightmost point of the left hull and the leftmost point of the right hull
    left_hull_rightmost_point = findHullsRightMostPoint(left_hull)
    right_hull_leftmost_point = findHullsLeftMostPoint(right_hull)

    # Find the middle x value between the two hulls
    midpoint_line = (left_hull_rightmost_point[0] + right_hull_leftmost_point[0]) / 2

    # Find the top and bottom connecting points between the two hulls
    top_left_point, top_right_point = find_top_connector_line_segment(left_hull, right_hull, left_hull_rightmost_point, right_hull_leftmost_point, midpoint_line)
    bottom_left_point, bottom_right_point = find_bottom_connector_line_segment(left_hull, right_hull, left_hull_rightmost_point, right_hull_leftmost_point, midpoint_line)

    combined_hull = []  # This will hold the final combined hull

    # Add points from the left hull starting from the bottom left to the top left
    index_left = left_hull.index(bottom_left_point)
    while True:
        combined_hull.append(left_hull[index_left])
        if left_hull[index_left] == top_left_point:
            break
        index_left = (index_left + 1) % len(left_hull)

    # Add points from the right hull starting from the bottom right to the top right
    index_right = right_hull.index(bottom_right_point)
    while True:
        combined_hull.append(right_hull[index_right])
        if right_hull[index_right] == top_right_point:
            break
        index_right = (index_right - 1) % len(right_hull)

    # Remove duplicate points and sort the combined points
    combined_hull = list(set(combined_hull))
    combined_hull.sort()

    # Create the lower hull
    lower_hull = []
    for p in combined_hull:
        while len(lower_hull) >= 2 and triangle_area(lower_hull[-2], lower_hull[-1], p) <= 0:
            lower_hull.pop()
        lower_hull.append(p)

    # Create the upper hull
    upper_hull = []
    for p in reversed(combined_hull):
        while len(upper_hull) >= 2 and triangle_area(upper_hull[-2], upper_hull[-1], p) <= 0:
            upper_hull.pop()
        upper_hull.append(p)

    return lower_hull[:-1] + upper_hull[:-1]  # Return the full convex hull


def orientation(p: Point, q: Point, r: Point) -> int:
    """ 
    Find out the orientation of three points (p, q, r). 
    Returns 0 if the points are collinear, 1 if clockwise, 2 if counterclockwise.
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or counterclockwise


def graham_scan(points: List[Point]) -> List[Point]:
    """ 
    Compute the convex hull using the Graham scan algorithm. 
    """
    point_len = len(points)

    # Find the point with the lowest y-value (or lowest x if y is the same)
    min_idx = 0
    for i in range(1, point_len):
        if points[i][1] < points[min_idx][1] or (points[i][1] == points[min_idx][1] and points[i][0] < points[min_idx][0]):
            min_idx = i

    # Move the lowest point to the first position
    points[0], points[min_idx] = points[min_idx], points[0]

    # Sort the points based on their polar angle with respect to the lowest point
    pivot = points[0]
    points[1:] = sorted(points[1:], key=lambda x: (180 + (180 / 3.1415926535) * math.atan2(x[1] - pivot[1], x[0] - pivot[0])) % 360)

    # Start building the hull with the first three points
    stack = [points[0], points[1], points[2]]

    # Process the remaining points
    for i in range(3, point_len):
        while orientation(stack[-2], stack[-1], points[i]) != 2:  # Keep removing points if they are not counterclockwise
            stack.pop()
        stack.append(points[i])  # Add the current point to the hull

    return stack  # Return the points that form the convex hull


def base_case_hull(points: List[Point]) -> List[Point]:
    """ 
    If we have a small number of points (3 or less), return them directly as the convex hull.
    Otherwise, use the Graham scan to compute the hull.
    """
    if len(points) <= 3:
        return points
    return graham_scan(points)


def compute_hull(points: List[Point]) -> List[Point]:
    """ 
    Compute the convex hull using a divide-and-conquer approach. 
    For small sets of points, use the base case directly.
    """
    if len(points) <= 5:
        return base_case_hull(points)

    # Split points into left and right groups
    left_points, right_points = split_in_two(points)

    # Compute the hulls for the left and right groups
    left_hull = compute_hull(left_points)
    right_hull = compute_hull(right_points)

    # Combine the left and right hulls into one
    return combine(left_hull, right_hull)