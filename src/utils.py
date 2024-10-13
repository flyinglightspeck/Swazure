import numpy as np


def get_closest_point_line(p1, p2, c):
    d = p2 - p1
    v = c - p1

    # Projection of v onto d
    t = np.dot(v, d) / np.dot(d, d)
    t = max(0, min(1, t))

    return p1 + t * d


def dist_point_line(p1, p2, c):
    p_closest = get_closest_point_line(p1, p2, c)
    return np.linalg.norm(c - p_closest)


def intersects_sphere(p1, p2, c, r):
    return dist_point_line(p1, p2, c) <= r


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    angle_radians = np.arctan2(np.linalg.norm(np.cross(v1, v2)), dot_product)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def diff_spherical_angles(v1, v2):
    _, theta1, phi1 = cartesian_to_spherical(*v1)
    _, theta2, phi2 = cartesian_to_spherical(*v2)
    d_theta = np.abs(theta2 - theta1)
    d_phi = np.abs(phi2 - phi1)
    return d_theta, min(d_phi, 360 - d_phi)


def quadratic_function(x, coeffs):
    a, b, c = coeffs
    return a * x ** 2 + b * x + c


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)  # inclination
    phi = np.arctan2(y, x)  # azimuth
    return r, np.degrees(theta), np.degrees(phi)


def intersection_two_spheres(center1, radius1, center2, radius2):
    # Compute the vector between the centers of the spheres
    d = np.linalg.norm(center1 - center2)

    if d > (radius1 + radius2) or d < abs(radius1 - radius2):
        # raise ValueError("Spheres do not intersect or are completely inside one another.")
        return None

    # Compute the distance from the center of sphere1 to the center of the intersection circle
    a = (radius1 ** 2 - radius2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(radius1 ** 2 - a ** 2)

    # Compute the center of the intersection circle
    P = center1 + a * (center2 - center1) / d

    # Compute the normal vector of the intersection circle
    D = center2 - center1
    D = D / np.linalg.norm(D)  # Normalize
    # Compute the radius of the intersection circle
    radius = h

    return P, radius, D


def intersection_plane_circle(circle_center, circle_radius, circle_normal, plane_point, plane_normal):
    # Convert inputs to numpy arrays
    C = np.array(circle_center)
    r = circle_radius
    n_circle = np.array(circle_normal)
    P = np.array(plane_point)
    n = np.array(plane_normal)

    d = np.cross(n_circle, n)

    # Check if circle plane and target plane are parallel
    if np.allclose(d, 0):
        # raise ValueError("The circle plane and the target plane are parallel and do not intersect.")
        return []

    # Find direction vector of line of intersection
    if np.linalg.norm(d) == 0:
        # raise ValueError("The planes do not intersect in a line.")
        return []

    # Find a point on the line of intersection
    # We solve n · (x - P) = 0 and n_circle · (x - C) = 0
    A = np.vstack([n, n_circle])
    b = np.array([np.dot(n, P), np.dot(n_circle, C)])
    line_point = np.linalg.lstsq(A, b, rcond=None)[0]

    # Compute intersection points
    # The circle is in the plane defined by circle_normal and circle_center
    # Define the parametric equation of the line
    L0 = line_point
    d = d / np.linalg.norm(d)  # Normalize direction vector

    # Calculate intersection points with circle
    a = np.dot(d, d)
    b = 2 * np.dot(d, L0 - C)
    c = np.dot(L0 - C, L0 - C) - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return []  # No intersection points

    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    # Calculate intersection points
    intersection1 = L0 + t1 * d
    intersection2 = L0 + t2 * d

    if np.allclose(intersection1 - intersection2, 0):
        return intersection1

    return [intersection1, intersection2]


def intersection_circle_sphere(circle_center, circle_radius, circle_normal, sphere_center, sphere_radius):
    int_circle = intersection_two_spheres(circle_center, circle_radius, sphere_center, sphere_radius)
    if int_circle is None:
        return []
    else:
        int_center, int_r, int_n = int_circle
        return intersection_plane_circle(int_center, int_r, int_n, circle_center, circle_normal)


if __name__ == "__main__":
    plane_point = np.array((10, 0, 0))
    circle_radius = 4
    normal_vector = np.array((0, 0, 1))
    sphere_center = np.array((0, 0, 0))
    sphere_radius = 5

    print(intersection_circle_sphere(plane_point, circle_radius, normal_vector, sphere_center, sphere_radius))
