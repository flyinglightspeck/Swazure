import logging
import os.path
from copy import deepcopy
from functools import cache

import networkx as nx
import numpy as np
from scipy.spatial import distance

from const import EdgeType
from utils import quadratic_function, intersects_sphere, dist_point_line, get_closest_point_line, intersection_circle_sphere


class Orchestrator:
    def __init__(self, args, alg):
        self.args = args
        self.points = None
        self.dist_tracking = None
        self.dist_center = None
        self.sorted_neighbor_ids = None
        self.sorted_neighbor_dists = None
        self.blind_neighbors = None
        self.all_sweet_neighbors = None
        self.visible_sweet_neighbors = None
        self.all_decaying_neighbors = None
        self.visible_decaying_neighbors = None
        self.all_sweet_neighbors_graph = None
        self.visible_sweet_neighbors_graph = None
        self.all_decaying_neighbors_graph = None
        self.visible_decaying_neighbors_graph = None
        self.all_sweet_and_decaying_neighbors_graph = None
        self.visible_sweet_and_decaying_neighbors_graph = None
        self.error_coefficients = None
        self.colliding_points = []
        self.radius = 0
        self.min_dist = 0
        self.avg_dist = 0
        self.max_dist = 0
        self.num_kissing_neighbors = None
        self.swarm_locator_df = None
        self.props_df = None
        self.alg = alg(orc=self)
        self.weight = 'weight' if args.weighted else None

    def initialize(self):
        self.read_point_cloud()
        self.add_dead_reckoning_error()
        self.compute_c2c_distance()
        self.compute_radius()
        self.compute_kissing_neighbors()
        self.remove_colliding_points()
        self.compute_dist_tracking()
        self.sort_neighbors()
        self.compute_blind_neighbors()
        self.compute_sweet_neighbors()
        self.compute_decaying_neighbors()
        self.combine_sweet_and_decay_neighbors()
        self.initialize_camera_error_model()
        self.create_results_dir()
        # exit()

    def start(self):
        self.alg.run()
        self.alg.store()

    def read_point_cloud(self):
        self.points = np.loadtxt(f'assets/{self.args.shape}.xyz', delimiter=' ') * self.args.scale

    def compute_c2c_distance(self):
        self.dist_center = distance.squareform(distance.pdist(self.points))
        np.fill_diagonal(self.dist_center, np.inf)

    def compute_radius(self):
        closest_neighbor_dist = np.min(self.dist_center, axis=1)
        self.min_dist = closest_neighbor_dist.min()
        self.avg_dist = closest_neighbor_dist.mean()
        self.max_dist = closest_neighbor_dist.max()
        if self.args.radius:
            self.radius = self.args.radius
        else:
            self.radius = self.args.radius_beta * self.min_dist

    def compute_kissing_neighbors(self):
        self.num_kissing_neighbors = ((self.radius * 2 - 1e-3 <= self.dist_center) & (
                self.dist_center <= self.radius * 2 + 1e-3)).sum() / 2

    def remove_colliding_points(self):
        if np.any(self.dist_center < self.radius * 2):
            logging.info("Removing colliding points")

            to_remove = set()

            # Iterate over the upper triangle of the distance matrix
            num_points = len(self.points)
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    if self.dist_center[i][j] < self.radius * 2:
                        # Add one of the points to the removal set randomly
                        if i not in to_remove and j not in to_remove:
                            to_remove.add(i if np.random.rand() > 0.5 else j)

            # Remove the points in reverse order to avoid index shift issues
            self.points = [p for idx, p in enumerate(self.points) if idx not in to_remove]

            self.colliding_points = np.array(list(to_remove))
            self.compute_c2c_distance()

    def compute_dist_tracking(self):
        self.dist_tracking = self.dist_center - self.radius * 2
        assert np.all(self.dist_tracking >= 0)

    def sort_neighbors(self):
        self.sorted_neighbor_ids = np.argsort(self.dist_tracking, axis=1)
        self.sorted_neighbor_dists = np.take_along_axis(self.dist_tracking, self.sorted_neighbor_ids, axis=1)

    def compute_blind_neighbors(self):
        self.blind_neighbors = [self.sorted_neighbor_ids[i][
                                    (self.sorted_neighbor_dists[i] > 0) &
                                    (self.sorted_neighbor_dists[i] < self.args.sweet_range_min)
                                    ] for i in
                                range(len(self.sorted_neighbor_ids))]

    def compute_sweet_neighbors(self):
        self.all_sweet_neighbors = self.compute_neighbors_in_range(self.args.sweet_range_min, self.args.sweet_range_max)
        self.visible_sweet_neighbors = self.compute_visible_neighbors(self.all_sweet_neighbors)
        self.all_sweet_neighbors_graph = self.create_neighbors_graph(self.all_sweet_neighbors, EdgeType.SWEET)
        self.visible_sweet_neighbors_graph = self.create_neighbors_graph(self.visible_sweet_neighbors, EdgeType.SWEET)

    def compute_decaying_neighbors(self):
        self.all_decaying_neighbors = self.compute_neighbors_in_range(self.args.decaying_range_min,
                                                                      self.args.decaying_range_max)
        self.visible_decaying_neighbors = self.compute_visible_neighbors(self.all_decaying_neighbors)
        self.all_decaying_neighbors_graph = self.create_neighbors_graph(self.all_decaying_neighbors, EdgeType.DECAYING)
        self.visible_decaying_neighbors_graph = self.create_neighbors_graph(self.visible_decaying_neighbors,
                                                                            EdgeType.DECAYING)

    def combine_sweet_and_decay_neighbors(self):
        self.all_sweet_and_decaying_neighbors_graph = nx.compose(self.all_sweet_neighbors_graph,
                                                                 self.all_decaying_neighbors_graph)
        self.visible_sweet_and_decaying_neighbors_graph = nx.compose(self.visible_sweet_neighbors_graph,
                                                                     self.visible_decaying_neighbors_graph)
        # print(self.visible_sweet_and_decaying_neighbors_graph)

    def compute_neighbors_in_range(self, range_min, range_max):
        neighbors = [self.sorted_neighbor_ids[i][
                         (self.sorted_neighbor_dists[i] >= range_min) &
                         (self.sorted_neighbor_dists[i] < range_max)
                         ] for i in
                     range(len(self.sorted_neighbor_ids))]

        return neighbors

    def compute_visible_neighbors(self, neighbors):
        visible_neighbors = deepcopy(neighbors)
        for i in range(len(visible_neighbors)):
            m = np.array([not self.is_occluded(i, j) for j in visible_neighbors[i]], dtype=bool)
            visible_neighbors[i] = visible_neighbors[i][m]

        return visible_neighbors

    def get_edge_types(self, path):
        edge_types = []
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edge_type = self.all_sweet_and_decaying_neighbors_graph[u][v]['edge_type']
            edge_types.append(edge_type.value)
        return edge_types

    def get_path_weight(self, path):
        if len(path) == 0:
            return None

        path_weight = 0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            w = self.all_sweet_and_decaying_neighbors_graph[u][v]['weight']
            path_weight += w
        return path_weight

    @staticmethod
    def get_path_weight_by_coord(path_coord):
        if len(path_coord) == 0:
            return None

        path_weight = 0
        for i in range(len(path_coord) - 1):
            u = path_coord[i]
            v = path_coord[i + 1]
            w = np.linalg.norm(u - v)
            path_weight += w
        return path_weight

    def get_edge_types_by_coord(self, path_coord):
        edge_types = []
        for i in range(len(path_coord) - 1):
            u = path_coord[i]
            v = path_coord[i + 1]
            edge_type = self.get_edge_type_by_points(u, v)
            edge_types.append(edge_type.value)
        return edge_types

    def get_edge_type_by_points(self, u, v):
        tracking_dist = np.linalg.norm(u - v) - self.radius * 2

        if tracking_dist < self.args.sweet_range_min:
            return EdgeType.BLIND
        elif tracking_dist <= self.args.sweet_range_max:
            return EdgeType.SWEET
        else:
            return EdgeType.DECAYING

    @staticmethod
    def get_path_hops(edge_types):
        num_sweet_hops = edge_types.count(EdgeType.SWEET.value)
        num_decaying_hops = edge_types.count(EdgeType.DECAYING.value)
        num_blind_hops = edge_types.count(EdgeType.BLIND.value)

        if len(edge_types):
            return len(edge_types), num_sweet_hops, num_decaying_hops, num_blind_hops
        else:
            return None, None, None, None

    def get_shortest_path(self, source, target):
        try:
            path = nx.shortest_path(self.all_sweet_neighbors_graph, source=source, target=target,
                                    weight=self.weight)
            return path
        except nx.NetworkXNoPath:
            return []

    def get_sweet_shortest_path(self, source, target):
        try:
            path = nx.shortest_path(self.visible_sweet_neighbors_graph, source=source, target=target,
                                    weight=self.weight)
            return path
        except nx.NetworkXNoPath:
            return []

    def get_sweet_and_decaying_shortest_path(self, source, target):
        try:
            path = nx.shortest_path(self.visible_sweet_and_decaying_neighbors_graph, source=source, target=target,
                                    weight=self.weight)
            return path
        except nx.NetworkXNoPath:
            return []

    def get_sweet_shortest_path_generator(self, source, target):
        return nx.shortest_simple_paths(self.visible_sweet_neighbors_graph, source=source, target=target,
                                        weight=self.weight)

    def get_sweet_and_decaying_shortest_path_generator(self, source, target):
        return nx.shortest_simple_paths(self.visible_sweet_and_decaying_neighbors_graph, source=source, target=target,
                                        weight=self.weight)

    def compute_relative_pose_by_path(self, path):
        if path is None or len(path) == 0:
            return np.full(3, np.nan)

        pose = np.zeros(3)

        for i in range(len(path) - 1):
            p = self.points[path[i + 1]] - self.points[path[i]]
            err_p = self.add_camera_error(p)
            pose += err_p

        return pose

    def compute_relative_pose_by_path_coord(self, path_coord):
        if path_coord is None or len(path_coord) == 0:
            return np.full(3, np.nan)

        pose = np.zeros(3)

        for i in range(len(path_coord) - 1):
            p = path_coord[i + 1] - path_coord[i]
            err_p = self.add_camera_error(p)
            pose += err_p

        return pose

    def initialize_camera_error_model(self):
        x = np.array([200, 150, 100, 75, 50, 45, 42.5, 40]) / 10  # cm
        y = np.array([30.46, 11.68, 5.82, 0.893333, 2.14, 5.4, 4.16471, 6.05]) / 100  # percent

        self.error_coefficients = np.polyfit(x, y, 2) - np.array([0, 0, .015])
        # print(self.error_coefficients)
        # print(quadratic_function(6, self.error_coefficients)*100)
        # print(quadratic_function(7.5, self.error_coefficients)*100)
        # print(quadratic_function(8, self.error_coefficients)*100)

    def add_camera_error(self, pose):
        if self.args.accurate:
            return pose

        c2c_dist = np.linalg.norm(pose)
        tracking_dist = c2c_dist - 2 * self.radius
        percentage_err = quadratic_function(tracking_dist, self.error_coefficients)
        erroneous_dist = tracking_dist * (1 + percentage_err) + 2 * self.radius
        return pose * erroneous_dist / c2c_dist

    def add_dead_reckoning_error(self):
        if self.args.dead_reckoning_angle == 0:
            return

        points = []
        for v in self.points:
            points.append(self.add_dead_reckoning_error_to_vector(v))
        self.points = np.vstack(points)

    def add_dead_reckoning_error_to_vector(self, vector):
        alpha = self.args.dead_reckoning_angle / 180 * np.pi
        if vector[0] or vector[1]:
            i = np.array([vector[1], -vector[0], 0])
        elif vector[2]:
            i = np.array([vector[2], 0, -vector[0]])
        else:
            return vector

        if alpha == 0:
            return vector

        j = np.cross(vector, i)
        norm_i = np.linalg.norm(i)
        norm_j = np.linalg.norm(j)
        norm_v = np.linalg.norm(vector)
        i = i / norm_i
        j = j / norm_j
        phi = np.random.uniform(0, 2 * np.pi)
        error = np.sin(phi) * i + np.cos(phi) * j
        r = np.linalg.norm(vector) * np.tan(alpha)

        erred_v = vector + np.random.uniform(0, r) * error
        return norm_v * erred_v / np.linalg.norm(erred_v)

    def collide_by_proximity(self, id1, new_coord1, d):
        proximity = self.sorted_neighbor_ids[id1][self.sorted_neighbor_dists[id1] <= d]
        for j in proximity:
            if np.linalg.norm(new_coord1 - self.points[j]) < 2 * self.radius:
                return True
        return False

    def get_all_collisions(self, id1, new_coord1):
        p = self.points[id1]
        proximity = self.sorted_neighbor_ids[id1][self.sorted_neighbor_dists[id1] <= np.linalg.norm(new_coord1 - p)]
        colliding_points = []
        for j in proximity:
            if np.linalg.norm(new_coord1 - self.points[j]) < 2 * self.radius:
                colliding_points.append(j)
        return colliding_points

    def is_occluded(self, id1, id2):
        if id1 < id2:
            return self._cached_is_occluded(id1, id2)
        return self._cached_is_occluded(id2, id1)

    def is_new_coord_occluded(self, id1, id2, new_coord1):
        if id1 == id2:
            return True

        p1 = new_coord1
        p2 = self.points[id2]

        d = np.linalg.norm(p1 - p2)

        for i in self.sorted_neighbor_ids[id1][self.sorted_neighbor_dists[id1] <= d]:
            if i != id1 and i != id2:
                if intersects_sphere(p1, p2, self.points[i], self.radius):
                    return True

        return False

    @cache
    def _cached_is_occluded(self, id1, id2):
        if id1 == id2:
            return True

        p1 = self.points[id1]
        p2 = self.points[id2]

        d = self.dist_center[id1][id2]

        for i in self.sorted_neighbor_ids[id1][self.sorted_neighbor_dists[id1] <= d]:
            if i != id1 and i != id2:
                if intersects_sphere(p1, p2, self.points[i], self.radius):
                    return True

        return False

    def get_correction_dist_all_occlusions(self, id1, id2, check_collision=False):
        p1 = self.points[id1]
        p2 = self.points[id2]

        d = self.dist_center[id1][id2]

        correction_dists = []
        occlusion_remains = False
        for i in self.sorted_neighbor_ids[id1][self.sorted_neighbor_dists[id1] <= d]:
            if i != id1 and i != id2:
                dist = dist_point_line(p1, p2, self.points[i])
                if dist < self.radius:
                    if check_collision:
                        min_dist = self.get_collision_free_correction_dist(id1, id2, i)
                        correction_dists.append(min_dist)
                        if min_dist == 0:
                            occlusion_remains = True
                    else:
                        correction_dists.append(self.radius - dist)

        return correction_dists, occlusion_remains

    def get_collision_free_correction_dist(self, s, t, o):
        """
        :param s: source
        :param t: target
        :param o: obstructing
        :return: minimum distance to move obstructing
        """
        p1 = self.points[s]
        p2 = self.points[t]
        o1 = self.points[o]

        dest_circle_c = get_closest_point_line(p1, p2, o1)
        dest_circle_n = (p2 - p1) / np.linalg.norm(p2 - p1)
        dest_circle_r = self.radius
        optimal_dest = o1 + (self.radius / np.linalg.norm(o1 - dest_circle_c) - 1) * (o1 - dest_circle_c)
        max_dist = 2 * self.radius

        # check if the optimal destination causes a collision
        if self.collide_by_proximity(o, optimal_dest, np.linalg.norm(o - optimal_dest)):
            # cant go to optimal
            # check potential collisions for destinations other than optimal
            proximity = self.sorted_neighbor_ids[o][self.sorted_neighbor_dists[o] <= max_dist]

            intersections = []
            candidates = []
            for k in proximity:
                intersections = intersection_circle_sphere(dest_circle_c, dest_circle_r, dest_circle_n, self.points[k],
                                                           2 * self.radius)
                intersections.extend(intersections)

            for intersection in intersections:
                for collision in proximity:
                    if np.linalg.norm(intersection - self.points[collision]) >= self.radius * 2:
                        candidates.append(intersection)

            if len(candidates) == 0:
                # print("No destination found")
                return 0
            alt_dists = np.linalg.norm(np.array(candidates) - o1, axis=1)
            alt_dest_idx = np.argmin(alt_dists)
            # alt_collisions = self.get_all_collisions(o, candidates[alt_dest_idx])
            # if len(alt_collisions) > 0:
            #     print("colliding with another")
            return alt_dists[alt_dest_idx]
        else:
            # return optimal distance
            return self.radius - np.linalg.norm(o1 - dest_circle_c)

    def get_tracking_distance(self, id1, id2):
        return self.dist_tracking[id1][id2]

    def create_neighbors_graph(self, neighbors, edge_type):
        G = nx.Graph()
        G.add_nodes_from(range(len(neighbors)))

        for i in range(len(neighbors)):
            for j in neighbors[i]:
                G.add_edge(i, j, weight=self.get_tracking_distance(i, j), edge_type=edge_type)
        return G

    @staticmethod
    def create_results_dir():
        if not os.path.exists('data'):
            os.makedirs('data', exist_ok=True)
