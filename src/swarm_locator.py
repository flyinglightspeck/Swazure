from itertools import product

import networkx as nx
import numpy as np
import pandas as pd

from const import PathStatus, PathShortestCol, PropCol, PathAveragingCol, StatCol, PathMoveBlockingCol, PathMoveSourceCol, EdgeType
from offline_alg import OfflineAlg
from utils import angle_between_vectors, diff_spherical_angles


def get_path(path_gen):
    try:
        return next(path_gen, None)
    except nx.NetworkXNoPath:
        return None


def get_col_stats(col):
    return None, col.min(), col.mean(), col.max(), col.std()


class SwarmLocatorShortest(OfflineAlg):
    def compute_path(self, i, j):
        sweet_path_status = PathStatus.NOT_COMPUTED
        decaying_path_status = PathStatus.NOT_COMPUTED
        shortest_path = self.orc.get_sweet_shortest_path(i, j)
        edge_types = self.orc.get_edge_types(shortest_path)
        if len(shortest_path):
            sweet_path_status = PathStatus.EXISTS
        else:
            if nx.has_path(self.orc.all_sweet_neighbors_graph, i, j):
                sweet_path_status = PathStatus.BLOCKED
            else:
                sweet_path_status = PathStatus.NO_PATH

            shortest_path = self.orc.get_sweet_and_decaying_shortest_path(i, j)
            edge_types = self.orc.get_edge_types(shortest_path)
            if len(shortest_path):
                decaying_path_status = PathStatus.EXISTS
            else:
                if nx.has_path(self.orc.all_sweet_and_decaying_neighbors_graph, i, j):
                    decaying_path_status = PathStatus.BLOCKED
                else:
                    decaying_path_status = PathStatus.NO_PATH

        return shortest_path, edge_types, sweet_path_status, decaying_path_status

    def run(self):
        result_rows = []
        for i in range(len(self.orc.visible_sweet_neighbors)):
            for j in self.orc.blind_neighbors[i]:
                shortest_path, edge_types, sweet_path_status, decaying_path_status = self.compute_path(i, j)
                num_total_hops, num_sweet_hops, num_decaying_hops, _ = self.orc.get_path_hops(edge_types)
                c_pose = self.orc.compute_relative_pose_by_path(shortest_path)
                gt_pose = self.orc.points[j] - self.orc.points[i]
                norm_c_pose = np.linalg.norm(c_pose)
                norm_gt_pose = np.linalg.norm(gt_pose)
                angle_between_poses = angle_between_vectors(c_pose, gt_pose)
                d_theta, d_phi = diff_spherical_angles(c_pose, gt_pose)
                result_rows.append(
                    [
                        i,
                        j,
                        shortest_path,
                        num_total_hops,
                        self.orc.get_path_weight(shortest_path),
                        sweet_path_status,
                        decaying_path_status,
                        edge_types,
                        num_sweet_hops,
                        num_decaying_hops,
                        gt_pose,
                        c_pose,
                        norm_gt_pose,
                        norm_c_pose,
                        abs(norm_c_pose - norm_gt_pose) / norm_gt_pose * 100,
                        angle_between_poses,
                        d_theta,
                        d_phi
                    ])
        self.df = pd.DataFrame(result_rows, columns=list(PathShortestCol))

    def get_file_name(self):
        w_flag = '_w_' if self.orc.args.weighted else '_'
        return f"data/single{w_flag}{self.orc.args.shape}_b{self.orc.args.radius_beta}"

    def store(self):
        args_df = pd.DataFrame(list(vars(self.orc.args).items()), columns=list(PropCol))
        grouped = self.df.groupby(PathShortestCol.SOURCE)
        unreachable_percentage = grouped[PathShortestCol.SHORTEST_PATH_LENGTH].apply(
            lambda x: x.isnull().mean() * 100).reset_index(
            name='value')
        decaying_paths_df = self.df[self.df[PathShortestCol.DECAYING_PATH_TYPE] == PathStatus.EXISTS]
        sweet_paths_df = self.df[self.df[PathShortestCol.SWEET_PATH_TYPE] == PathStatus.EXISTS]

        stat_rows = [
            ['Shortest Distance Between Pairs (cm)', None, self.orc.min_dist, self.orc.avg_dist, self.orc.max_dist],
            ['Computed Radius (cm)', self.orc.radius],
            ['Removed Points', len(self.orc.colliding_points)],
            ['Unreachable Blind Fls Pairs', self.df[PathShortestCol.SHORTEST_PATH_LENGTH].isnull().sum()],
            ['Total Blind Fls Pairs', self.df.shape[0]],
            ['Percentage of Unreachable Blind Neighbors (%)', *get_col_stats(unreachable_percentage['value'])],
            ['Kissing Neighbors', self.orc.num_kissing_neighbors],
            ['Pairs w/ No Sweet Path', (self.df[PathShortestCol.SWEET_PATH_TYPE] == PathStatus.NO_PATH).sum()],
            ['Pairs w/ Blocked Sweet Path', (self.df[PathShortestCol.SWEET_PATH_TYPE] == PathStatus.BLOCKED).sum()],
            ['Pairs w/ Sweet Path', (self.df[PathShortestCol.SWEET_PATH_TYPE] == PathStatus.EXISTS).sum()],
            ['Pairs w/ Decaying Path', (self.df[PathShortestCol.DECAYING_PATH_TYPE] == PathStatus.EXISTS).sum()],
            ['Hops of All Paths', *get_col_stats(self.df[PathShortestCol.SHORTEST_PATH_LENGTH])],
            ['Hops of Sweet Paths', *get_col_stats(sweet_paths_df[PathShortestCol.SHORTEST_PATH_LENGTH])],
            ['Hops of Decaying Paths', *get_col_stats(decaying_paths_df[PathShortestCol.SHORTEST_PATH_LENGTH])],
            ['Decaying Hops of Decaying Paths', *get_col_stats(decaying_paths_df[PathShortestCol.DECAYING_HOPS])],
            ['Sweet Hops of Decaying Paths', *get_col_stats(decaying_paths_df[PathShortestCol.SWEET_HOPS])],
            ['Weight of All Paths', *get_col_stats(self.df[PathShortestCol.SHORTEST_PATH_WEIGHT])],
            ['Weight of Sweet Paths', *get_col_stats(sweet_paths_df[PathShortestCol.SHORTEST_PATH_WEIGHT])],
            ['Weight of Decaying Paths', *get_col_stats(decaying_paths_df[PathShortestCol.SHORTEST_PATH_WEIGHT])],
            ['Distance Error (%)', *get_col_stats(self.df[PathShortestCol.DIST_ERROR])],
            ['Angle Error (degree)', *get_col_stats(self.df[PathShortestCol.ANGLE_ERROR])],
            ['Theta Error (degree)', *get_col_stats(self.df[PathShortestCol.THETA_ERROR])],
            ['Phi Error (degree)', *get_col_stats(self.df[PathShortestCol.PHI_ERROR])],
        ]

        stats_df = pd.DataFrame(stat_rows, columns=list(StatCol))

        file_name = self.get_file_name()
        self.df.to_csv(f'{file_name}_paths.csv', index=False)
        with pd.ExcelWriter(f'{file_name}.xlsx', engine='xlsxwriter') as writer:
            stats_df.to_excel(writer, sheet_name='stats', index=False)
            args_df.to_excel(writer, sheet_name='args', index=False)


class SwarmLocatorAveraging(OfflineAlg):
    def compute_two_path(self, i, j):
        sweet_path_status = [PathStatus.NOT_COMPUTED, PathStatus.NOT_COMPUTED]
        decaying_path_status = [PathStatus.NOT_COMPUTED, PathStatus.NOT_COMPUTED]

        sweet_shortest_paths = self.orc.get_sweet_shortest_path_generator(source=i, target=j)
        decaying_shortest_paths = self.orc.get_sweet_and_decaying_shortest_path_generator(source=i, target=j)

        sweet_path1 = get_path(sweet_shortest_paths)
        if sweet_path1:
            sweet_path_status[0] = PathStatus.EXISTS
            sweet_path2 = get_path(sweet_shortest_paths)
            if sweet_path2:
                sweet_path_status[1] = PathStatus.EXISTS
                return [sweet_path1, sweet_path2], sweet_path_status, decaying_path_status
            else:
                sweet_path_status[1] = PathStatus.BLOCKED_OR_NO_PATH

                decaying_path1 = get_path(decaying_shortest_paths)
                if decaying_path1:
                    decaying_path_status[0] = PathStatus.EXISTS
                    return [sweet_path1, decaying_path1], sweet_path_status, decaying_path_status
                else:
                    decaying_path_status[0] = PathStatus.BLOCKED_OR_NO_PATH
                    return [sweet_path1], sweet_path_status, decaying_path_status
        else:
            sweet_path_status[0] = PathStatus.BLOCKED_OR_NO_PATH

            decaying_path1 = get_path(decaying_shortest_paths)
            if decaying_path1:
                decaying_path_status[0] = PathStatus.EXISTS
                decaying_path2 = get_path(decaying_shortest_paths)
                if decaying_path2:
                    decaying_path_status[1] = PathStatus.EXISTS
                    return [decaying_path1, decaying_path2], sweet_path_status, decaying_path_status
                else:
                    decaying_path_status[1] = PathStatus.BLOCKED_OR_NO_PATH
                    return [decaying_path1], sweet_path_status, decaying_path_status
            else:
                decaying_path_status[0] = PathStatus.BLOCKED_OR_NO_PATH
                return [[]], sweet_path_status, decaying_path_status

    def run(self):
        result_rows = []
        for i in range(len(self.orc.visible_sweet_neighbors)):
            for j in self.orc.blind_neighbors[i]:
                shortest_paths, sweet_path_status, decaying_path_status = self.compute_two_path(i, j)
                path1 = shortest_paths[0]
                path2 = shortest_paths[1] if len(shortest_paths) == 2 else []

                edge_types1 = self.orc.get_edge_types(path1)
                edge_types2 = self.orc.get_edge_types(path2)
                num_total_hops1, num_sweet_hops1, num_decaying_hops1, _ = self.orc.get_path_hops(edge_types1)
                num_total_hops2, num_sweet_hops2, num_decaying_hops2, _ = self.orc.get_path_hops(edge_types2)

                c_pose1 = self.orc.compute_relative_pose_by_path(path1)
                if len(path2):
                    c_pose2 = self.orc.compute_relative_pose_by_path(path2)
                    c_pose_avg = (c_pose1 + c_pose2) / 2
                else:
                    c_pose_avg = c_pose1
                gt_pose = self.orc.points[j] - self.orc.points[i]
                norm_c_pose = np.linalg.norm(c_pose_avg)
                norm_gt_pose = np.linalg.norm(gt_pose)
                angle_between_poses = angle_between_vectors(c_pose_avg, gt_pose)
                d_theta, d_phi = diff_spherical_angles(c_pose_avg, gt_pose)
                result_rows.append(
                    [
                        i,
                        j,
                        path1,
                        path2,
                        num_total_hops1,
                        num_total_hops2,
                        self.orc.get_path_weight(path1),
                        self.orc.get_path_weight(path2),
                        sweet_path_status[0],
                        sweet_path_status[1],
                        decaying_path_status[0],
                        decaying_path_status[1],
                        num_sweet_hops1,
                        num_sweet_hops2,
                        num_decaying_hops1,
                        num_decaying_hops2,
                        gt_pose,
                        c_pose_avg,
                        norm_gt_pose,
                        norm_c_pose,
                        abs(norm_c_pose - norm_gt_pose) / norm_gt_pose * 100,
                        angle_between_poses,
                        d_theta,
                        d_phi
                    ])
        self.df = pd.DataFrame(result_rows, columns=list(PathAveragingCol))

    def get_file_name(self):
        w_flag = '_w_' if self.orc.args.weighted else '_'
        return f"data/dual{w_flag}{self.orc.args.shape}_b{self.orc.args.radius_beta}"

    def store(self):
        args_df = pd.DataFrame(list(vars(self.orc.args).items()), columns=list(PropCol))
        grouped = self.df.groupby(PathAveragingCol.SOURCE)
        unreachable_percentage = grouped[PathAveragingCol.SHORTEST_PATH_1_LENGTH].apply(
            lambda x: x.isnull().mean() * 100).reset_index(
            name='value')
        # decaying_paths_df = self.df[self.df[PathDualCol.DECAYING_PATH_TYPE] == PathStatus.EXISTS]
        # sweet_paths_df = self.df[self.df[PathDualCol.SWEET_PATH_TYPE] == PathStatus.EXISTS]

        stat_rows = [
            ['Shortest Distance Between Pairs (cm)', None, self.orc.min_dist, self.orc.avg_dist, self.orc.max_dist],
            ['Computed Radius (cm)', self.orc.radius],
            ['Removed Points', len(self.orc.colliding_points)],
            ['Unreachable Blind Fls Pairs', self.df[PathAveragingCol.SHORTEST_PATH_1_LENGTH].isnull().sum()],
            ['Total Blind Fls Pairs', self.df.shape[0]],
            ['Percentage of Unreachable Blind Neighbors (%)', *get_col_stats(unreachable_percentage['value'])],
            ['Kissing Neighbors', self.orc.num_kissing_neighbors],
            ['Pairs w/ 1 Sweet Path', ((self.df[PathAveragingCol.SWEET_PATH_1_TYPE] == PathStatus.EXISTS) ^ (
                    self.df[PathAveragingCol.SWEET_PATH_2_TYPE] == PathStatus.EXISTS)).sum()],
            ['Pairs w/ 2 Sweet Path', ((self.df[PathAveragingCol.SWEET_PATH_1_TYPE] == PathStatus.EXISTS) & (
                    self.df[PathAveragingCol.SWEET_PATH_2_TYPE] == PathStatus.EXISTS)).sum()],
            ['Pairs w/ 2 Decaying Path', ((self.df[PathAveragingCol.DECAYING_PATH_1_TYPE] == PathStatus.EXISTS) & (
                    self.df[PathAveragingCol.DECAYING_PATH_2_TYPE] == PathStatus.EXISTS)).sum()],
            ['Pairs w/ only 1 Sweet Path', ((self.df[PathAveragingCol.SWEET_PATH_1_TYPE] == PathStatus.EXISTS) & (
                    self.df[PathAveragingCol.SWEET_PATH_2_TYPE] != PathStatus.EXISTS) & (self.df[
                                                                                        PathAveragingCol.DECAYING_PATH_1_TYPE] != PathStatus.EXISTS)).sum()],
            ['Pairs w/ only 1 Decaying Path', ((self.df[PathAveragingCol.DECAYING_PATH_1_TYPE] == PathStatus.EXISTS) & (
                    self.df[PathAveragingCol.DECAYING_PATH_2_TYPE] != PathStatus.EXISTS) & (self.df[
                                                                                           PathAveragingCol.SWEET_PATH_1_TYPE] != PathStatus.EXISTS)).sum()],
            ['Hops of Path 1', *get_col_stats(self.df[PathAveragingCol.SHORTEST_PATH_1_LENGTH])],
            ['Hops of Path 2', *get_col_stats(self.df[PathAveragingCol.SHORTEST_PATH_2_LENGTH])],
            ['Weight of Path 1', *get_col_stats(self.df[PathAveragingCol.SHORTEST_PATH_1_WEIGHT])],
            ['Weight of Path 2', *get_col_stats(self.df[PathAveragingCol.SHORTEST_PATH_2_WEIGHT])],
            ['Sweet Hops of Path 1', *get_col_stats(self.df[PathAveragingCol.SWEET_HOPS_1])],
            ['Sweet Hops of Path 2', *get_col_stats(self.df[PathAveragingCol.SWEET_HOPS_2])],
            ['Decaying Hops of Path 1', *get_col_stats(self.df[PathAveragingCol.DECAYING_HOPS_1])],
            ['Decaying Hops of Path 2', *get_col_stats(self.df[PathAveragingCol.DECAYING_HOPS_2])],
            ['Distance Error (%)', *get_col_stats(self.df[PathAveragingCol.DIST_ERROR])],
            ['Angle Error (degree)', *get_col_stats(self.df[PathAveragingCol.ANGLE_ERROR])],
            ['Theta Error (degree)', *get_col_stats(self.df[PathAveragingCol.THETA_ERROR])],
            ['Phi Error (degree)', *get_col_stats(self.df[PathAveragingCol.PHI_ERROR])],
        ]

        stats_df = pd.DataFrame(stat_rows, columns=list(StatCol))

        file_name = self.get_file_name()
        self.df.to_csv(f'{file_name}_paths.csv', index=False)
        with pd.ExcelWriter(f'{file_name}.xlsx', engine='xlsxwriter') as writer:
            stats_df.to_excel(writer, sheet_name='stats', index=False)
            args_df.to_excel(writer, sheet_name='args', index=False)


class SwarmLocatorMoveBlocking(OfflineAlg):
    def compute_visible_sweet_path(self, i, j):
        sweet_path_status = PathStatus.NOT_COMPUTED
        decaying_path_status = PathStatus.NOT_COMPUTED
        shortest_path = self.orc.get_sweet_shortest_path(i, j)
        edge_types = self.orc.get_edge_types(shortest_path)
        if len(shortest_path):
            sweet_path_status = PathStatus.EXISTS
        else:
            if nx.has_path(self.orc.all_sweet_neighbors_graph, i, j):
                sweet_path_status = PathStatus.BLOCKED
            else:
                sweet_path_status = PathStatus.NO_PATH

        return shortest_path, edge_types, sweet_path_status, decaying_path_status

    def compute_shortest_sweet_path(self, i, j):
        sweet_path_status = PathStatus.NOT_COMPUTED
        decaying_path_status = PathStatus.NOT_COMPUTED
        shortest_path = self.orc.get_shortest_path(i, j)
        edge_types = self.orc.get_edge_types(shortest_path)

        return shortest_path, edge_types, sweet_path_status, decaying_path_status

    def compute_decaying_path(self, i, j):
        shortest_path = self.orc.get_sweet_and_decaying_shortest_path(i, j)
        edge_types = self.orc.get_edge_types(shortest_path)
        if len(shortest_path):
            decaying_path_status = PathStatus.EXISTS
        else:
            if nx.has_path(self.orc.all_sweet_and_decaying_neighbors_graph, i, j):
                decaying_path_status = PathStatus.BLOCKED
            else:
                decaying_path_status = PathStatus.NO_PATH
        return shortest_path, edge_types, decaying_path_status

    def resolve_occlusions(self, path):
        all_dists = []
        occlusion_remains = None
        for i in range(len(path) - 1):
            is_occluded = self.orc.is_occluded(path[i], path[i + 1])
            if is_occluded:
                dists, occlusion_remains = self.orc.get_correction_dist_all_occlusions(path[i], path[i + 1],
                                                                                       check_collision=True)
                all_dists += dists

        if len(all_dists):
            if occlusion_remains:
                return len(all_dists), None, None, None, None, occlusion_remains
            else:
                return len(all_dists), np.sum(all_dists), np.min(all_dists), np.mean(all_dists), np.max(
                    all_dists), occlusion_remains
        else:
            return 0, None, None, None, None, occlusion_remains

    def run(self):
        use_decaying = '+' in self.orc.args.solution
        result_rows = []
        for i in range(len(self.orc.visible_sweet_neighbors)):
            for j in self.orc.blind_neighbors[i]:
                shortest_path, edge_types, sweet_path_status, decaying_path_status = self.compute_visible_sweet_path(i,
                                                                                                                     j)
                num_blocking, total_dist_moved, min_dist_moved, avg_dist_moved, max_dist_moved, occlusion_remains = 0, None, None, None, None, None
                if len(shortest_path) == 0:
                    shortest_path, edge_types, sweet_path_status, decaying_path_status = self.compute_shortest_sweet_path(
                        i, j)
                    num_blocking, total_dist_moved, min_dist_moved, avg_dist_moved, max_dist_moved, occlusion_remains = self.resolve_occlusions(
                        shortest_path)
                    if occlusion_remains:
                        # use decaying path
                        if use_decaying:
                            shortest_path, edge_types, decaying_path_status = self.compute_decaying_path(i, j)
                        else:
                            shortest_path, edge_types = [], []

                num_total_hops, num_sweet_hops, num_decaying_hops, _ = self.orc.get_path_hops(edge_types)
                c_pose = self.orc.compute_relative_pose_by_path(shortest_path)
                gt_pose = self.orc.points[j] - self.orc.points[i]
                norm_c_pose = np.linalg.norm(c_pose)
                norm_gt_pose = np.linalg.norm(gt_pose)
                angle_between_poses = angle_between_vectors(c_pose, gt_pose)
                d_theta, d_phi = diff_spherical_angles(c_pose, gt_pose)

                if num_sweet_hops is not None and num_sweet_hops > 0 and num_decaying_hops == 0:
                    sweet_path_status = PathStatus.EXISTS
                if num_decaying_hops is not None and num_decaying_hops > 0:
                    decaying_path_status = PathStatus.EXISTS
                result_rows.append(
                    [
                        i,
                        j,
                        shortest_path,
                        num_total_hops,
                        self.orc.get_path_weight(shortest_path),
                        sweet_path_status,
                        decaying_path_status,
                        edge_types,
                        num_sweet_hops,
                        num_decaying_hops,
                        num_blocking,
                        total_dist_moved,
                        min_dist_moved,
                        avg_dist_moved,
                        max_dist_moved,
                        occlusion_remains,
                        gt_pose,
                        c_pose,
                        norm_gt_pose,
                        norm_c_pose,
                        abs(norm_c_pose - norm_gt_pose) / norm_gt_pose * 100,
                        angle_between_poses,
                        d_theta,
                        d_phi
                    ])
        self.df = pd.DataFrame(result_rows, columns=list(PathMoveBlockingCol))

    def get_file_name(self):
        w_flag = '_w_' if self.orc.args.weighted else '_'
        return f"data/{self.orc.args.solution}{w_flag}{self.orc.args.shape}_b{self.orc.args.radius_beta}"

    def store(self):
        args_df = pd.DataFrame(list(vars(self.orc.args).items()), columns=list(PropCol))
        grouped = self.df.groupby(PathMoveBlockingCol.SOURCE)
        unreachable_percentage = grouped[PathMoveBlockingCol.SHORTEST_PATH_LENGTH].apply(
            lambda x: x.isnull().mean() * 100).reset_index(
            name='value')
        decaying_paths_df = self.df[self.df[PathMoveBlockingCol.DECAYING_PATH_TYPE] == PathStatus.EXISTS]
        sweet_paths_df = self.df[self.df[PathMoveBlockingCol.SWEET_PATH_TYPE] == PathStatus.EXISTS]

        stat_rows = [
            ['Shortest Distance Between Pairs (cm)', None, self.orc.min_dist, self.orc.avg_dist, self.orc.max_dist],
            ['Computed Radius (cm)', self.orc.radius],
            ['Removed Points', len(self.orc.colliding_points)],
            ['Unreachable Blind FLS Pairs', self.df[PathMoveBlockingCol.SHORTEST_PATH_LENGTH].isnull().sum()],
            ['Total Blind FLS Pairs', self.df.shape[0]],
            ['Total Times Solution Applied', self.df[PathMoveBlockingCol.CANNOT_RESOLVE_OCCLUSION].notna().sum()],
            ['Total Times Solution Worked',
             self.df[self.df[PathMoveBlockingCol.CANNOT_RESOLVE_OCCLUSION] == False].shape[0]],
            ['Percentage of Unreachable Blind Neighbors (%)', *get_col_stats(unreachable_percentage['value'])],
            ['Kissing Neighbors', self.orc.num_kissing_neighbors],
            ['Pairs w/ No Sweet Path', (self.df[PathMoveBlockingCol.SWEET_PATH_TYPE] == PathStatus.NO_PATH).sum()],
            ['Pairs w/ Blocked Sweet Path', (self.df[PathMoveBlockingCol.SWEET_PATH_TYPE] == PathStatus.BLOCKED).sum()],
            ['Pairs w/ Sweet Path', (self.df[PathMoveBlockingCol.SWEET_PATH_TYPE] == PathStatus.EXISTS).sum()],
            ['Pairs w/ Decaying Path', (self.df[PathMoveBlockingCol.DECAYING_PATH_TYPE] == PathStatus.EXISTS).sum()],
            ['Hops of All Paths', *get_col_stats(self.df[PathMoveBlockingCol.SHORTEST_PATH_LENGTH])],
            ['Hops of Sweet Paths', *get_col_stats(sweet_paths_df[PathMoveBlockingCol.SHORTEST_PATH_LENGTH])],
            ['Hops of Decaying Paths', *get_col_stats(decaying_paths_df[PathMoveBlockingCol.SHORTEST_PATH_LENGTH])],
            ['Decaying Hops of Decaying Paths', *get_col_stats(decaying_paths_df[PathMoveBlockingCol.DECAYING_HOPS])],
            ['Sweet Hops of Decaying Paths', *get_col_stats(decaying_paths_df[PathMoveBlockingCol.SWEET_HOPS])],
            ['Weight of All Paths', *get_col_stats(self.df[PathMoveBlockingCol.SHORTEST_PATH_WEIGHT])],
            ['Weight of Sweet Paths', *get_col_stats(sweet_paths_df[PathMoveBlockingCol.SHORTEST_PATH_WEIGHT])],
            ['Weight of Decaying Paths', *get_col_stats(decaying_paths_df[PathMoveBlockingCol.SHORTEST_PATH_WEIGHT])],
            ['Moved Blocking FLSs', *get_col_stats(self.df[PathMoveBlockingCol.NUM_BLOCKING_FLSS])],
            ['Total Distance Blocking FLSs Moved (cm)',
             *get_col_stats(self.df[PathMoveBlockingCol.TOTAL_DIST_BLOCKING_MOVED])],
            ['Avg Distance Blocking FLSs Moved (cm)',
             *get_col_stats(self.df[PathMoveBlockingCol.AVG_DIST_BLOCKING_MOVED])],
            ['Distance Error (%)', *get_col_stats(self.df[PathMoveBlockingCol.DIST_ERROR])],
            ['Angle Error (degree)', *get_col_stats(self.df[PathMoveBlockingCol.ANGLE_ERROR])],
            ['Theta Error (degree)', *get_col_stats(self.df[PathMoveBlockingCol.THETA_ERROR])],
            ['Phi Error (degree)', *get_col_stats(self.df[PathMoveBlockingCol.PHI_ERROR])],
        ]

        stats_df = pd.DataFrame(stat_rows, columns=list(StatCol))

        file_name = self.get_file_name()
        self.df.to_csv(f'{file_name}_paths.csv', index=False)
        with pd.ExcelWriter(f'{file_name}.xlsx', engine='xlsxwriter') as writer:
            stats_df.to_excel(writer, sheet_name='stats', index=False)
            args_df.to_excel(writer, sheet_name='args', index=False)


class SwarmLocatorMoveSource(OfflineAlg):
    def __init__(self, orc):
        super().__init__(orc)
        self.direction_vectors = []
        self.steps_threshold = orc.args.steps_threshold

    def set_direction_vectors(self, steps=1):
        direction_vectors = list(product(range(-steps, steps + 1), repeat=3))
        direction_vectors.remove((0, 0, 0))

        self.direction_vectors = np.array(direction_vectors) * self.orc.radius

    def compute_sweet_path(self, i, j):
        shortest_path = self.orc.get_sweet_shortest_path(i, j)
        edge_types = self.orc.get_edge_types(shortest_path)
        if len(shortest_path):
            sweet_path_status = PathStatus.EXISTS
        else:
            if nx.has_path(self.orc.all_sweet_neighbors_graph, i, j):
                sweet_path_status = PathStatus.BLOCKED
            else:
                sweet_path_status = PathStatus.NO_PATH

        return shortest_path, edge_types, sweet_path_status

    def compute_decaying_path(self, i, j):
        shortest_path = self.orc.get_sweet_and_decaying_shortest_path(i, j)
        edge_types = self.orc.get_edge_types(shortest_path)
        if len(shortest_path):
            decaying_path_status = PathStatus.EXISTS
        else:
            if nx.has_path(self.orc.all_sweet_and_decaying_neighbors_graph, i, j):
                decaying_path_status = PathStatus.BLOCKED
            else:
                decaying_path_status = PathStatus.NO_PATH
        return shortest_path, edge_types, decaying_path_status

    def search_for_common_visible_neighbor(self, source, neighbors):
        p = self.orc.points[source]
        dist_moved = 0
        num_moved = 0
        new_coord = p
        num_collisions = 0
        blocked_directions = set()
        for k in range(self.steps_threshold):
            for i, v in enumerate(self.direction_vectors):
                if i in blocked_directions:
                    continue

                kv = (k + 1) * v
                dist_moved += np.linalg.norm(new_coord - p + kv)
                num_moved += 1
                new_coord = p + kv

                if self.orc.collide_by_proximity(source, new_coord, self.steps_threshold * self.orc.radius):
                    blocked_directions.add(i)
                    num_collisions += 1
                    continue

                for j in neighbors:
                    if self.orc.get_edge_type_by_points(new_coord, self.orc.points[j]) == EdgeType.SWEET:
                        if not self.orc.is_new_coord_occluded(source, j, new_coord):
                            return j, new_coord, dist_moved, np.linalg.norm(p - new_coord), num_moved, num_collisions
        return None, p, dist_moved, 0, num_moved, num_collisions

    def run(self):
        use_decaying = '+' in self.orc.args.solution
        self.set_direction_vectors()

        result_rows = []
        for i in range(len(self.orc.visible_sweet_neighbors)):
            for j in self.orc.blind_neighbors[i]:
                shortest_path, edge_types, sweet_path_status = self.compute_sweet_path(i, j)
                path_weight = self.orc.get_path_weight(shortest_path)
                c_pose = self.orc.compute_relative_pose_by_path(shortest_path)
                relative_dist_moved, total_dist_moved, num_moved, num_collisions = None, None, None, None
                decaying_path_status = PathStatus.NOT_COMPUTED
                new_coord = self.orc.points[i]
                if sweet_path_status == PathStatus.BLOCKED:
                    target_sweet_neighbors = self.orc.visible_sweet_neighbors[j]
                    hop, new_coord, total_dist_moved, relative_dist_moved, num_moved, num_collisions = self.search_for_common_visible_neighbor(
                        i, target_sweet_neighbors)
                    if hop is not None:
                        shortest_path = [i, hop, j]
                        shortest_path_coord = [new_coord, self.orc.points[hop], self.orc.points[j]]
                        path_weight = self.orc.get_path_weight_by_coord(shortest_path_coord)
                        edge_types = self.orc.get_edge_types_by_coord(shortest_path_coord)
                        c_pose = self.orc.compute_relative_pose_by_path_coord(shortest_path_coord)
                    elif use_decaying:
                        # use decaying path
                        shortest_path, edge_types, decaying_path_status = self.compute_decaying_path(i, j)
                        path_weight = self.orc.get_path_weight(shortest_path)
                        c_pose = self.orc.compute_relative_pose_by_path(shortest_path)

                num_total_hops, num_sweet_hops, num_decaying_hops, num_blind_hops = self.orc.get_path_hops(edge_types)
                gt_pose = self.orc.points[j] - new_coord
                norm_c_pose = np.linalg.norm(c_pose)
                norm_gt_pose = np.linalg.norm(gt_pose)
                angle_between_poses = angle_between_vectors(c_pose, gt_pose)
                d_theta, d_phi = diff_spherical_angles(c_pose, gt_pose)

                result_rows.append(
                    [
                        i,
                        j,
                        shortest_path,
                        num_total_hops,
                        path_weight,
                        sweet_path_status,
                        decaying_path_status,
                        edge_types,
                        num_sweet_hops,
                        num_decaying_hops,
                        num_blind_hops,
                        num_moved,
                        total_dist_moved,
                        relative_dist_moved,
                        num_collisions,
                        gt_pose,
                        c_pose,
                        norm_gt_pose,
                        norm_c_pose,
                        abs(norm_c_pose - norm_gt_pose) / norm_gt_pose * 100,
                        angle_between_poses,
                        d_theta,
                        d_phi
                    ])
        self.df = pd.DataFrame(result_rows, columns=list(PathMoveSourceCol))

    def get_file_name(self):
        w_flag = '_w_' if self.orc.args.weighted else '_'
        return f"data/{self.orc.args.solution}{w_flag}{self.orc.args.shape}_b{self.orc.args.radius_beta}_t{self.steps_threshold}"

    def store(self):
        args_df = pd.DataFrame(list(vars(self.orc.args).items()), columns=list(PropCol))
        grouped = self.df.groupby(PathMoveSourceCol.SOURCE)
        unreachable_percentage = grouped[PathMoveSourceCol.SHORTEST_PATH_LENGTH].apply(
            lambda x: x.isnull().mean() * 100).reset_index(
            name='value')
        decaying_paths_df = self.df[self.df[PathMoveSourceCol.DECAYING_PATH_TYPE] == PathStatus.EXISTS]
        sweet_paths_df = self.df[self.df[PathMoveSourceCol.SWEET_PATH_TYPE] == PathStatus.EXISTS]

        stat_rows = [
            ['Shortest Distance Between Pairs (cm)', None, self.orc.min_dist, self.orc.avg_dist, self.orc.max_dist],
            ['Computed Radius (cm)', self.orc.radius],
            ['Removed Points', len(self.orc.colliding_points)],
            ['Unreachable Blind FLS Pairs', self.df[PathMoveSourceCol.SHORTEST_PATH_LENGTH].isnull().sum()],
            ['Total Blind FLS Pairs', self.df.shape[0]],
            ['Total Times Solution Applied', self.df[PathMoveSourceCol.NUM_MOVED].notna().sum()],
            ['Total Times Solution Worked', len(self.df[(self.df[PathMoveSourceCol.SHORTEST_PATH_LENGTH].notna()) & (
                self.df[PathMoveSourceCol.NUM_MOVED].notna()) & (
                                                            self.df[PathMoveSourceCol.DECAYING_PATH_TYPE] == PathStatus.NOT_COMPUTED)])],
            ['Percentage of Unreachable Blind Neighbors (%)', *get_col_stats(unreachable_percentage['value'])],
            ['Kissing Neighbors', self.orc.num_kissing_neighbors],
            ['Pairs w/ No Sweet Path', (self.df[PathMoveSourceCol.SWEET_PATH_TYPE] == PathStatus.NO_PATH).sum()],
            ['Pairs w/ Blocked Sweet Path', (self.df[PathMoveSourceCol.SWEET_PATH_TYPE] == PathStatus.BLOCKED).sum()],
            ['Pairs w/ Sweet Path', (self.df[PathMoveSourceCol.SWEET_PATH_TYPE] == PathStatus.EXISTS).sum()],
            ['Pairs w/ Decaying Path', (self.df[PathMoveSourceCol.DECAYING_HOPS] > 0).sum()],
            ['Pairs w/ Blind Path', (self.df[PathMoveSourceCol.BLIND_HOPS] > 0).sum()],
            # ['Pairs w/ Decaying Path', (self.df[PathMoveSourceCol.DECAYING_PATH_TYPE] == PathStatus.EXISTS).sum()],
            ['Hops of All Paths', *get_col_stats(self.df[PathMoveSourceCol.SHORTEST_PATH_LENGTH])],
            ['Sweet Hops of All Paths', *get_col_stats(self.df[PathMoveSourceCol.SWEET_HOPS])],
            ['Decaying Hops of All Paths', *get_col_stats(self.df[PathMoveSourceCol.DECAYING_HOPS])],
            ['Blind Hops of All Paths', *get_col_stats(self.df[PathMoveSourceCol.BLIND_HOPS])],
            ['Hops of Sweet Paths', *get_col_stats(sweet_paths_df[PathMoveSourceCol.SHORTEST_PATH_LENGTH])],
            ['Hops of Decaying Paths', *get_col_stats(decaying_paths_df[PathMoveSourceCol.SHORTEST_PATH_LENGTH])],
            ['Decaying Hops of Decaying Paths', *get_col_stats(decaying_paths_df[PathMoveSourceCol.DECAYING_HOPS])],
            ['Sweet Hops of Decaying Paths', *get_col_stats(decaying_paths_df[PathMoveSourceCol.SWEET_HOPS])],
            ['Weight of All Paths', *get_col_stats(self.df[PathMoveSourceCol.SHORTEST_PATH_WEIGHT])],
            ['Weight of Sweet Paths', *get_col_stats(sweet_paths_df[PathMoveSourceCol.SHORTEST_PATH_WEIGHT])],
            ['Weight of Decaying Paths', *get_col_stats(decaying_paths_df[PathMoveSourceCol.SHORTEST_PATH_WEIGHT])],
            ['Num Times Source FLS Moved', *get_col_stats(self.df[PathMoveSourceCol.NUM_MOVED])],
            ['Total Distance Source FLS Moved (cm)',
             *get_col_stats(self.df[PathMoveSourceCol.TOTAL_DIST_SOURCE_MOVED])],
            ['Number of Source Collisions', *get_col_stats(self.df[PathMoveSourceCol.NUM_SOURCE_COLLISION])],
            ['Distance Error (%)', *get_col_stats(self.df[PathMoveSourceCol.DIST_ERROR])],
            ['Angle Error (degree)', *get_col_stats(self.df[PathMoveSourceCol.ANGLE_ERROR])],
            ['Theta Error (degree)', *get_col_stats(self.df[PathMoveSourceCol.THETA_ERROR])],
            ['Phi Error (degree)', *get_col_stats(self.df[PathMoveSourceCol.PHI_ERROR])],
        ]

        stats_df = pd.DataFrame(stat_rows, columns=list(StatCol))

        file_name = self.get_file_name()
        self.df.to_csv(f'{file_name}_paths.csv', index=False)
        with pd.ExcelWriter(f'{file_name}.xlsx', engine='xlsxwriter') as writer:
            stats_df.to_excel(writer, sheet_name='stats', index=False)
            args_df.to_excel(writer, sheet_name='args', index=False)
