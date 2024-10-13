from enum import Enum


class PathShortestCol(Enum):
    SOURCE = 'Source'
    TARGET = 'Target'
    SHORTEST_PATH = 'Shortest Path'
    SHORTEST_PATH_LENGTH = 'Shortest Path Length'
    SHORTEST_PATH_WEIGHT = 'Shortest Path Weight'
    SWEET_PATH_TYPE = 'Sweet Path Status'
    DECAYING_PATH_TYPE = 'Decaying Path Status'
    EDGE_TYPES = 'Edge Types'
    SWEET_HOPS = 'Sweet Hops'
    DECAYING_HOPS = 'Decaying Hops'
    GT_C2C_REL_POSE = 'Ground Truth C2C Relative Pose (cm)'
    C_C2C_REL_POSE = 'Computed C2C Relative Pose (cm)'
    GT_C2C_DIST = 'Ground Truth C2C Distance (cm)'
    C_C2C_DIST = 'Computed C2C Distance (cm)'
    DIST_ERROR = 'Distance Error (%)'
    ANGLE_ERROR = 'Angle Between Computed Ground Truth (degree)'
    THETA_ERROR = 'Theta Error (degree)'
    PHI_ERROR = 'Phi Error (degree)'

    def __str__(self):
        return self.value


class PathAveragingCol(Enum):
    SOURCE = 'Source'
    TARGET = 'Target'
    SHORTEST_PATH_1 = 'Shortest Path 1'
    SHORTEST_PATH_2 = 'Shortest Path 2'
    SHORTEST_PATH_1_LENGTH = 'Shortest Path 1 Length'
    SHORTEST_PATH_2_LENGTH = 'Shortest Path 2 Length'
    SHORTEST_PATH_1_WEIGHT = 'Shortest Path 1 Weight'
    SHORTEST_PATH_2_WEIGHT = 'Shortest Path 2 Weight'
    SWEET_PATH_1_TYPE = 'Sweet Path 1 Status'
    SWEET_PATH_2_TYPE = 'Sweet Path 2 Status'
    DECAYING_PATH_1_TYPE = 'Decaying Path 1 Status'
    DECAYING_PATH_2_TYPE = 'Decaying Path 2 Status'
    SWEET_HOPS_1 = 'Sweet Hops in Path 1'
    SWEET_HOPS_2 = 'Sweet Hops in Path 2'
    DECAYING_HOPS_1 = 'Decaying Hops in Path 1'
    DECAYING_HOPS_2 = 'Decaying Hops in Path 2'
    GT_C2C_REL_POSE = 'Ground Truth C2C Relative Pose (cm)'
    C_C2C_REL_POSE = 'Computed C2C Avg Relative Pose (cm)'
    GT_C2C_DIST = 'Ground Truth C2C Distance (cm)'
    C_C2C_DIST = 'Computed C2C Avg Distance (cm)'
    DIST_ERROR = 'Distance Error (%)'
    ANGLE_ERROR = 'Angle Between Computed Ground Truth (degree)'
    THETA_ERROR = 'Theta Error (degree)'
    PHI_ERROR = 'Phi Error (degree)'

    def __str__(self):
        return self.value


class PathMoveBlockingCol(Enum):
    SOURCE = 'Source'
    TARGET = 'Target'
    SHORTEST_PATH = 'Shortest Path'
    SHORTEST_PATH_LENGTH = 'Shortest Path Length'
    SHORTEST_PATH_WEIGHT = 'Shortest Path Weight'
    SWEET_PATH_TYPE = 'Sweet Path Status'
    DECAYING_PATH_TYPE = 'Decaying Path Status'
    EDGE_TYPES = 'Edge Types'
    SWEET_HOPS = 'Sweet Hops'
    DECAYING_HOPS = 'Decaying Hops'
    NUM_BLOCKING_FLSS = 'Number of Blocking FLSs'
    TOTAL_DIST_BLOCKING_MOVED = 'Total Dist Blocking Moved (cm)'
    MIN_DIST_BLOCKING_MOVED = 'Min Dist Blocking Moved (cm)'
    AVG_DIST_BLOCKING_MOVED = 'Avg Dist Blocking Moved (cm)'
    MAX_DIST_BLOCKING_MOVED = 'Max Dist Blocking Moved (cm)'
    CANNOT_RESOLVE_OCCLUSION = 'occlusion not resolved'
    GT_C2C_REL_POSE = 'Ground Truth C2C Relative Pose (cm)'
    C_C2C_REL_POSE = 'Computed C2C Relative Pose (cm)'
    GT_C2C_DIST = 'Ground Truth C2C Distance (cm)'
    C_C2C_DIST = 'Computed C2C Distance (cm)'
    DIST_ERROR = 'Distance Error (%)'
    ANGLE_ERROR = 'Angle Between Computed Ground Truth (degree)'
    THETA_ERROR = 'Theta Error (degree)'
    PHI_ERROR = 'Phi Error (degree)'

    def __str__(self):
        return self.value


class PathMoveSourceCol(Enum):
    SOURCE = 'Source'
    TARGET = 'Target'
    SHORTEST_PATH = 'Shortest Path'
    SHORTEST_PATH_LENGTH = 'Shortest Path Length'
    SHORTEST_PATH_WEIGHT = 'Shortest Path Weight'
    SWEET_PATH_TYPE = 'Sweet Path Status'
    DECAYING_PATH_TYPE = 'Decaying Path Status'
    EDGE_TYPES = 'Edge Types'
    SWEET_HOPS = 'Sweet Hops'
    DECAYING_HOPS = 'Decaying Hops'
    BLIND_HOPS = 'Blind Hops'
    NUM_MOVED = 'Number of Times Source Moved'
    TOTAL_DIST_SOURCE_MOVED = 'Total Dist Source Moved (cm)'
    RELATIVE_DIST_SOURCE_MOVED = 'Relative Dist Source Moved (cm)'
    NUM_SOURCE_COLLISION = 'Number of Source Collisions'
    GT_C2C_REL_POSE = 'Ground Truth C2C Relative Pose (cm)'
    C_C2C_REL_POSE = 'Computed C2C Relative Pose (cm)'
    GT_C2C_DIST = 'Ground Truth C2C Distance (cm)'
    C_C2C_DIST = 'Computed C2C Distance (cm)'
    DIST_ERROR = 'Distance Error (%)'
    ANGLE_ERROR = 'Angle Between Computed Ground Truth (degree)'
    THETA_ERROR = 'Theta Error (degree)'
    PHI_ERROR = 'Phi Error (degree)'

    def __str__(self):
        return self.value


class StatCol(Enum):
    NAME = 'Name'
    VALUE = 'Value'
    MIN = 'Min'
    MEAN = 'Mean'
    MAX = 'Max'
    STD = 'std'

    def __str__(self):
        return self.value


class PropCol(Enum):
    NAME = 'Name'
    VALUE = 'Value'

    def __str__(self):
        return self.value


class EdgeType(Enum):
    BLIND = 'Blind'
    SWEET = 'Sweet'
    DECAYING = 'Decaying'
    ALL = 'All'


class PathStatus(Enum):
    EXISTS = 'Exists'
    BLOCKED = 'Blocked'
    NO_PATH = 'No Path'
    NOT_COMPUTED = 'Not Computed'
    BLOCKED_OR_NO_PATH = 'Blocked or No Path'

    def __str__(self):
        return self.value
