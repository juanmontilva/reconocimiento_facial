import numpy as np
from abc import ABC, abstractmethod


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, point1, point2):
        pass


class EuclideanDistanceCalculator(DistanceCalculator):
    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))


class EyebrowArchCalculator(ABC):
    @abstractmethod
    def calculate_eyebrow_arch(self, eyebrow_points):
        pass


class PolynomialEyebrowArchCalculator(EyebrowArchCalculator):
    def calculate_eyebrow_arch(self, eyebrow_points):
        x = [point[0] for point in eyebrow_points]
        y = [point[1] for point in eyebrow_points]
        z = np.polyfit(x, y, 2)
        return z[0]


class EyeBrowsPointsProcessing:
    def __init__(self, arch_calculator: EyebrowArchCalculator, distance_calculator: DistanceCalculator):
        self.arch_calculator = arch_calculator
        self.distance_calculator = distance_calculator
        self.eyebrows: dict = {}

    def calculate_distances(self, eyebrows_points: dict):
        dist1 = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][0], eyebrows_points['distances'][1])
        dist2 = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][2], eyebrows_points['distances'][3])
        dist3 = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][4], eyebrows_points['distances'][5])
        dist4 = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][6], eyebrows_points['distances'][7])
        dist5 = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][8], eyebrows_points['distances'][9])
        dist6 = self.distance_calculator.calculate_distance(
            eyebrows_points['distances'][10], eyebrows_points['distances'][11])

        return dist1, dist2, dist3, dist4, dist5, dist6

    def main(self, eyebrows_points: dict):
        # calculate eyebrow arch
        right_eyebrow_arch = self.arch_calculator.calculate_eyebrow_arch(eyebrows_points['right arch'])
        left_eyebrow_arch = self.arch_calculator.calculate_eyebrow_arch(eyebrows_points['left arch'])
        self.eyebrows['arch_right'] = right_eyebrow_arch
        self.eyebrows['arch_left'] = left_eyebrow_arch

        # calculate distances
        (dist1, dist2, dist3, dist4, dist5, dist6) = self.calculate_distances(eyebrows_points)
        self.eyebrows['dist1'] = dist1
        self.eyebrows['dist2'] = dist2
        self.eyebrows['dist3'] = dist3
        self.eyebrows['dist4'] = dist4
        self.eyebrows['dist5'] = dist5
        self.eyebrows['dist6'] = dist6
        return self.eyebrows

from abc import ABC, abstractmethod


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, point1, point2):
        pass


class EuclideanDistanceCalculator(DistanceCalculator):
    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))


class EyebrowArchCalculator(ABC):
    @abstractmethod
    def calculate_eyebrow_arch(self, eyebrow_points):
        pass


class PolynomialEyebrowArchCalculator(EyebrowArchCalculator):
    def calculate_eyebrow_arch(self, eyebrow_points):
        x = [point[0] for point in eyebrow_points]
        y = [point[1] for point in eyebrow_points]
        z = np.polyfit(x, y, 2)
        return z[0]


class EyeBrowsPointsProcessing:
    def __init__(self, arch_calculator: EyebrowArchCalculator, distance_calculator: DistanceCalculator):
        self.arch_calculator = arch_calculator
        self.distance_calculator = distance_calculator
        self.eyebrows: dict = {}

    def calculate_distances(self, eyebrows_points: dict):
        # Calculate mandatory distances (pairs are consecutive indices)
        distances = []
        num_pairs = len(eyebrows_points['distances']) // 2
        for i in range(num_pairs):
            p1 = eyebrows_points['distances'][2 * i]
            p2 = eyebrows_points['distances'][2 * i + 1]
            distances.append(self.distance_calculator.calculate_distance(p1, p2))

        # Ensure we always return exactly six distance values for downstream code
        # If some optional distances are missing, duplicate the last calculated one.
        while len(distances) < 6:
            distances.append(distances[-1] if distances else 0.0)

        (right_eyebrow_to_eye_distance,
         left_eyebrow_to_eye_distance,
         right_eyebrow_to_forehead_distance,
         left_eyebrow_to_forehead_distance,
         distance_between_eyebrows,
         distance_between_eyebrow_forehead) = distances[:6]

        return (right_eyebrow_to_eye_distance, left_eyebrow_to_eye_distance,
                right_eyebrow_to_forehead_distance, left_eyebrow_to_forehead_distance,
                distance_between_eyebrows, distance_between_eyebrow_forehead)

    def main(self, eyebrows_points: dict):
        # calculate eyebrow arch
        right_eyebrow_arch = self.arch_calculator.calculate_eyebrow_arch(eyebrows_points['right arch'])
        left_eyebrow_arch = self.arch_calculator.calculate_eyebrow_arch(eyebrows_points['left arch'])
        self.eyebrows['arch_right'] = right_eyebrow_arch
        self.eyebrows['arch_left'] = left_eyebrow_arch

        # calculate distance between eyebrow and its eye
        (right_eye_distance, left_eye_distance, right_forehead_distance, left_forehead_distance, eyebrows_distance,
         eyebrow_distance_forehead) = (self.calculate_distances(eyebrows_points))
        self.eyebrows['eye_right_distance'] = right_eye_distance
        self.eyebrows['eye_left_distance'] = left_eye_distance
        self.eyebrows['forehead_right_distance'] = right_forehead_distance
        self.eyebrows['forehead_left_distance'] = left_forehead_distance
        self.eyebrows['eyebrows_distance'] = eyebrows_distance
        self.eyebrows['eyebrow_distance_forehead'] = eyebrow_distance_forehead
        #print(f'Eyebrows: { {k: (round(float(v),4)) for k,v in self.eyebrows.items()}}')
        return self.eyebrows
