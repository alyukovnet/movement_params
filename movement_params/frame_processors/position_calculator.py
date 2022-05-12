import numpy as np
from cv2 import getPerspectiveTransform as Transform
from cv2 import aruco
from typing import Optional

from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame
from movement_params.CONFIG_STATIC import default_static as CFG


class LocationConverter:
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilateral in both planes
    Parameters
    ----------
    camera_matrix : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    world_matrix : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """

    def __init__(self, camera_matrix: np.ndarray, world_matrix: np.ndarray):
        assert camera_matrix.shape == (4, 2), "Expected array size (4,2) for camera matrix"
        assert world_matrix.shape == (4, 2), "Expected array size (4,2) for world matrix"
        self.world_matrix = np.array(world_matrix, dtype=np.float32)
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.transform_matrix = Transform(self.camera_matrix, self.world_matrix)
        self.inverse_transform_matrix = Transform(self.world_matrix, self.camera_matrix)

    def set_camera_matrix(self, new_matrix: np.ndarray) -> None:
        """
        Use the new camera matrix to recalculate transform matrices.
        Parameters
        ----------
        new_matrix : (4,2) shape numpy array
            The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
            pixels of the known region
        """
        assert new_matrix.shape == (4, 2), "Expected array size (4,2) for camera matrix"
        self.camera_matrix = np.array(new_matrix, dtype=np.float32)
        self.transform_matrix = Transform(self.camera_matrix, self.world_matrix)
        self.inverse_transform_matrix = Transform(self.world_matrix, self.camera_matrix)

    def camera_to_world(self, pixel: tuple[float, float]) -> tuple[float, float]:
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (lon, lat) tuple
            The corresponding (lon, lat) coordinates
        """
        pixel = np.array(pixel, dtype=np.float32).reshape(1, 2)
        assert pixel.shape[1] == 2, "Expected array size (N,2)"
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0], 1))], axis=1)
        coord = np.dot(self.transform_matrix, pixel.T)

        return tuple(np.squeeze((coord[:2, :] / coord[2, :]).T))

    def world_to_camera(self, coord: tuple[float, float]) -> tuple[float, float]:
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        coord : (lon, lat) tuple
            The (lon, lat) coordinates to be converted
        Returns
        -------
        (x, y) tuple
            The corresponding (x, y) pixel coordinates
        """
        coord = np.array(coord, dtype=np.float32).reshape(1, 2)
        assert coord.shape[1] == 2, "Need (N,2) input array"
        coord = np.concatenate([coord, np.ones((coord.shape[0], 1))], axis=1)
        pixel = np.dot(self.inverse_transform_matrix, coord.T)

        return tuple(np.squeeze((pixel[:2, :] / pixel[2, :]).T))


class PlaneFinder:
    """
    Create an object that searches for a plane of four points,
    where points are AruCo codes found on the screen.
    Parameters
    ----------
    ids : (4, ) shape numpy array
        4-element array containing 4 unique IDs of AruCo codes to look for
    """

    def __init__(self, ids: np.ndarray):
        assert ids.shape == (4, ), "Expected an array of 4 elements"
        assert np.unique(ids).shape == ids.shape, "Expected to have only unique values"
        self.ids: np.ndarray = ids
        self.plane: Optional[list[tuple[float, float]]] = None
        self.arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        self.arucoParams = aruco.DetectorParameters_create()

    def get_aruco_codes(self, image: np.ndarray) -> dict:
        """
        Get all AruCo codes that can be found on a given frame
        Parameters
        ----------
        image : numpy array
            array containing the image with alleged AruCo codes
        Returns
        -------
        An AruCo code location dictionary where
            key - AruCo code ID
            value - list of the AruCo code corners with points stored as (x, y) tuples
        """
        (corners, ids, rejected) = aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams)
        if corners is None or ids is None:
            return dict()
        corners = [corner.squeeze().tolist() for corner in corners]
        ids = np.array(ids).flatten().tolist()
        return dict(zip(ids, corners))

    def extract_plane(self, codes: dict) -> list[tuple[float, float]] or None:
        """
        Get the final plane by finding preemptively configured AruCo codes
        Parameters
        ----------
        codes : dictionary with the following structure
            key - AruCo code ID
            value - list of the AruCo code corners with points stored as (x, y) tuples
        Returns
        -------
        TO-DO: Describe output format
        """
        result = list()
        for identifier in self.ids:
            corners = codes.get(identifier, None)
            if corners is None:  # we lack some vital aruco code
                return None
            # Now let's get some middle point out of the array
            point = tuple(np.average(np.array(corners), axis=0))
            result.append(point)
        return result


class PositionCalculator(FrameProcessor):
    def __init__(self):
        self.default_matrix = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        camera_matrix = self.default_matrix
        world_matrix = CFG.world_matrix
        self.convert = LocationConverter(camera_matrix, world_matrix)
        ids = CFG.aruco_ids
        self.plane = self.default_matrix
        self.plane_finder = PlaneFinder(ids)

    def process(self, frame: Frame) -> Frame:
        # try to find and update camera plane:
        found_codes = self.plane_finder.get_aruco_codes(frame.image)
        plane = self.plane_finder.extract_plane(found_codes)
        if plane is not None and (self.plane != plane).all():
            plane = np.array(plane)
            self.plane = plane
            self.convert.set_camera_matrix(plane)
        # in case we haven't yet found camera matrix, do not perform conversions
        if (self.convert.camera_matrix == self.default_matrix).all():
            return frame
        # update positions
        for this_object in frame.objects:
            coordinates = self.convert.camera_to_world(this_object.box.center)
            this_object.set_world_pos(coordinates)

        return frame
