import numpy as np
from cv2 import getPerspectiveTransform as Transform

from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame
from movement_params import CONFIG



class LocationConverter:
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
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
        camera = np.array(camera_matrix, dtype=np.float32)
        world = np.array(world_matrix, dtype=np.float32)
        self.M = Transform(camera, world)
        self.invM = Transform(world, camera)

    def camera_to_world(self, pixel: tuple[float, float]):
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
        coord = np.dot(self.M, pixel.T)

        return tuple(np.squeeze((coord[:2, :] / coord[2, :]).T))

    def world_to_camera(self, coord: tuple[float, float]):
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
        pixel = np.dot(self.invM, coord.T)

        return tuple(np.squeeze((pixel[:2, :] / pixel[2, :]).T))


class PositionCalculator(FrameProcessor):
    def __init__(self):
        camera_matrix = getattr(CONFIG, 'camera_matrix', np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
        world_matrix = getattr(CONFIG, 'world_matrix', np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
        self.convert: LocationConverter = LocationConverter(camera_matrix, world_matrix)

    def process(self, frame: Frame) -> Frame:
        for obj in frame.objects:
            coordinates = self.convert.camera_to_world(obj.box().center())
            obj.set_world_pos(coordinates)

        return frame
