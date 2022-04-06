import unittest
from pathlib import Path

from movement_params.CONFIG import Config
from movement_params.frame import ObjectType
from movement_params.io import PhotoInput
from movement_params.frame_processors import ObjectsDetector


class ModelCarsDetectingTestCase(unittest.TestCase):
    def test_detecting(self):
        IMAGES_LIST = (Path(__file__).parent / 'images').iterdir()

        results = []

        for image_path in IMAGES_LIST:
            print(image_path)
            CONFIG = Config(
                input_type=PhotoInput(Path(image_path)),
                output_type=None,
                processors=[
                    ObjectsDetector()
                ],
            )
            frame = CONFIG.input_type.get_frame()
            for p in CONFIG.processors:
                frame = p.process(frame)
            objects = filter(lambda o: o.type == ObjectType.CAR, frame.objects)
            results.append(len(list(objects)))

        self.assertEqual(results, [1]*len(results))


if __name__ == '__main__':
    unittest.main()
