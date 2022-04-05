import unittest

from movement_params.frame import BoundingBox, FrameObject, Frame


class BoundingBoxTestCase(unittest.TestCase):
    def test_bounding_box(self):
        box = BoundingBox(10, 20, 80, 60)

        self.assertEqual(box.h, 40)
        self.assertEqual(box.w, 70)
        self.assertEqual([*box], [10, 20, 80, 60])


if __name__ == '__main__':
    unittest.main()
