import unittest

from PIL import Image

from experiments.tbs.image_transforms import LetterboxResize


class LetterboxResizeTests(unittest.TestCase):
    def test_preserves_aspect_ratio_and_centers_landscape_image(self):
        image = Image.new("RGB", (4, 2), (255, 0, 0))

        result = LetterboxResize(8, fill=(0, 0, 0))(image)

        self.assertEqual(result.size, (8, 8))
        self.assertEqual(result.getpixel((0, 1)), (0, 0, 0))
        self.assertEqual(result.getpixel((0, 2)), (255, 0, 0))
        self.assertEqual(result.getpixel((7, 5)), (255, 0, 0))
        self.assertEqual(result.getpixel((7, 6)), (0, 0, 0))

    def test_supports_rectangular_target_size(self):
        image = Image.new("RGB", (2, 4), (10, 20, 30))

        result = LetterboxResize((8, 6), fill=(1, 2, 3))(image)

        self.assertEqual(result.size, (8, 6))
        self.assertEqual(result.getpixel((1, 0)), (1, 2, 3))
        self.assertEqual(result.getpixel((2, 0)), (10, 20, 30))
        self.assertEqual(result.getpixel((4, 5)), (10, 20, 30))
        self.assertEqual(result.getpixel((5, 5)), (1, 2, 3))

    def test_border_median_fill_uses_source_background_color(self):
        image = Image.new("RGB", (4, 2), (240, 245, 250))
        image.putpixel((1, 1), (20, 30, 40))

        result = LetterboxResize(8, fill="border_median")(image)

        self.assertEqual(result.getpixel((0, 0)), (240, 245, 250))

    def test_rejects_invalid_size_and_fill(self):
        for size in (0, -1, (8,), (8, 0)):
            with self.subTest(size=size):
                with self.assertRaises((TypeError, ValueError)):
                    LetterboxResize(size)

        with self.assertRaises(ValueError):
            LetterboxResize(8, fill="white")


if __name__ == "__main__":
    unittest.main()
