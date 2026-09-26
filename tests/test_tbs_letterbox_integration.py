import unittest

from PIL import Image
from torchvision import transforms

from experiments.tbs.dataset import build_transforms
from experiments.tbs.image_transforms import LetterboxResize


class LetterboxIntegrationTests(unittest.TestCase):
    def test_letterbox_mode_is_used_for_train_and_evaluation(self):
        train_transform, eval_transform = build_transforms(
            32,
            input_mode="letterbox",
        )

        self.assertIsInstance(train_transform.transforms[0], LetterboxResize)
        self.assertIsInstance(eval_transform.transforms[0], LetterboxResize)

        image = Image.new("RGB", (20, 10), (230, 240, 250))
        self.assertEqual(tuple(eval_transform(image).shape), (3, 32, 32))

    def test_crop_mode_remains_the_default(self):
        train_transform, eval_transform = build_transforms(32)

        self.assertIsInstance(
            train_transform.transforms[0],
            transforms.RandomResizedCrop,
        )
        self.assertIsInstance(eval_transform.transforms[0], transforms.Compose)

    def test_rejects_unknown_input_mode(self):
        with self.assertRaisesRegex(ValueError, "Unknown input_mode"):
            build_transforms(32, input_mode="stretch")


if __name__ == "__main__":
    unittest.main()
