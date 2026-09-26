from statistics import median

from PIL import Image


def _validate_target_size(size):
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("size must be positive")
        return size, size

    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise TypeError("size must be a positive integer or a (width, height) pair")

    width, height = size
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("target width and height must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("target width and height must be positive")
    return width, height


def _validate_fill(fill):
    if fill == "border_median":
        return fill
    if not isinstance(fill, (tuple, list)) or len(fill) != 3:
        raise ValueError("fill must be 'border_median' or an RGB triplet")
    if any(
        not isinstance(channel, int) or not 0 <= channel <= 255
        for channel in fill
    ):
        raise ValueError("RGB fill channels must be integers between 0 and 255")
    return tuple(fill)


def _border_median_color(image):
    pixels = image.load()
    width, height = image.size
    border = []

    for x in range(width):
        border.append(pixels[x, 0])
        if height > 1:
            border.append(pixels[x, height - 1])
    for y in range(1, height - 1):
        border.append(pixels[0, y])
        if width > 1:
            border.append(pixels[width - 1, y])

    return tuple(
        int(round(median(pixel[channel] for pixel in border)))
        for channel in range(3)
    )


class LetterboxResize:
    """Resize an RGB image to fit inside a canvas without cropping."""

    def __init__(self, size, fill="border_median"):
        self.target_width, self.target_height = _validate_target_size(size)
        self.fill = _validate_fill(fill)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("LetterboxResize expects a PIL image")

        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("source image dimensions must be positive")

        scale = min(self.target_width / width, self.target_height / height)
        resized_width = max(1, min(self.target_width, int(round(width * scale))))
        resized_height = max(1, min(self.target_height, int(round(height * scale))))
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        resized = image.resize((resized_width, resized_height), resampling)

        fill = (
            _border_median_color(image)
            if self.fill == "border_median"
            else self.fill
        )
        canvas = Image.new("RGB", (self.target_width, self.target_height), fill)
        left = (self.target_width - resized_width) // 2
        top = (self.target_height - resized_height) // 2
        canvas.paste(resized, (left, top))
        return canvas

    def __repr__(self):
        size = (self.target_width, self.target_height)
        return f"{self.__class__.__name__}(size={size}, fill={self.fill!r})"
