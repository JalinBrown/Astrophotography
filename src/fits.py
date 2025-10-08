from image_io import ImageIO


class Fits:
    def __init__(self, path) -> None:
        self.__file_path: str = path
        self.hdul = ImageIO().load_fits(self.__file_path)
        self.header = self.hdul[0].header  # type: ignore
        self.data = self.hdul[0].data  # type: ignore
