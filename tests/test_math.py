from featurelib.math import squared


class TestSquared:
    def test_square_positive(self):
        assert squared(4) == 16

    def test_square_negative(self):
        assert squared(-4) == 16
