"""单应矩阵求解与应用的回归测试。"""


from src.trajectory.homography import apply_homography, compute_homography, parse_point_list


def test_parse_point_list():
    points = parse_point_list("0,0;10,5;20,10")
    assert points == [[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]]


def test_compute_homography_from_four_points():
    world = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    # 已知变换：像素 = 10 * 世界坐标
    pixels = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]

    matrix = compute_homography(pixels, world)

    assert matrix.shape == (3, 3)
    x, y = apply_homography(matrix, 100.0, 0.0)
    assert abs(x - 10.0) < 1e-5
    assert abs(y - 0.0) < 1e-5

    x, y = apply_homography(matrix, 50.0, 50.0)
    assert abs(x - 5.0) < 1e-5
    assert abs(y - 5.0) < 1e-5


def test_compute_homography_rejects_too_few_points():
    import pytest

    with pytest.raises(ValueError):
        compute_homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])
