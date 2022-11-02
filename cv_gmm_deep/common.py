import numpy as np


def split_data(x, y, ratios):
    np.testing.assert_equal(len(x), len(y), "size of x and y must be equal")
    np.testing.assert_equal(np.sum(ratios), 1, "ratios should sum to 1")

    N = len(x)
    accessors = np.random.permutation(N)

    indices = np.floor(np.array(ratios) * N).cumsum().astype(np.int64)
    split_accessors = np.split(accessors, indices[:-1])

    return tuple((x[accessor], y[accessor]) for accessor in split_accessors)


def preprocess_data(x, y, features):
    x_new = np.empty((*x.shape[:-1], 0))
    y_new = y

    if len(y_new.shape) == 3:
        y_new = np.expand_dims(y_new, axis=3)

    if 'rgb' in features:
        x_new = np.append(x_new, x, axis=3)

    if 'DoG' in features:
        import cv2

        sigma = 1
        K = 3
        kernel_shape = (49, 49)

        temp = []

        for i in range(len(x_new)):
            G1 = cv2.GaussianBlur(x_new[i], ksize=kernel_shape, sigmaX=sigma)
            G2 = cv2.GaussianBlur(
                x_new[i], ksize=kernel_shape, sigmaX=K**2 * sigma)
            temp.append(G2 - G1)

        x_new = np.append(x_new, temp, axis=3)

    return x_new, y_new
