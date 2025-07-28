import numpy as np
import scipy.io as io
from sklearn.preprocessing import StandardScaler, minmax_scale, normalize
from sklearn.decomposition import PCA

from model.emp import EMPBulider

def get_config(config_path):
    import json
    fp = open(config_path, 'r')
    return json.load(fp)


def preprocess(config, dataset_name):
    dataset = io.loadmat(config['dataset'][dataset_name]['img_path'])
    key = [k for k in dataset if k != '__version__' and k != '__header__' and k != '__globals__']
    pixels = dataset[key[0]]
    groundtruth = io.loadmat(config['dataset'][dataset_name]['gt_path'])
    key = [k for k in groundtruth if k != '__version__' and k != '__header__' and k != '__globals__']
    gt = groundtruth[key[0]]
    if dataset_name == 'Indian_pines':
        pixels = pixels[30:115, 24:94, :]
        gt = gt[30:115, 24:94]
    elif dataset_name == 'paviaU':
        pixels = pixels[150:350, 100:200, :]
        gt = gt[150:350, 100:200]

    rows, cols, bands = pixels.shape
    pca = PCA(n_components=config['num_pc'])
    pc = pca.fit_transform(pixels.reshape(-1, bands))
    pc = minmax_scale(pc)

    pc = np.reshape(pc, (rows, cols, -1))
    gt = np.reshape(gt, (rows, cols, -1))

    return pc, gt

# 创建滚动窗口视图
def rw(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.

    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.

    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.

    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],

            [[1, 2],
             [4, 5]]],


           [[[3, 4],
             [6, 7]],

            [[4, 5],
             [7, 8]]]])

    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])

    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)

    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])

    Or delay embedding (3D embedding with delay 2):
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1  # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)

# 提取HSI的空间-光谱特征
def get_HSI_patches_rw(x, gt, ksize, stride=(1, 1), padding='reflect', indix=False):
    """
    extract HSI spectral-spatial
    :param x: 3-D HSI (n_row, n_clm, n_band)
    :param gt: 2-D ground truth
    :param ksize: must be odd numbers, i.e. (3, 3) (7,7) ...
    :param stride:
    :param padding: padding mode: constant, reflect (default), etc.
    :return: (n_sample, ksize1, ksize2, n_band)
    """
    # # padding with boundary pixels
    new_height = np.ceil(x.shape[0] / stride[0])
    new_width = np.ceil(x.shape[1] / stride[1])
    pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
    pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]
    pad_top = int(pad_needed_height / 2)
    pad_down = int(pad_needed_height - pad_top)
    pad_left = int(pad_needed_width / 2)
    pad_right = int(pad_needed_width - pad_left)
    x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
    gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)
    n_row, n_clm, n_band = x.shape
    x = np.reshape(x, (n_row, n_clm, n_band))
    y = np.reshape(gt, (n_row, n_clm))
    ksizes_ = (ksize[0], ksize[1])
    x_patches = rw(x, ksizes_, axes=(1, 0))  # divide data into 5x5 blocks
    y_patches = rw(y, ksizes_, axes=(1, 0))
    i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[0] - 1) // 2)
    nonzero_index = y_patches[:, :, i_1, i_2].nonzero()
    x_patches_nonzero = x_patches[nonzero_index]
    y_patches_nonzero = (y_patches[:, :, i_1, i_2])[nonzero_index]
    x_patches_nonzero = np.transpose(x_patches_nonzero, [0, 2, 3, 1])
    if indix is True:
        return x_patches_nonzero, y_patches_nonzero, nonzero_index
    x_patches_2d = np.reshape(x_patches_nonzero, (x_patches_nonzero.shape[0], -1))
    x_patches_2d = normalize(x_patches_2d)
    return x_patches_2d


# class HSIDataset():
#     def __init__(self, config, pc, emp_image, gt, points) -> None:
#         self.pc = pc
#         self.emp_image = emp_image
#         self.gt = gt
#         self.points = points
#         self.size = config['size']      # 邻域大小
#         pad = config['size'] // 2         
#         self.pc = np.pad(self.pc, ((pad, pad), (pad, pad), (0, 0)), 'constant')

#     def __getitem__(self, index):
#         point = self.points[index]
#         pad = self.size // 2
#         row, col = point + pad
#         near = self.pc[row-pad:row+pad, col-pad:col+pad, :]
#         return self.emp_image[point[0], point[1], :], np.reshape(near, (-1)), self.gt[:, :, 0][point[0], point[1]] / 1.0 - 1

#     def __len__(self):
#         return self.points.shape[0]

class HSIDataset():
    # 初始化数据集
    def __init__(self, config, pc, emp_image, gt, points) -> None:
        self.size = config['size']      # 邻域大小
        self.pc = pc
        self.emp_image = emp_image
        self.spe_feat = get_HSI_patches_rw(pc, gt[:, :, 0], (self.size, self.size))
        self.gt = gt
        self.points = points

    # 返回给定索引的数据
    def __getitem__(self, index):
        point = self.points[index]
        return self.emp_image[point[0], point[1], :], self.spe_feat[index], self.gt[:, :, 0][point[0], point[1]] / 1.0 - 1

    # 返回数据集的长度
    def __len__(self):
        return self.points.shape[0]

# 分割数据集
def split_dataset(config, dataset_name, shuffle=False):
    pc, gt = preprocess(config, dataset_name)
    emp_image = EMPBulider(pc, gt, config['num_openings_closings']).build_emp()
    points = np.argwhere(gt[:, :, 0]!=0)                     # 把背景去掉，免得影响分类精度
    if shuffle:
        np.random.shuffle(points)
    data = HSIDataset(config, pc, emp_image, gt, points)          # 这个函数主要是是用来初始化一些变量的。在全数据集上测试
    return data, np.unique(gt).shape[0] - 1

# 主函数
if __name__ == '__main__':
    data, gt= split_dataset(get_config('./config.json'), 'Indian_pines')
    a, b, c = data[0]
    print(a.shape, b.shape, c.shape)