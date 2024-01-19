import numpy as np
import matplotlib.pyplot as plt

# 计算变差函数
def compute_variation(image, l):
    # 获取图像的高度和宽度
    h, w = image.shape[1:]

    # 计算 x 方向的变差函数
    var_x = []
    for d in l:
        diff_x = image[:, :, :-d] - image[:, :, d:]
        diff_x_squared = diff_x ** 2
        var_x.append(np.mean(diff_x_squared))

    # 计算 y 方向的变差函数
    var_y = []
    for d in l:
        diff_y = image[:, :-d, :] - image[:, d:, :]
        diff_y_squared = diff_y ** 2
        var_y.append(np.mean(diff_y_squared))

    return var_x, var_y


# 计算两点相关性
def compute_correlation(image, l):
    # 获取图像的高度和宽度
    h, w = image.shape[1:]

    # 计算 x 方向上的两点相关性
    x_correlation = []
    for d in l:
        correlation = 0
        count = 0
        for i in range(h):
            for j in range(w - d):
                if image[0, i, j] == image[0, i, j + d]:
                    correlation += 1
                count += 1
        correlation /= count
        x_correlation.append(correlation)

    # 计算 y 方向上的两点相关性
    y_correlation = []
    for d in l:
        correlation = 0
        count = 0
        for i in range(h - d):
            for j in range(w):
                if image[0, i, j] == image[0, i + d, j]:
                    correlation += 1
                count += 1
        correlation /= count
        y_correlation.append(correlation)

    return x_correlation, y_correlation

# 假设图像为 image，维度为 [1, h, w]
image = np.random.randint(0, 2, size=(1, 100, 100))  # 替换成你的图像数据
l = range(1, 21)  # 采样的距离范围

x_correlation, y_correlation = compute_correlation(image, h)

# 绘制 x 方向的两点相关性曲线
plt.plot(h, x_correlation)
plt.xlabel('距离 d')
plt.ylabel('x 方向上的两点相关性')
plt.title('x 方向上的两点相关性曲线')
plt.show()

# 绘制 y 方向的两点相关性曲线
plt.plot(h, y_correlation)
plt.xlabel('距离 d')
plt.ylabel('y 方向上的两点相关性')
plt.title('y 方向上的两点相关性曲线')
plt.show()

# 假设图像为 image，维度为 [1, h, w]
image = np.random.rand(1, 100, 100)  # 替换成你的图像数据
l = range(1, 21)  # 采样的距离范围

var_x, var_y = compute_variation(image, h)

# 绘制 x 方向的变差函数曲线
plt.plot(h, var_x)
plt.xlabel('距离 h')
plt.ylabel('x 方向的变差函数')
plt.title('x 方向的变差函数曲线')
plt.show()

# 绘制 y 方向的变差函数曲线
plt.plot(h, var_y)
plt.xlabel('距离 h')
plt.ylabel('y 方向的变差函数')
plt.title('y 方向的变差函数曲线')
plt.show()