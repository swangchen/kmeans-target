import cv2
import numpy as np
from sklearn.cluster import KMeans

def dfs(image, visited, center, target_label, connected_pixels):
    stack = [(center[1], center[0])]

    while stack:
        row, col = stack.pop()

        if row < 0 or row >= image.shape[0] or col < 0 or col >= image.shape[1] or visited[row, col] or image[row, col] != target_label:
            continue

        visited[row, col] = True
        connected_pixels.append((col, row))  # 添加到Connected Pixels

        # 添加相邻像素到堆栈
        stack.extend([(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])

# 读取图像
image = cv2.imread('2.jpg')

# 将图像转换为RGB格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像转换为一维数组
pixels = image_rgb.reshape((-1, 3))

# 使用K均值聚类将像素分为5类
kmeans = KMeans(n_clusters=5)
kmeans.fit(pixels)

# 获取聚类的标签
labels = kmeans.labels_

# 将标签数组的形状调整为图像数组的形状
labels_reshaped = labels.reshape(image_rgb.shape[:2])

# 打印图像中心坐标和所属类别
image_center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
center_label = labels_reshaped[image_center[1], image_center[0]]

print("Center Coordinates:", image_center)
print("Class Label at Center:", center_label)

# 找到中心坐标附近同一类像素构成的封闭区间
visited_pixels = np.zeros_like(labels_reshaped, dtype=bool)
connected_pixels = []
dfs(labels_reshaped, visited_pixels, image_center, center_label, connected_pixels)

# 根据聚类标签重新构建图像
segmented_image = np.zeros_like(image_rgb)
for i in range(5):
    mask = labels_reshaped == i
    if i == center_label:
        # 标红中心坐标附近同一类像素构成的封闭区间
        segmented_image[mask] = [255, 0, 0]  # 红色
    else:
        segmented_image[mask] = image_rgb[mask]

# 显示原始图像、标注后的图像和中心附近封闭区间

# 保存标注后的图像
cv2.imwrite('segmented_image_with_center.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

# 保存Connected Pixels图像
cv2.imwrite('connected_pixels.jpg', visited_pixels.astype(np.uint8) * 255)

cv2.destroyAllWindows()

# 提取Connected Pixels中最上边缘和最左边缘的点
connected_pixels = np.array(connected_pixels)
min_x, min_y = np.min(connected_pixels, axis=0)

# 计算斜率
slope_x = (connected_pixels[:, 0].mean() - min_x) / (connected_pixels[:, 1].mean() - min_y)
slope_y = (connected_pixels[:, 1].mean() - min_y) / (connected_pixels[:, 0].mean() - min_x)

print("Slope X:", slope_x)
print("Slope Y:", slope_y)
