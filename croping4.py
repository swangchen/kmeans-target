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
        connected_pixels.append((col, row))

        # 添加相邻像素到堆栈
        stack.extend([(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])

# 读取图像
image = cv2.imread('2.jpg')

# 将图像转换为RGB格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像转换为一维数组，使用颜色值标准化到 [0, 1] 范围
pixels = image_rgb.reshape((-1, 3)).astype(np.float32) / 255.0

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

# 在原图上用红线框出裁剪区域
bounding_poly = cv2.convexHull(np.array(connected_pixels))
cv2.polylines(image_rgb, [bounding_poly], True, (255, 0, 0), 2)

# 显示原始图像、标注后的图像和中心附近封闭区间
cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imshow('Segmented Image', segmented_image)
cv2.imshow('Connected Pixels', visited_pixels.astype(np.uint8) * 255)
cv2.imshow('Original Image with Red Box', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)

# 保存标注后的图像
cv2.imwrite('segmented_image_with_center.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

# 保存Connected Pixels图像
cv2.imwrite('connected_pixels.jpg', visited_pixels.astype(np.uint8) * 255)

# 裁剪中心区域并保存
box = cv2.boxPoints(cv2.minAreaRect(np.array(connected_pixels)))
box = box.astype(np.float32)
transform_matrix = cv2.getPerspectiveTransform(box, np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32))
crop_image = cv2.warpPerspective(image_rgb, transform_matrix, (100, 100))
cv2.imwrite('cropped_center_region.jpg', cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))

# 保存带有红线框的原图
cv2.imwrite('original_image_with_red_box.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

cv2.destroyAllWindows()
