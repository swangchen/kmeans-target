import cv2
import numpy as np
from sklearn.cluster import KMeans

def kmeans_segmentation(image, n_clusters):
    pixels = image.reshape((-1, 3)).astype(np.float32) / 255.0
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    segmented_image = labels.reshape(image.shape[:2])
    return segmented_image

# 读取原始图像和裁剪后的图像
image_original = cv2.imread('2.jpg')
image_cropped = cv2.imread('cropped_center_region.jpg')

# 使用K均值聚类对两个图像进行分类
n_clusters = 8  # 根据实际情况调整类别数
segmented_original = kmeans_segmentation(image_original, n_clusters)
segmented_cropped = kmeans_segmentation(image_cropped, n_clusters)

# 比较类别之间的差异
diff = cv2.absdiff(segmented_original, segmented_cropped)

# 使用阈值将差异转换为二值图像
_, thresholded_diff = cv2.threshold(diff.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)

# 找到轮廓
contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 标记差异区域
image_diff_marked = image_original.copy()
cv2.drawContours(image_diff_marked, contours, -1, (0, 0, 255), 2)
# 保存结果图像
cv2.imwrite('difference_marked_image.jpg', image_diff_marked)

# 显示结果
cv2.imshow('Original Image', image_original)
cv2.imshow('Cropped Image', image_cropped)
cv2.imshow('Difference Marked', image_diff_marked)
cv2.waitKey(0)
cv2.destroyAllWindows()
