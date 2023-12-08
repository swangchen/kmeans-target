
import cv2
import numpy as np

# 读取原始图像和裁剪后的图像
image_original = cv2.imread('2.jpg')
image_cropped = cv2.imread('cropped_center_region.jpg')

# 转换为灰度图像
gray_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
gray_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

# 计算差异
diff = cv2.absdiff(gray_original, gray_cropped)

# 使用阈值将差异转换为二值图像
_, thresholded_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

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

# 计算差异区域的质心并画线
for contour in contours:
    # 计算轮廓的矩特征
    moments = cv2.moments(contour)

    # 避免除以零
    if moments['m00'] != 0:
        # 计算质心坐标
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # 打印质心坐标
        print("Centroid:", (cx, cy))

        # 在差异标记图像上画红线
        cv2.line(image_diff_marked, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 2)
        cv2.line(image_diff_marked, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 2)

# 显示并保存结果
cv2.imshow('Difference Marked', image_diff_marked)
cv2.imwrite('difference_marked_image_with_centers.jpg', image_diff_marked)
cv2.waitKey(0)
cv2.destroyAllWindows()

