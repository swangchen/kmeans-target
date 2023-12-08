import cv2
import numpy as np

# 加载裁剪后的图像和原始图像
cropped_image = cv2.imread('cropped_center_region.jpg')
original_image = cv2.imread('2.jpg')

# 确保两个图像具有相同的大小
cropped_image = cv2.resize(cropped_image, (original_image.shape[1], original_image.shape[0]))

# 将裁剪后的图像转换为灰度图
gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# 通过对灰度图进行阈值处理创建二值掩码
_, mask = cv2.threshold(gray_cropped, 1, 255, cv2.THRESH_BINARY)

# 反转掩码
mask_inv = cv2.bitwise_not(mask)

# 将裁剪后的图像转换为4通道图像
cropped_alpha = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)

# 使用反转后的掩码设置裁剪图像的alpha通道
cropped_alpha[:, :, 3] = mask_inv

# 将原始图像转换为4通道图像，并使用掩码设置alpha通道
original_alpha = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
original_alpha[:, :, 3] = mask

# 使用带有alpha通道的裁剪图像减去带有alpha通道的原始图像
result_alpha = cv2.subtract(cropped_alpha, original_alpha)

# 提取没有alpha通道的结果
result = result_alpha[:, :, :3]

# 显示结果
cv2.imshow('结果', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
