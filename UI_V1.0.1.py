import cv2
import numpy as np
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import time


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

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("预处理v1.0.1")

        self.root.geometry("1200x600")

        # 创建UI界面组件
        self.label = tk.Label(root, text="选择一张图片进行分割")
        self.label.pack(pady=10,anchor="w")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.browse_button = tk.Button(root, text="选择图片", command=self.browse_image)
        self.browse_button.pack(pady=10,anchor="w")

        self.display_segmented_image_button = tk.Button(root, text="曝光效果预览", command=self.display_segmented)
        self.display_segmented_image_button.pack(pady=10,anchor="w")

        self.display_connected_pixels_button = tk.Button(root, text="切割效果预览", command=self.display_connected_pixels)
        self.display_connected_pixels_button.pack(pady=10,anchor="w")

        # 初始化图像变量
        self.image = None

    def browse_image(self):
        file_path = filedialog.askopenfilename(initialdir="/", title="选择图片文件",
                                               filetypes=(("图像文件", "*.png;*.jpg;*.jpeg"), ("所有文件", "*.*")))
        if file_path:
            self.image_path = file_path
            # 更新图像变量
            self.image = Image.open(file_path)
            image = ImageTk.PhotoImage(self.image.resize((300, 300)))
            self.image_label.config(image=image)
            self.image_label.image = image
            self.label.config(text="已选择图片：" + file_path)

    def display_connected_pixels(self):
        if self.image:

            image_rgb, connected_pixels_image = self.get_connected_pixels_image(self.image)

            # 显示Connected Pixels图像
            connected_pixels_image = Image.fromarray(connected_pixels_image)
            connected_pixels_image = ImageTk.PhotoImage(connected_pixels_image.resize((300, 300)))
            self.image_label.config(image=connected_pixels_image)
            self.image_label.image = connected_pixels_image
        else:
            self.label.config(text="请先选择一张图片")

    def display_segmented(self):
        if self.image:

            image_rgb, segmented_image = self.get_segmented_image_image(self.image)

            segmented_image = Image.fromarray(segmented_image)
            segmented_image = ImageTk.PhotoImage(segmented_image.resize((300, 300)))
            self.image_label.config(image=segmented_image)
            self.image_label.image = segmented_image
        else:
            self.label.config(text="请先选择一张图片")

    def get_connected_pixels_image(self, image):
        # 你的图像分割代码
        # 这里只是一个示例，需要替换成你的实际代码
        image_rgb = np.array(image)
        pixels = image_rgb.reshape((-1, 3)).astype(np.float32) / 255.0
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(pixels)
        labels = kmeans.labels_
        labels_reshaped = labels.reshape(image_rgb.shape[:2])

        # 选择中心点
        image_center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
        center_label = labels_reshaped[image_center[1], image_center[0]]

        # 找到中心坐标附近同一类像素构成的封闭区间
        visited_pixels = np.zeros_like(labels_reshaped, dtype=bool)
        connected_pixels = []
        dfs(labels_reshaped, visited_pixels, image_center, center_label, connected_pixels)

        # 创建Connected Pixels图像
        connected_pixels_image = np.zeros_like(image_rgb)
        connected_pixels_image[visited_pixels] = [255, 0, 0]  # 红色


        return image_rgb, connected_pixels_image

    def get_segmented_image_image(self, image):
        # 你的图像分割代码
        # 这里只是一个示例，需要替换成你的实际代码
        image_rgb = np.array(image)
        pixels = image_rgb.reshape((-1, 3)).astype(np.float32) / 255.0
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(pixels)
        labels = kmeans.labels_
        labels_reshaped = labels.reshape(image_rgb.shape[:2])

        # 选择中心点
        image_center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
        center_label = labels_reshaped[image_center[1], image_center[0]]

        # 找到中心坐标附近同一类像素构成的封闭区间
        visited_pixels = np.zeros_like(labels_reshaped, dtype=bool)
        connected_pixels = []
        dfs(labels_reshaped, visited_pixels, image_center, center_label, connected_pixels)

        # 创建Connected Pixels图像
        connected_pixels_image = np.zeros_like(image_rgb)
        connected_pixels_image[visited_pixels] = [255, 0, 0]  # 红色
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

        return image_rgb, segmented_image



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
