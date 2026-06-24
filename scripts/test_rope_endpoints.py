"""
绳子端点提取测试脚本
用法:
  python scripts/test_rope_endpoints.py --image path/to/rope_crop.jpg
  python scripts/test_rope_endpoints.py --image path/to/rope_crop.jpg --method all

输出: 在原图上标记两个端点（红色=端点1，蓝色=端点2），保存到同目录。
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def extract_endpoints_skeleton(roi):
    """骨架法：提取缆绳主轴两端"""
    # 骨架细化
    skeleton = np.zeros_like(roi)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = roi.copy()
    while True:
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    # 找骨架上的点
    pts = np.column_stack(np.where(skeleton > 0))
    if len(pts) < 2:
        return None, None, skeleton
    # PCA取主轴方向，沿主轴取最远两点
    pts_xy = pts[:, ::-1].astype(float)
    mean = np.mean(pts_xy, axis=0)
    centered = pts_xy - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    projections = centered @ principal
    idx_min = np.argmin(projections)
    idx_max = np.argmax(projections)
    p1 = tuple(pts_xy[idx_min].astype(int))
    p2 = tuple(pts_xy[idx_max].astype(int))
    return p1, p2, skeleton


def extract_endpoints_pca(roi):
    """PCA法：对亮像素做主成分分析取最远两点"""
    pts = np.column_stack(np.where(roi > 0))
    if len(pts) < 2:
        return None, None
    pts_xy = pts[:, ::-1].astype(float)
    mean = np.mean(pts_xy, axis=0)
    centered = pts_xy - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    projections = centered @ principal
    idx_min = np.argmin(projections)
    idx_max = np.argmax(projections)
    p1 = tuple(pts_xy[idx_min].astype(int))
    p2 = tuple(pts_xy[idx_max].astype(int))
    return p1, p2


def extract_endpoints_contour(roi):
    """轮廓法：找最大轮廓的最远点对"""
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2)
    if len(pts) < 2:
        return None, None
    # 找距离最远的两个点
    max_dist = 0
    best_pair = (tuple(pts[0]), tuple(pts[1]))
    for i in range(0, len(pts), max(1, len(pts)//50)):
        for j in range(i+1, len(pts), max(1, len(pts)//50)):
            d = np.linalg.norm(pts[i].astype(float) - pts[j].astype(float))
            if d > max_dist:
                max_dist = d
                best_pair = (tuple(pts[i]), tuple(pts[j]))
    return best_pair[0], best_pair[1]


def preprocess(image):
    """预处理：灰度化 + 二值化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, binary


def draw_endpoints(image, p1, p2, label=""):
    """在图上标记两个端点"""
    vis = image.copy()
    if p1 is not None:
        cv2.circle(vis, p1, 8, (0, 0, 255), -1)      # 红色
        cv2.putText(vis, "P1", (p1[0]+10, p1[1]-5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if p2 is not None:
        cv2.circle(vis, p2, 8, (255, 0, 0), -1)       # 蓝色
        cv2.putText(vis, "P2", (p2[0]+10, p2[1]-5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if p1 is not None and p2 is not None:
        cv2.line(vis, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)  # 绿色连线
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        cv2.putText(vis, "%.0fpx" % dist, (mid[0]+5, mid[1]-5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if label:
        cv2.putText(vis, label, (5, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Test rope endpoint extraction")
    parser.add_argument("--image", type=str, default="../outputs/111.jpg", help="Cropped rope image")
    parser.add_argument("--method", type=str, default="all",
                        choices=["skeleton", "pca", "contour", "all"])
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print("Error: cannot read image:", args.image)
        return
    gray, binary = preprocess(image)

    output_dir = os.path.dirname(args.image) or "."
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    results = []
    methods = ["skeleton", "pca", "contour"] if args.method == "all" else [args.method]

    for method in methods:
        if method == "skeleton":
            p1, p2, skeleton = extract_endpoints_skeleton(binary)
            vis = draw_endpoints(image, p1, p2, "SKELETON")
            cv2.imwrite(os.path.join(output_dir, base_name + "_skeleton.jpg"), vis)
            # 也保存骨架图
            cv2.imwrite(os.path.join(output_dir, base_name + "_skeleton_mask.jpg"), skeleton)
        elif method == "pca":
            p1, p2 = extract_endpoints_pca(binary)
            vis = draw_endpoints(image, p1, p2, "PCA")
            cv2.imwrite(os.path.join(output_dir, base_name + "_pca.jpg"), vis)
        elif method == "contour":
            p1, p2 = extract_endpoints_contour(binary)
            vis = draw_endpoints(image, p1, p2, "CONTOUR")
            cv2.imwrite(os.path.join(output_dir, base_name + "_contour.jpg"), vis)

        if p1 and p2:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            results.append((method, p1, p2, dist))
            print("%s: P1=%s P2=%s dist=%.0fpx" % (method, p1, p2, dist))
        else:
            print("%s: FAILED - could not extract endpoints" % method)

    # 保存二值化图
    cv2.imwrite(os.path.join(output_dir, base_name + "_binary.jpg"), binary)
    print("\nResults saved to:", output_dir)
    print("Files: *_skeleton.jpg, *_pca.jpg, *_contour.jpg, *_binary.jpg")

if __name__ == "__main__":
    main()
