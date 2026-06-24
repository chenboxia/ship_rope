"""
几何计算工具模块
提供端点提取、张紧度计算、角度计算、IoU计算、距离计算等基础几何运算。
"""
import numpy as np
from typing import Tuple, Optional


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    计算两个边界框的交并比(IoU)。
    box格式: [x1, y1, x2, y2]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_center(box: np.ndarray) -> Tuple[float, float]:
    """返回边界框中心点坐标。"""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点之间的欧氏距离。"""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def point_to_box_distance(point: Tuple[float, float], box: np.ndarray) -> float:
    """计算点到边界框边缘的最短距离。点在框内时返回0。"""
    cx, cy = point
    if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
        return 0.0
    dx = max(box[0] - cx, 0, cx - box[2])
    dy = max(box[1] - cy, 0, cy - box[3])
    return float(np.sqrt(dx * dx + dy * dy))


def extract_endpoints_from_roi(roi_mask: np.ndarray,
                                offset_x: int = 0,
                                offset_y: int = 0) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    从缆绳检测框内的二值掩码中提取缆绳主轴的两个端点。
    使用骨架细化 + 端点检测。
    返回: ((x1,y1), (x2,y2)) 或 None（无法提取时）
    """
    import cv2
    if roi_mask is None or np.sum(roi_mask) == 0:
        return None
    # 骨架提取
    skeleton = _thin_skeleton(roi_mask)
    # 找非零点坐标
    pts = np.column_stack(np.where(skeleton > 0))  # (row, col)
    if len(pts) < 2:
        return None
    # 转为 (x, y) 格式
    pts_xy = pts[:, ::-1].astype(float)
    # PCA找主轴方向，沿主轴取最远两点
    if len(pts_xy) < 3:
        p1 = tuple(pts_xy[0].astype(int) + [offset_x, offset_y])
        p2 = tuple(pts_xy[-1].astype(int) + [offset_x, offset_y])
        return (p1, p2)
    mean = np.mean(pts_xy, axis=0)
    centered = pts_xy - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    projections = centered @ principal
    idx_min = np.argmin(projections)
    idx_max = np.argmax(projections)
    p1 = tuple(pts_xy[idx_min].astype(int) + [offset_x, offset_y])
    p2 = tuple(pts_xy[idx_max].astype(int) + [offset_x, offset_y])
    return (p1, p2)


def extract_endpoints_from_mask(mask: np.ndarray,
                                 box: np.ndarray,
                                 expand_ratio: float = 0.05) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    从全图掩码中提取指定检测框内缆绳的两个端点。
    box: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box[:4].astype(int)
    h, w = mask.shape[:2]
    # 扩展一点区域
    dx = int((x2 - x1) * expand_ratio)
    dy = int((y2 - y1) * expand_ratio)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    roi = mask[y1:y2, x1:x2]
    return extract_endpoints_from_roi(roi, offset_x=x1, offset_y=y1)


def compute_rope_contour(image: np.ndarray,
                          box: np.ndarray,
                          method: str = "color") -> Optional[np.ndarray]:
    """
    在检测框内提取缆绳的轮廓/掩码。
    method: "color" 基于颜色阈值，"edge" 基于边缘检测。
    """
    import cv2
    x1, y1, x2, y2 = box[:4].astype(int)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    if method == "edge":
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(edges, kernel, iterations=1)
    else:
        # OTSU二值化
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def compute_tension_ratio(p1: Tuple[int, int],
                           p2: Tuple[int, int],
                           mask: np.ndarray,
                           offset_x: int = 0,
                           offset_y: int = 0) -> float:
    """
    计算缆绳张紧度（曲线伸直比）。
    张紧度 = 两端点直线距离 / 缆绳实际曲线长度
    接近1表示张紧，接近0表示松弛。
    """
    import cv2
    straight_dist = point_distance(p1, p2)
    if straight_dist < 1.0:
        return 0.0
    # 从掩码中计算曲线长度
    skeleton = _thin_skeleton(mask)
    # 计算骨架像素总数作为近似曲线长度
    curve_length = float(np.sum(skeleton > 0))
    if curve_length < 1.0:
        return 0.0
    return min(straight_dist / curve_length, 1.0)


def compute_rope_angle_at_endpoint(anchor_point: Tuple[int, int],
                                     other_point: Tuple[int, int],
                                     wall_normal: np.ndarray) -> float:
    """
    计算缆绳在系结锚点处的走向角度（与壁面法线的夹角，单位：度）。
    anchor_point: 壁面侧端点（系结锚点）
    other_point: 船侧端点
    wall_normal: 壁面法线方向向量（归一化）
    """
    rope_dir = np.array([other_point[0] - anchor_point[0],
                         other_point[1] - anchor_point[1]], dtype=float)
    rope_len = np.linalg.norm(rope_dir)
    if rope_len < 1.0:
        return 90.0
    rope_dir /= rope_len
    cos_angle = np.clip(np.dot(rope_dir, wall_normal), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def _thin_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """骨架细化（Zhang-Suen算法简化版）。"""
    import cv2
    img = (binary_mask > 0).astype(np.uint8) * 255
    # 使用OpenCV的形态学骨架提取
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = img.copy()
    while True:
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skeleton
