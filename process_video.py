import cv2
import numpy as np
import os
import collections
from typing import Tuple, List, Optional, Dict, Any, Deque
from PIL import Image, ImageDraw, ImageFont
import logging
import sys
import time

def setup_logging():
    """Logging sistemini yapılandır"""
    logger = logging.getLogger('LaneDetector')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

class LaneDetector:
    def __init__(self):
        self.frame_width = None
        self.frame_height = None
        self.perspective_M = None
        self.perspective_Minv = None
        self.recent_left_fits: Deque[np.ndarray] = collections.deque(maxlen=10)
        self.recent_right_fits: Deque[np.ndarray] = collections.deque(maxlen=10)
        self.recent_deviation_angles: Deque[float] = collections.deque(maxlen=15)
        self.lane_detected = False
        self.detection_confidence = 0
        self.debug = False
        self.logger = setup_logging()
        self.frame_stats = {
            'processed_frames': 0,
            'lanes_detected': 0,
            'avg_confidence': 0,
            'avg_left_curve': 0,
            'avg_right_curve': 0,
            'avg_deviation': 0
        }
    
    def resize_frame_if_needed(self, frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
        """Yüksek çözünürlüklü videoları 480p'ye küçült"""
        height, width = frame.shape[:2]
        if width > max_width:
            new_width = 854
            new_height = 480
            frame = cv2.resize(frame, (new_width, new_height))
        
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        
        return frame
    
    def initialize_perspective_transform(self) -> None:
        """Kuşbakışı dönüşüm için perspektif matrislerini başlat"""
        if self.frame_width is None or self.frame_height is None:
            return
        
        src_points = np.float32([
            [self.frame_width * 0.43, self.frame_height * 0.65],
            [self.frame_width * 0.57, self.frame_height * 0.65],
            [self.frame_width * 0.95, self.frame_height],
            [self.frame_width * 0.05, self.frame_height]
        ])
        
        offset = 100 
        dst_points = np.float32([
            [offset, 0],
            [self.frame_width - offset, 0],
            [self.frame_width - offset, self.frame_height],
            [offset, self.frame_height]
        ])
        
        self.perspective_M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.perspective_Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    
    def warp_perspective(self, frame: np.ndarray) -> np.ndarray:
        """Görüntüyü kuşbakışı görünümüne dönüştür"""
        if self.perspective_M is None:
            self.initialize_perspective_transform()
        
        warped = cv2.warpPerspective(
            frame, self.perspective_M, 
            (self.frame_width, self.frame_height),
            flags=cv2.INTER_LINEAR
        )
        return warped
    
    def unwarp_perspective(self, warped: np.ndarray) -> np.ndarray:
        """Kuşbakışı görünümünden normal görünüme dönüştür"""
        if self.perspective_Minv is None:
            self.initialize_perspective_transform()
            
        unwarped = cv2.warpPerspective(
            warped, self.perspective_Minv, 
            (self.frame_width, self.frame_height),
            flags=cv2.INTER_LINEAR
        )
        return unwarped
    
    def adaptive_canny(self, gray: np.ndarray) -> np.ndarray:
        """Adaptif eşiklerle Canny kenar algılama"""
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        lower_thresh = max(0, mean_val - 0.33 * std_val)
        upper_thresh = min(255, mean_val + 1.33 * std_val)
        
        return cv2.Canny(gray, 50, 150)
    
    def create_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Alt yarı bölge için ROI maskesi oluştur"""
        height, width = frame.shape[:2]
        vertices = np.array([[
            [0, height],
            [0, int(height * 0.5)],
            [width, int(height * 0.5)],
            [width, height]
        ]], dtype=np.int32)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        return mask
    
    def threshold_image(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Renk ve gradient eşikleme ile şerit çizgilerini vurgula"""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        s_channel = hls[:,:,2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        max_val = np.max(abs_sobelx)
        if max_val == 0:
            scaled_sobelx = np.zeros_like(abs_sobelx, dtype=np.uint8)
        else:
            scaled_sobelx = np.uint8(255 * abs_sobelx / max_val)
        
        sx_binary = np.zeros_like(scaled_sobelx)
        sx_binary[(scaled_sobelx >= 20) & (scaled_sobelx <= 255)] = 1
        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 90) & (s_channel <= 255)] = 1
        
        combined_binary = np.zeros_like(sx_binary)
        combined_binary[(sx_binary == 1) | (s_binary == 1)] = 255
        
        return combined_binary, gray
    
    def detect_lines(self, masked_edges: np.ndarray) -> List[np.ndarray]:
        """Hough transform ile çizgileri tespit et"""
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=15,
            minLineLength=25,
            maxLineGap=150
        )
        return lines if lines is not None else []
    
    def classify_lines(self, lines: List[np.ndarray], frame_width: int) -> Tuple[List, List]:
        """Çizgileri eğime göre sol/sağ şerit olarak sınıflandır"""
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if np.sqrt((x2-x1)**2 + (y2-y1)**2) < 30:
                continue
                
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            if abs(slope) < 0.1 or abs(slope) > 10:
                continue
            
            mid_x = (x1 + x2) / 2
            
            if slope < 0 and mid_x < frame_width * 0.55:
                left_lines.append(line[0])
            elif slope > 0 and mid_x > frame_width * 0.45:
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    
    def fit_polynomial(self, binary_warped: np.ndarray, previous_fit=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Şerit çizgileri için polinom uydur"""
        if previous_fit is not None:
            prev_left, prev_right = previous_fit
        else:
            prev_left, prev_right = None, None
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        if previous_fit is None or self.detection_confidence < 3:
            left_fit, right_fit, out_img = self.find_lanes_histogram(binary_warped)
        else:
            margin = 100
            
            left_lane_inds = ((nonzerox > (previous_fit[0][0]*(nonzeroy**2) + previous_fit[0][1]*nonzeroy + previous_fit[0][2] - margin)) & 
                             (nonzerox < (previous_fit[0][0]*(nonzeroy**2) + previous_fit[0][1]*nonzeroy + previous_fit[0][2] + margin)))
            
            right_lane_inds = ((nonzerox > (previous_fit[1][0]*(nonzeroy**2) + previous_fit[1][1]*nonzeroy + previous_fit[1][2] - margin)) & 
                              (nonzerox < (previous_fit[1][0]*(nonzeroy**2) + previous_fit[1][1]*nonzeroy + previous_fit[1][2] + margin)))
            
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            
            if len(leftx) > 0:
                out_img[lefty, leftx] = [255, 0, 0]
            if len(rightx) > 0:
                out_img[righty, rightx] = [0, 0, 255]
            
            left_fit = None
            right_fit = None
            
            if len(leftx) > 500:
                left_fit = np.polyfit(lefty, leftx, 2)
            
            if len(rightx) > 500:
                right_fit = np.polyfit(righty, rightx, 2)
                
        return left_fit, right_fit, out_img
    
    def find_lanes_histogram(self, binary_warped: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Histogram tabanlı şerit tespiti"""
        bottom_half = binary_warped[binary_warped.shape[0]//2:,:]
        histogram = np.sum(bottom_half, axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        midpoint = int(histogram.shape[0]//2)
        
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 9
        window_height = int(binary_warped.shape[0]//nwindows)
        margin = 100
        minpix = 50
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except:
            pass
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        if len(leftx) > 0:
            out_img[lefty, leftx] = [255, 0, 0]
        if len(rightx) > 0:
            out_img[righty, rightx] = [0, 0, 255]
            
        left_fit = None
        right_fit = None
        
        if len(leftx) > 500:
            left_fit = np.polyfit(lefty, leftx, 2)
            
        if len(rightx) > 500:
            right_fit = np.polyfit(righty, rightx, 2)
        
        return left_fit, right_fit, out_img
    
    def validate_lanes(self, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray]) -> bool:
        """Tespit edilen şeritlerin uygunluğunu kontrol et"""
        if left_fit is None and right_fit is None:
            return False
            
        if left_fit is not None and right_fit is not None:
            y_eval_bottom = self.frame_height - 1
            y_eval_middle = self.frame_height // 2
            
            left_bottom_x = left_fit[0]*(y_eval_bottom**2) + left_fit[1]*y_eval_bottom + left_fit[2]
            right_bottom_x = right_fit[0]*(y_eval_bottom**2) + right_fit[1]*y_eval_bottom + right_fit[2]
            
            left_middle_x = left_fit[0]*(y_eval_middle**2) + left_fit[1]*y_eval_middle + left_fit[2]
            right_middle_x = right_fit[0]*(y_eval_middle**2) + right_fit[1]*y_eval_middle + right_fit[2]
            
            bottom_lane_width = right_bottom_x - left_bottom_x
            middle_lane_width = right_middle_x - left_middle_x
            
            min_lane_width = self.frame_width * 0.4
            max_lane_width = self.frame_width * 0.9
            
            if (bottom_lane_width < min_lane_width or bottom_lane_width > max_lane_width or
                middle_lane_width < min_lane_width or middle_lane_width > max_lane_width):
                return False
                
            width_ratio = abs(bottom_lane_width - middle_lane_width) / bottom_lane_width
            if width_ratio > 0.3:
                return False
            
        return True
    
    def smooth_lanes(self, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Son N fit'i kullanarak şeritleri yumuşatla"""
        if left_fit is not None:
            self.recent_left_fits.append(left_fit)
        
        if right_fit is not None:
            self.recent_right_fits.append(right_fit)
        
        smoothed_left = None
        if len(self.recent_left_fits) > 0:
            smoothed_left = np.mean(self.recent_left_fits, axis=0)
        
        smoothed_right = None
        if len(self.recent_right_fits) > 0:
            smoothed_right = np.mean(self.recent_right_fits, axis=0)
            
        return smoothed_left, smoothed_right
    
    def calculate_curvature(self, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray]) -> Tuple[float, float]:
        """Şerit eğriliklerini hesapla (metre cinsinden)"""
        ym_per_pix = 30/720
        xm_per_pix = 3.7/700
        
        y_eval = self.frame_height - 1
        
        left_curverad = float('inf')
        right_curverad = float('inf')
        
        if left_fit is not None:
            left_fit_cr = np.polyfit(
                np.array(range(self.frame_height))*ym_per_pix, 
                np.polyval(left_fit, np.array(range(self.frame_height)))*xm_per_pix, 
                2
            )
            
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        
        if right_fit is not None:
            right_fit_cr = np.polyfit(
                np.array(range(self.frame_height))*ym_per_pix, 
                np.polyval(right_fit, np.array(range(self.frame_height)))*xm_per_pix, 
                2
            )
            
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
            
        return left_curverad, right_curverad
    
    def calculate_deviation_angle(self, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray]) -> float:
        """Şerit ortası ile görüntü ortası arasındaki sapma açısını hesapla"""
        frame_center_x = self.frame_width // 2
        y_eval = self.frame_height - 1
        
        lane_center_x = None
        
        if left_fit is not None and right_fit is not None:
            left_x = np.polyval(left_fit, y_eval)
            right_x = np.polyval(right_fit, y_eval)
            lane_center_x = (left_x + right_x) / 2
            
        elif left_fit is not None:
            left_x = np.polyval(left_fit, y_eval)
            lane_center_x = left_x + (self.frame_width * 0.25)
            
        elif right_fit is not None:
            right_x = np.polyval(right_fit, y_eval)
            lane_center_x = right_x - (self.frame_width * 0.25)
        
        if lane_center_x is None:
            return 0.0
        
        horizontal_deviation = lane_center_x - frame_center_x
        
        camera_height = self.frame_height * 0.4
        focal_length = self.frame_width * 0.8 
        
        deviation_angle = np.degrees(np.arctan(horizontal_deviation / focal_length))
        
        self.recent_deviation_angles.append(deviation_angle)
        smoothed_angle = np.mean(self.recent_deviation_angles)
        
        return smoothed_angle
    
    def draw_lanes_on_warped(self, warped: np.ndarray, left_fit: Optional[np.ndarray], 
                            right_fit: Optional[np.ndarray]) -> np.ndarray:
        """Tespit edilen şeritleri kuşbakışı görünüme çiz"""
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        if left_fit is None and right_fit is None:
            return color_warp
        
        y_coords = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        
        left_x = None
        right_x = None
        
        if left_fit is not None:
            left_x = left_fit[0]*(y_coords**2) + left_fit[1]*y_coords + left_fit[2]
        
        if right_fit is not None:
            right_x = right_fit[0]*(y_coords**2) + right_fit[1]*y_coords + right_fit[2]
        
        if left_x is not None and right_x is not None:
            pts_left = np.array([np.transpose(np.vstack([left_x, y_coords]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, y_coords])))])
            pts = np.hstack((pts_left, pts_right))
            
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            
            cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), thickness=15)
            cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), thickness=15)
            
        elif left_x is not None:
            pts_left = np.array([np.transpose(np.vstack([left_x, y_coords]))])
            cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), thickness=15)
            
        elif right_x is not None:
            pts_right = np.array([np.transpose(np.vstack([right_x, y_coords]))])
            cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), thickness=15)
        
        return color_warp
    
    def draw_lanes_and_info(self, frame: np.ndarray, warped_lanes: np.ndarray, 
                           left_curve: float, right_curve: float, 
                           deviation_angle: float) -> np.ndarray:
        """Şeritleri ve bilgileri çerçeve üzerine çiz"""
        result = frame.copy()
        
        unwarped_lanes = self.unwarp_perspective(warped_lanes)
        result = cv2.addWeighted(result, 1, unwarped_lanes, 0.5, 0)
        
        panel_height = 110
        panel_margin = 10
        cv2.rectangle(result, (panel_margin, panel_margin), 
                     (self.frame_width - panel_margin, panel_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(result, (panel_margin, panel_margin), 
                     (self.frame_width - panel_margin, panel_height), 
                     (255, 255, 255), 2)
        
        pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        curve_text = f"Curvature: L {left_curve:.0f}m, R {right_curve:.0f}m"
        deviation_text = f"Lane Deviation: {deviation_angle:.1f}°"
        status_text = "Lane Status: Good" if self.detection_confidence >= 5 else "Lane Status: Unstable"
        
        draw.text((20, 25), curve_text, font=font, fill=(255, 255, 255))
        draw.text((20, 55), deviation_text, font=font, fill=(255, 255, 255))
        
        status_color = (0, 255, 0) if self.detection_confidence >= 5 else (255, 255, 0)
        draw.text((20, 85), status_text, font=font, fill=status_color)
        
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return result
    
    def log_frame_stats(self, left_curve: float, right_curve: float, deviation_angle: float):
        """Frame istatistiklerini logla ve güncelle"""
        self.frame_stats['processed_frames'] += 1
        
        if self.lane_detected:
            self.frame_stats['lanes_detected'] += 1
        
        n = self.frame_stats['processed_frames']
        self.frame_stats['avg_confidence'] = ((self.frame_stats['avg_confidence'] * (n-1)) + self.detection_confidence) / n
        self.frame_stats['avg_left_curve'] = ((self.frame_stats['avg_left_curve'] * (n-1)) + left_curve) / n
        self.frame_stats['avg_right_curve'] = ((self.frame_stats['avg_right_curve'] * (n-1)) + right_curve) / n
        self.frame_stats['avg_deviation'] = ((self.frame_stats['avg_deviation'] * (n-1)) + abs(deviation_angle)) / n
        
        if self.frame_stats['processed_frames'] % 30 == 0:
            detection_rate = (self.frame_stats['lanes_detected'] / self.frame_stats['processed_frames']) * 100
            
            self.logger.info(f"Frame: {self.frame_stats['processed_frames']} | "
                           f"Detection Rate: {detection_rate:.1f}% | "
                           f"Confidence: {self.frame_stats['avg_confidence']:.1f}/10 | "
                           f"Avg Curves: L{self.frame_stats['avg_left_curve']:.0f}m R{self.frame_stats['avg_right_curve']:.0f}m | "
                           f"Avg Deviation: {self.frame_stats['avg_deviation']:.1f}°")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Tek frame'i işle"""
        frame = self.resize_frame_if_needed(frame)
        
        combined_binary, gray = self.threshold_image(frame)
        
        mask = self.create_roi_mask(combined_binary)
        masked_binary = cv2.bitwise_and(combined_binary, mask)
        
        warped_binary = self.warp_perspective(masked_binary)
        
        previous_left_fit = None if not self.recent_left_fits else np.mean(self.recent_left_fits, axis=0)
        previous_right_fit = None if not self.recent_right_fits else np.mean(self.recent_right_fits, axis=0)
        previous_fit = (previous_left_fit, previous_right_fit) if previous_left_fit is not None or previous_right_fit is not None else None
        
        left_fit, right_fit, lane_debug = self.fit_polynomial(warped_binary, previous_fit)
        
        valid_lanes = self.validate_lanes(left_fit, right_fit)
        
        if valid_lanes:
            self.detection_confidence = min(10, self.detection_confidence + 1)
            self.lane_detected = True
        else:
            self.detection_confidence = max(0, self.detection_confidence - 1)
            if self.detection_confidence < 1:
                self.lane_detected = False
        
        if valid_lanes:
            smoothed_left, smoothed_right = self.smooth_lanes(left_fit, right_fit)
        else:
            smoothed_left = previous_left_fit
            smoothed_right = previous_right_fit
        
        left_curve, right_curve = self.calculate_curvature(smoothed_left, smoothed_right)
        deviation_angle = self.calculate_deviation_angle(smoothed_left, smoothed_right)
        
        self.log_frame_stats(left_curve, right_curve, deviation_angle)
        
        if self.frame_stats['processed_frames'] % 100 == 0:
            status = "GOOD" if self.detection_confidence >= 5 else "UNSTABLE"
            self.logger.info(f"Current: Curves L{left_curve:.0f}m R{right_curve:.0f}m | "
                           f"Deviation: {deviation_angle:.1f}° | Status: {status}")
        
        warped_lanes = self.draw_lanes_on_warped(warped_binary, smoothed_left, smoothed_right)
        
        result = self.draw_lanes_and_info(frame, warped_lanes, left_curve, right_curve, deviation_angle)
        
        if self.debug:
            h, w = result.shape[:2]
            debug_size = (w//4, h//4)
            
            warped_binary_colored = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR)
            warped_binary_resized = cv2.resize(warped_binary_colored, debug_size)
            result[h-debug_size[1]:h, 0:debug_size[0]] = warped_binary_resized
            
            lane_debug_resized = cv2.resize(lane_debug, debug_size) 
            result[h-debug_size[1]:h, debug_size[0]:debug_size[0]*2] = lane_debug_resized
        
        return result


def process_video(input_path: str, output_path: str) -> bool:
    """Video dosyasını işle"""
    logger = setup_logging()
    
    try:
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logger.error(f"Video açılamadı: {input_path}")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video işleme başlatılıyor...")
        logger.info(f"Dosya: {os.path.basename(input_path)}")
        logger.info(f"Süre: {duration:.1f}s | Toplam Frame: {total_frames} | FPS: {fps}")
        
        ret, first_frame = cap.read()
        if not ret:
            logger.error("İlk frame okunamadı")
            return False
        
        detector = LaneDetector()
        first_frame = detector.resize_frame_if_needed(first_frame)
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = detector.process_frame(frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Frame okunamadı, video bozuk/sonlandı")
                break
            
            out.write(processed_frame)
            
            frame_count += 1
            
            if frame_count % (fps * 10) == 0 or frame_count % (total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time * total_frames / frame_count
                remaining_time = estimated_total - elapsed_time
                
                logger.info(f"⏳ İlerleme: {progress:.1f}% ({frame_count}/{total_frames}) | "
                          f"Geçen: {elapsed_time:.1f}s | Kalan: {remaining_time:.1f}s")
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        logger.info(f"Video işleme tamamlandı!")
        logger.info(f"Toplam süre: {total_time:.1f}s | Ortalama FPS: {avg_fps:.1f}")
        logger.info(f"Çıktı: {output_path}")
        
        detection_rate = (detector.frame_stats['lanes_detected'] / detector.frame_stats['processed_frames']) * 100
        logger.info(f"Final İstatistikler:")
        logger.info(f"Şerit Tespit Oranı: {detection_rate:.1f}%")
        logger.info(f"Ortalama Eğrilik: L{detector.frame_stats['avg_left_curve']:.0f}m R{detector.frame_stats['avg_right_curve']:.0f}m")
        logger.info(f"Ortalama Sapma: {detector.frame_stats['avg_deviation']:.1f}°")
        
        return True
        
    except Exception as e:
        logger.error(f"Video işleme hatası: {str(e)}")
        return False


if __name__ == "__main__":
    input_video = "test_input.mp4"
    output_video = "test_output.mp4"
    
    if os.path.exists(input_video):
        success = process_video(input_video, output_video)
        print(f"İşlem {'başarılı' if success else 'başarısız'}")
    else:
        print(f"Test video bulunamadı: {input_video}")