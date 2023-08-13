import cv2
import numpy as np

def detect_lane_lines(frame):
    # 將影像轉換為灰度圖
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 進行高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 偵測邊緣
    edges = cv2.Canny(blurred, 50, 150)

    # 設定偵測車道線的 ROI 區域
    height, width = frame.shape[:2]
    center_x = width // 2

    left_vertices = np.array([[( 1*width/8, height), (center_x - 20, 11*height / 16), (center_x -20 , height )]], dtype=np.int32)
    right_vertices = np.array([[(center_x + 20, height ), (center_x + 20, 11*height / 16), ( 7*width/8 , height)]], dtype=np.int32)

    left_mask = np.zeros_like(edges)
    right_mask = np.zeros_like(edges)

    cv2.fillPoly(left_mask, left_vertices, 255)
    cv2.fillPoly(right_mask, right_vertices, 255)

    left_masked_edges = cv2.bitwise_and(edges, left_mask)
    right_masked_edges = cv2.bitwise_and(edges, right_mask)

    # 將edges和mask合併為一個影像
    combined_mask = cv2.add(left_masked_edges, right_masked_edges)
    ROI = cv2.add(left_mask, right_mask)

    # 顯示合併後的影像
    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("ROI", ROI)

    cv2.imshow("LEFT", left_masked_edges)
    cv2.imshow("RIGHT", right_masked_edges)

    # 進行霍夫變換偵測直線
    left_lines = cv2.HoughLinesP(left_masked_edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)
    right_lines = cv2.HoughLinesP(right_masked_edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)

    return left_lines, right_lines

def calculate_offset(frame, left_lines, right_lines):
    if left_lines is not None and right_lines is not None:
        # 計算左車道和右車道的中心點x座標
        left_center_x = np.mean([np.mean(left_line[0][0::2]) for left_line in left_lines])
        right_center_x = np.mean([np.mean(right_line[0][0::2]) for right_line in right_lines])

        # 計算影像中心點的x座標
        center_x = frame.shape[1] / 2

        # 計算車輛偏移量
        offset = (center_x - (left_center_x + right_center_x) / 2) / (frame.shape[1] / 2) * 100

        return offset

    return None

def main():
    video = cv2.VideoCapture("C:/Users/julia/Desktop/ypoutube/test1.mp4")

    count = 0
    danger_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0,0),fx=0.5, fy=0.5)

        # 偵測車道線
        left_lines, right_lines = detect_lane_lines(frame)

        if left_lines is not None:
            for line in left_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if right_lines is not None:
            for line in right_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # 計算車輛偏移量
        offset = calculate_offset(frame, left_lines, right_lines)

        if offset is not None:
            if offset < -7:
                cv2.rectangle(frame, (40, 20), (200, 70), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, "Too Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                count += 1

            elif offset > 7:
                cv2.rectangle(frame, (40, 20), (200, 70), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, "Too Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                count += 1

            if count > 30:
                cv2.rectangle(frame, (300, 260), (640, 310), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, "Dangerous driving", (330, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 增加 danger_count
                danger_count += 1
                if danger_count >= 5 * video.get(cv2.CAP_PROP_FPS):  # 顯示五秒
                    danger_count = 0  # 重置計數器
                    count = 0  # 重置偏移量計數器
       

        # 顯示處理後的影像
        cv2.imshow("Lane Detection", frame)

        key = cv2.waitKey(1)
        if key == 27:  # 按下ESC鍵結束迴圈
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
