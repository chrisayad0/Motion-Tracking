import cv2
import numpy as np
import time
import math
import threading
import os
from queue import Queue
from tkinter import filedialog, Tk

def tracking(frame, bbox, tracker, initialized):
    x0, y0, x1, y1 = bbox
    ix, iy, iw, ih = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
    if not initialized:
        tracker.init(frame, (ix, iy, max(1, iw), max(1, ih)))
        return bbox, True
    ok, new_box = tracker.update(frame)
    if ok:
        tx, ty, tw, th = [int(v) for v in new_box]
        return [tx, ty, tx + tw, ty + th], True
    return bbox, False

def create_tracker():
    for attr in ['TrackerCSRT_create', 'TrackerCSRT']:
        if hasattr(cv2, attr):
            item = getattr(cv2, attr)
            return item.create() if hasattr(item, 'create') else item()
    return cv2.TrackerKCF_create()

def bg_feature_worker(input_queue, output_queue):
    orb = cv2.ORB_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    while True:
        task = input_queue.get()
        if task is None: break
        
        frame, template = task
        if template is None or frame is None:
            output_queue.put((None, False, None, None))
            continue
            
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(frame, None)
        
        if des1 is None or des2 is None:
            output_queue.put((None, False, None, None))
            continue

        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        debug_view = cv2.drawMatches(template, kp1, frame, kp2, matches[:50], None, flags=2)

        if len(matches) > 15:
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]])
            avg_x, avg_y = np.mean(dst_pts, axis=0)
            h, w = template.shape[:2]
            new_bbox = [int(avg_x - w/2), int(avg_y - h/2), int(avg_x + w/2), int(avg_y + h/2)]
            
            y1, y2 = max(0, new_bbox[1]), min(frame.shape[0], new_bbox[3])
            x1, x2 = max(0, new_bbox[0]), min(frame.shape[1], new_bbox[2])
            
            if (y2 - y1) > 5 and (x2 - x1) > 5:
                output_queue.put((new_bbox, True, frame[y1:y2, x1:x2].copy(), debug_view))
                continue
        
        output_queue.put((None, False, None, debug_view))

def draw_large_triangle(img, direction, center_x, center_y, size=100, color=(0, 0, 255)):
    half = size // 2
    if direction == "left":
        pts = np.array([[center_x - half, center_y], [center_x + half, center_y - half], [center_x + half, center_y + half]], np.int32)
    elif direction == "right":
        pts = np.array([[center_x + half, center_y], [center_x - half, center_y - half], [center_x - half, center_y + half]], np.int32)
    elif direction == "up":
        pts = np.array([[center_x, center_y - half], [center_x - half, center_y + half], [center_x + half, center_y + half]], np.int32)
    elif direction == "down":
        pts = np.array([[center_x, center_y + half], [center_x - half, center_y - half], [center_x + half, center_y - half]], np.int32)
    cv2.polylines(img, [pts], True, color, 5)

def start_soccer_tracker():
    root = Tk()
    root.withdraw()
    
    cap = cv2.VideoCapture(0)
    tracker = create_tracker()
    
    search_input_q = Queue(maxsize=1)
    search_output_q = Queue(maxsize=1)
    threading.Thread(target=bg_feature_worker, args=(search_input_q, search_output_q), daemon=True).start()

    initialized, selection, trail, last_pos = False, [], [], None
    cropping, trigger_init, searching_in_bg, reset_flag = False, False, False, False
    paused, last_debug_frame = False, np.zeros((300, 600, 3), dtype=np.uint8)
    saved_crop, last_time, frame_count = None, time.time(), 0
    
    init_diag = 0
    motion_m_s = 0.0
    alpha = 0.2

    def mouse_callback(event, x, y, flags, param):
        nonlocal selection, cropping, trigger_init, reset_flag
        if event == cv2.EVENT_RBUTTONDOWN: reset_flag = True
        elif not initialized or paused:
            if event == cv2.EVENT_LBUTTONDOWN: selection, cropping = [x, y, x, y], True
            elif event == cv2.EVENT_MOUSEMOVE and cropping: selection[2], selection[3] = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                selection[2], selection[3], cropping = x, y, False
                if abs(selection[2] - selection[0]) > 5: trigger_init = True

    cv2.namedWindow("Main Tracker")
    cv2.setMouseCallback("Main Tracker", mouse_callback)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        
        fh, fw = frame.shape[:2]
        display_frame = frame.copy()
        frame_count += 1

        if reset_flag:
            initialized, selection, trail, last_pos, trigger_init, searching_in_bg = False, [], [], None, False, False
            init_diag, motion_m_s = 0, 0.0
            tracker = create_tracker()
            while not search_output_q.empty(): search_output_q.get()
            reset_flag = False

        if trigger_init and (not initialized or paused) and len(selection) == 4:
            x0, y0, x1, y1 = selection
            ix, iy, iw, ih = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
            if iw > 5 and ih > 5:
                saved_crop = frame[iy:iy+ih, ix:ix+iw].copy()
                init_diag = math.sqrt(iw**2 + ih**2)
                tracker = create_tracker()
                _, initialized = tracking(frame, selection, tracker, False)
                paused = False
            trigger_init = False

        if initialized:
            selection, ok = tracking(frame, selection, tracker, True)
            if not search_output_q.empty():
                new_selection, found, new_template, debug_img = search_output_q.get()
                searching_in_bg = False
                if debug_img is not None: last_debug_frame = debug_img
                if found and not ok:
                    selection, saved_crop = new_selection, new_template
                    tracker = create_tracker()
                    tracking(frame, selection, tracker, False)
                    ok = True

            if ok:
                x0, y0, x1, y1 = selection
                w, h = abs(x1 - x0), abs(y1 - y0)
                mx, my = (x0 + x1) // 2, (y0 + y1) // 2
                
                curr_diag = math.sqrt(w**2 + h**2)
                curr_time = time.time()
                dt = curr_time - last_time
                if dt > 0 and init_diag > 0:
                    raw_delta = (curr_diag - init_diag) / (init_diag * dt)
                    motion_m_s = (alpha * raw_delta) + (1 - alpha) * motion_m_s
                last_time = curr_time

                # Triangle indicators
                t_size = 100
                t_color = (0, 0, 255)
                
                if mx < fw // 3:
                    draw_large_triangle(display_frame, "left", t_size//2 + 10, fh // 2, t_size, t_color)
                elif mx > 2 * fw // 3:
                    draw_large_triangle(display_frame, "right", fw - (t_size//2 + 10), fh // 2, t_size, t_color)
                
                if my < fh // 3:
                    draw_large_triangle(display_frame, "up", fw // 2, t_size//2 + 10, t_size, t_color)
                elif my > 2 * fh // 3:
                    draw_large_triangle(display_frame, "down", fw // 2, fh - (t_size//2 + 10), t_size, t_color)

                cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Motion: {motion_m_s:+.2f}", (x0, y0 - 10), 1, 1.2, (0, 255, 255), 2)

                trail.append((mx, my))
                for i in range(1, len(trail[-20:])):
                    cv2.line(display_frame, trail[-20:][i-1], trail[-20:][i], (0, 255, 255), 1)
                
                if frame_count % 30 == 0 and not searching_in_bg:
                    try: search_input_q.put_nowait((frame.copy(), saved_crop)); searching_in_bg = True
                    except: pass
            else:
                if not searching_in_bg:
                    try: search_input_q.put_nowait((frame.copy(), saved_crop)); searching_in_bg = True
                    except: pass
        
        msg = "PAUSED" if paused else "F: LOAD | P: PAUSE | R-CLICK: RESET"
        cv2.putText(display_frame, msg, (10, 30), 1, 1.0, (255, 255, 255), 2)
        if cropping: cv2.rectangle(display_frame, (selection[0], selection[1]), (selection[2], selection[3]), (255, 255, 0), 2)

        cv2.imshow("Main Tracker", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('f'):
            video_path = filedialog.askopenfilename(title="Select Video", filetypes=(("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")))
            if video_path:
                cap.release()
                cap = cv2.VideoCapture(video_path)
                reset_flag = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_soccer_tracker()
