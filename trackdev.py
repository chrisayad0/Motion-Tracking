import cv2
import numpy as np
import time
import math
import threading
from queue import Queue

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
    orb = cv2.ORB_create(nfeatures=1000)
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

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Generate debug view for the inference window
        debug_view = cv2.drawMatches(template, kp1, frame, kp2, matches[:20], None, flags=2)

        if len(matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                h, w = template.shape[:2]
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                x_coords = [p[0][0] for p in dst]
                y_coords = [p[0][1] for p in dst]
                new_bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                
                new_template = frame[max(0, new_bbox[1]):new_bbox[3], max(0, new_bbox[0]):new_bbox[2]].copy()
                output_queue.put((new_bbox, True, new_template, debug_view))
                continue
        
        output_queue.put((None, False, None, debug_view))

def show_splash():
    splash = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(15):
        splash[:] = 0
        cv2.putText(splash, "INITIALIZING SYSTEM", (140, 230), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
        cv2.circle(splash, (320, 300), 20 + (i * 2), (0, i * 15, 0), 2)
        cv2.imshow("Main Tracker", splash)
        cv2.waitKey(30)

def start_soccer_tracker():
    show_splash()
    cap = cv2.VideoCapture(0)
    tracker = create_tracker()
    
    search_input_q = Queue(maxsize=1)
    search_output_q = Queue(maxsize=1)
    threading.Thread(target=bg_feature_worker, args=(search_input_q, search_output_q), daemon=True).start()

    initialized, selection, trail, last_pos = False, [], [], None
    cropping, trigger_init, searching_in_bg, reset_flag = False, False, False, False
    paused, last_debug_frame = False, np.zeros((300, 600, 3), dtype=np.uint8)
    saved_crop, last_time, frame_count, velocity = None, time.time(), 0, 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal selection, cropping, trigger_init, reset_flag
        if event == cv2.EVENT_RBUTTONDOWN:
            reset_flag = True
        elif not initialized:
            if event == cv2.EVENT_LBUTTONDOWN:
                selection, cropping = [x, y, x, y], True
            elif event == cv2.EVENT_MOUSEMOVE and cropping:
                selection[2], selection[3] = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                selection[2], selection[3], cropping = x, y, False
                if abs(selection[2] - selection[0]) > 5: trigger_init = True

    cv2.namedWindow("Main Tracker")
    cv2.namedWindow("Inference Debug")
    cv2.setMouseCallback("Main Tracker", mouse_callback)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
        
        display_frame = frame.copy()
        frame_count += 1

        if reset_flag:
            initialized, selection, trail, last_pos, trigger_init, searching_in_bg = False, [], [], None, False, False
            tracker = create_tracker()
            while not search_output_q.empty(): search_output_q.get()
            reset_flag = False

        if trigger_init and not initialized and len(selection) == 4:
            x0, y0, x1, y1 = selection
            ix, iy, iw, ih = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
            if iw > 5 and ih > 5:
                saved_crop = frame[iy:iy+ih, ix:ix+iw].copy()
                _, initialized = tracking(frame, selection, tracker, False)
            trigger_init = False

        if initialized:
            selection, ok = tracking(frame, selection, tracker, True)
            
            if not search_output_q.empty():
                new_selection, found, new_template, debug_img = search_output_q.get()
                searching_in_bg = False
                if debug_img is not None: last_debug_frame = debug_img
                if found:
                    selection, saved_crop = new_selection, new_template
                    tracker = create_tracker()
                    tracking(frame, selection, tracker, False)
                    ok = True

            if ok:
                x0, y0, x1, y1 = selection
                center = (int((x0 + x1) / 2), int((y0 + y1) / 2))
                trail.append(center)
                
                curr_time = time.time()
                dt = curr_time - last_time
                if last_pos and dt > 0:
                    velocity = math.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2) / dt
                last_pos, last_time = center, curr_time

                cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Speed: {int(velocity)} px/s", (10, 30), 1, 1.5, (0, 255, 0), 2)
                
                for i in range(1, len(trail[-30:])):
                    cv2.line(display_frame, trail[-30:][i-1], trail[-30:][i], (0, 255, 255), 2)
                
                if frame_count % 30 == 0 and not searching_in_bg:
                    try: search_input_q.put_nowait((frame.copy(), saved_crop)); searching_in_bg = True
                    except: pass
            else:
                cv2.putText(display_frame, "LOST - SEARCHING...", (10, 60), 1, 1.5, (0, 0, 255), 2)
                if not searching_in_bg:
                    try: search_input_q.put_nowait((frame.copy(), saved_crop)); searching_in_bg = True
                    except: pass
        else:
            if cropping:
                cv2.rectangle(display_frame, (selection[0], selection[1]), (selection[2], selection[3]), (255, 255, 0), 2)
            msg = "PAUSED - SELECT TARGET" if paused else "L-CLICK: SELECT | R-CLICK: RESET | P: PAUSE"
            cv2.putText(display_frame, msg, (10, 30), 1, 1.2, (255, 255, 255), 2)

        cv2.imshow("Main Tracker", display_frame)
        cv2.imshow("Inference Debug", last_debug_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_soccer_tracker()
