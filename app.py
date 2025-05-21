import cv2
import os
import json
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
import mediapipe as mp

mp_holistic = mp.solutions.holistic

def detect_activity(cur_lm, prev_lm, img_h, img_w):
    if cur_lm is None:
        return "unknown"
    L_WRIST = mp_holistic.PoseLandmark.LEFT_WRIST.value
    R_WRIST = mp_holistic.PoseLandmark.RIGHT_WRIST.value
    L_ANKLE = mp_holistic.PoseLandmark.LEFT_ANKLE.value
    R_ANKLE = mp_holistic.PoseLandmark.RIGHT_ANKLE.value
    NOSE    = mp_holistic.PoseLandmark.NOSE.value

    def speed(idx):
        if prev_lm is None or idx not in prev_lm or idx not in cur_lm:
            return 0
        x0, y0 = prev_lm[idx]
        x1, y1 = cur_lm[idx]
        return np.hypot(x1 - x0, y1 - y0)

    ws_l = speed(L_WRIST); ws_r = speed(R_WRIST)
    as_l = speed(L_ANKLE); as_r = speed(R_ANKLE)
    ws = (ws_l + ws_r) / 2
    as_ = (as_l + as_r) / 2

    hand_thresh    = img_h * 0.02
    ankle_thresh   = img_h * 0.015
    dance_thresh   = img_h * 0.05
    anomaly_thresh = img_h * 0.15

    # 1) anomaly
    if ws_l > anomaly_thresh or ws_r > anomaly_thresh or as_l > anomaly_thresh or as_r > anomaly_thresh:
        return "anomaly"

    # 2) hands up
    c0 = cur_lm.get(L_WRIST, (0,0))
    c1 = cur_lm.get(R_WRIST, (0,0))
    nose_y = cur_lm.get(NOSE,(img_w/2,img_h/2))[1]
    if c0[1] < nose_y and c1[1] < nose_y and abs(c0[0] - c1[0]) < img_w * 0.2:
        return "hands_up"

    # 3) walking
    if as_ > ankle_thresh and ws < hand_thresh:
        return "walking"
    # 4) moving hands
    if ws > hand_thresh and as_ < ankle_thresh:
        return "moving_hands"
    # 5) dancing
    if ws > dance_thresh and as_ > dance_thresh:
        return "dancing"
    # 6) stopped
    if (ws_l + ws_r + as_l + as_r) < (hand_thresh * 2):
        return "stopped"
    # 7) writing/typing
    avg_wr_y = (c0[1] + c1[1]) / 2
    if avg_wr_y > img_h * 0.6 and ws < hand_thresh * 1.5:
        return "writing_or_typing"
    # default
    return "moving"

def detect_emotions_and_activity(video_path, output_path,
                                 detector_backend='retinaface',
                                 detect_every_n=10):
    summary = {
        "video": os.path.basename(video_path),
        "total_frames": 0,
        "frames_processed": 0,
        "anomalies_detected": 0,
        "emotions": {},
        "activities": {}
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    summary["total_frames"] = total
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    holistic     = mp_holistic.Holistic(min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
    trackers     = []
    prev_pose_lm = None
    curr_em      = "unknown"
    curr_act     = "unknown"

    for fidx in tqdm(range(total), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break
        summary["frames_processed"] += 1

        if fidx % detect_every_n == 0:
            trackers.clear()

            # Emotion
            try:
                dets = DeepFace.analyze(frame,
                                         actions=['emotion'],
                                         detector_backend=detector_backend,
                                         enforce_detection=False)
            except Exception:
                dets = []
                summary["anomalies_detected"] += 1
            if isinstance(dets, dict):
                dets = [dets]

            # Pose → activity
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            cur_lm  = None
            if results.pose_landmarks:
                cur_lm = {idx:(lm.x*width, lm.y*height)
                          for idx,lm in enumerate(results.pose_landmarks.landmark)}

            curr_act = detect_activity(cur_lm, prev_pose_lm, height, width)
            if curr_act == "anomaly":
                summary["anomalies_detected"] += 1
            else:
                summary["activities"][curr_act] = summary["activities"].get(curr_act,0)+1
            prev_pose_lm = cur_lm

            # Pick first emotion and init trackers
            curr_em = "unknown"
            for face in dets:
                r = face.get('region',{})
                if all(k in r for k in ('x','y','w','h')):
                    x,y,w,h = map(int,(r['x'],r['y'],r['w'],r['h']))
                    emo     = face.get('dominant_emotion','unknown')
                    curr_em = emo
                    try:
                        tr = cv2.TrackerCSRT_create()
                    except AttributeError:
                        tr = cv2.legacy.TrackerCSRT_create()
                    tr.init(frame,(x,y,w,h))
                    trackers.append((tr, emo))
            summary["emotions"][curr_em] = summary["emotions"].get(curr_em,0)+1

        else:
            for tr, emo in trackers:
                ok, bbox = tr.update(frame)
                if not ok: continue
                x,y,w,h = map(int,bbox)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame, emo, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)

        out.write(frame)

    cap.release()
    out.release()
    holistic.close()
    cv2.destroyAllWindows()

    # save summary next to output video
    folder = os.path.dirname(output_path)
    summary_path = os.path.join(folder, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    inp = 'input_video.mp4'
    out = 'output_video.mp4'
    detect_emotions_and_activity(inp, out)
