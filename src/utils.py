def uniform_indices(num_frames, T):
    if num_frames <= 0:
        return []
    idx = np.linspace(0, num_frames-1, T).astype(int)
    return idx.tolist()

def read_clip_cv2(path, T=16, size=112):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = uniform_indices(n, T)
    frames = []
    for j, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            # fall back: duplicate last good frame or zeros
            frame = frames[-1] if frames else np.zeros((size, size, 3), np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    clip = np.stack(frames, axis=0).astype("float32") / 255.0  # (T,H,W,C)
    return clip

def pseudo_label_day_night(path, sample_frames=24, thresh=0.45):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = uniform_indices(n, sample_frames)
    vals = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok: continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        vals.append(g.mean()/255.0)
    cap.release()
    if not vals: return 0
    mean_brightness = float(np.mean(vals))
    return 1 if mean_brightness < thresh else 0
