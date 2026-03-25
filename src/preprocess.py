def sparse_flow( old_frame, frame ):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    print ( "good features found: ", p0.shape)
    #print (p0)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    p1, st, err =  cv.calcOpticalFlowPyrLK( old_frame, frame, p0, None, **lk_params  )
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(np.zeros_like(frame), 
                       (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 1, color[i].tolist(), -1)
        frame = cv.add(frame, mask)
    return np.uint8(frame)
