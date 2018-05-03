




# algorithm 

def flow_track(frames, pool_cap):
    bbox = frames[0]
    pose = pose_detector(bbox, frames)

