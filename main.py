from utilities import vid_to_frames,frames_to_vid
from tracker import get_tracks,draw_annotations,interpolate_ball_positions,assign_ball_to_player,add_position_to_tracks,add_transformed_position_to_tracks
import cv2
from team_assigner import assign_team_color,get_player_team
import numpy as np
from camera_movement import get_camera_movement,add_adjust_positions_to_tracks
from speed_dist import add_speed_and_distance_to_tracks,draw_speed_and_distance

frames = vid_to_frames("data/match.mp4")

tracks = get_tracks(frames,read_from_stub=True,stub_path="stubs/track_file.pkl")

add_position_to_tracks(tracks)

camera_movement_per_frame = get_camera_movement(frames,read_from_stub=True,stub_path='stubs/camera_movement_stub.pkl')

add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
                                                                                
add_transformed_position_to_tracks(tracks)

tracks["ball"] = interpolate_ball_positions(tracks["ball"])
add_speed_and_distance_to_tracks(tracks)


#Team

team_colors,kmeans_ret = assign_team_color(frames[0], 
                                    tracks['players'][0])   
for frame_num, player_track in enumerate(tracks['players']):
    for player_id, track in player_track.items():
        team = get_player_team(frames[frame_num],   
                                                 track['bbox'],
                                                 player_id, kmeans_ret)
        tracks['players'][frame_num][player_id]['team'] = team 
        tracks['players'][frame_num][player_id]['team_color'] = team_colors[team]


#Possesion


team_ball_control= []
for frame_num, player_track in enumerate(tracks['players']):
    ball_bbox = tracks['ball'][frame_num][1]['bbox']
    assigned_player = assign_ball_to_player(player_track, ball_bbox)
    if assigned_player != -1:
        tracks['players'][frame_num][assigned_player]['has_ball'] = True
        team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
    else:
        team_ball_control.append(team_ball_control[-1])
team_ball_control= np.array(team_ball_control)




output_video_frames = draw_annotations(frames, tracks,team_ball_control)

out_vid_frames = draw_speed_and_distance(output_video_frames,tracks)

frames_to_vid(out_vid_frames,"output/output_vid.avi")