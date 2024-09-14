from moviepy.editor import VideoFileClip, concatenate_videoclips
clip1 = VideoFileClip("/media/dell/data/比赛/提交/a.MOV")
clip2 = VideoFileClip("/media/dell/data/比赛/提交/b.MOV")
clip3 = VideoFileClip("/media/dell/data/比赛/提交/c.MOV")
final_clip = concatenate_videoclips([clip1,clip2,clip3])
final_clip.write_videofile("my_concatenation.mp4")