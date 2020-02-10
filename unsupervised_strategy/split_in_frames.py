"""
	Downsample videos as a sequence of frames

	Broadcast videos usually have a high rate of frames per second (FPS).
	Working on them all would result in a high computational expense,
	and long processing times.
	As a trade-off between performance and expense, we split the original
	recorded audiovisual material in frames. In other words, 
	we downsample the original programs.

	author: Ricardo Kleinlein
	date: 02/2020

	Usage:
		python split_in_frames.py <path-to-video>

	Options:
		--fps  Set the FPS ratio of the outcome
		--frame_height	Set the height of the frames
		--width_height	Set the width of the frames
		--output-dir	Directory to save results in
		--quiet	Hide visual information
		-h, --help	Display script additional help
"""


import os
from os.path import join, isdir
from arguments import SplitFramesArgs

if __name__ == "__main__":
	args = SplitFramesArgs().parse()
	video_path = args.video_path
	video_name = os.path.basename(video_path).split('.')[0]

	if not isdir(join(args.output_dir, video_name)):
		os.makedirs(join(args.output_dir, video_name, 'frames'), exist_ok=True)

	cmd = 'ffmpeg -i ' + video_path
	cmd += ' -vf fps=' + str(args.fps)
	cmd += ' -s ' + str(args.frame_height)
	cmd += 'x' + str(args.frame_width) + ' '
	cmd += join(args.output_dir, video_name, 'frames', video_name + '_%05d.png')
	if args.quiet:
		cmd += ' -hide_banner -loglevel panic'
	os.system(cmd)