"""
	author: Ricardo Kleinlein
	date: 02/2020

	Script to define the Arguments class. Every script will have its own 
	set of arguments as a rule, though some may be shared between tasks.
	These objects are not thought to be used independently, but simply 
	as a method to automate the argument passing between scripts in the 
	retrieval pipeline.
"""

import os
import argparse
import __main__ as main


class BaseArgs:
	def __init__(self):
		self.parser = argparse.ArgumentParser(
			description=__doc__)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument(
			"--output-dir",
			type=str,
			default="results",
			help="Directory to export the script s output to")
		self.parser.add_argument(
			"--quiet",
			action='store_true',
			help='Fewer information displayed on screen')

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.args = self.parser.parse_args()
		self._correct()

		if not self.args.quiet:
			print('-' * 10 + ' Arguments ' + '-' * 10)
			print('>>> Script: %s' % (os.path.basename(main.__file__)))
			print_args = vars(self.args)
			for key, val in sorted(print_args.items()):
				print('%s: %s' % (str(key), str(val)))
			print('-' * 30)
		return self.args

	def _correct(self):
		"""Assert ranges of params, mistypes..."""
		raise NotImplementedError


class SplitFramesArgs(BaseArgs):
	def initialize(self):
		BaseArgs.initialize(self)

		self.parser.add_argument(
			"video_path",
			type=str,
			default=None,
			help="Path to a video file")
		self.parser.add_argument(
			"--fps",
			type=int,
			default=1,
			help="Frames per second")
		self.parser.add_argument(
			"--frame_height",
			type=int,
			default=854,
			help="Height of the frames")
		self.parser.add_argument(
			"--frame_width",
			type=int,
			default=480,
			help="Width of the frames")

	def _correct(self):
		assert os.path.isfile(self.args.video_path)
		assert isinstance(self.args.fps, int)
		assert isinstance(self.args.frame_height, int)
		assert isinstance(self.args.frame_width, int)


class FaceDetEncArgs(BaseArgs):
	def initialize(self):
		BaseArgs.initialize(self)

		self.parser.add_argument(
			"video_dir",
			type=str,
			default=None,
			help="Path to the directory of frames of a video")
		self.parser.add_argument(
			"--encoding_model",
			type=str,
			default='facenet_keras.h5',
			help="Face encoding model [default: Keras Facenet")
		self.parser.add_argument(
			"--face-size",
			type=int,
			default=0,
			help="Min size (in pixel area) to keep a face at detection time [default: 0]")
		self.parser.add_argument(
			"--save-bb",
			action='store_true',
			help="Saves in memory the bounding boxes")

	def _correct(self):
		assert os.path.isdir(self.args.video_dir)
		self.args.output_dir = os.path.dirname(
			os.path.dirname(self.args.video_dir))


class DbscanArgs(BaseArgs):
	def initialize(self):
		BaseArgs.initialize(self)

		self.parser.add_argument(
			"program_csv",
			type=str,
			default=None,
			help="Path to the summary of the face detection")
		self.parser.add_argument(
			"--eps",
			type=float,
			default=None,
			help="Epsilon value [default: None]")
		self.parser.add_argument(
			"--min_samples",
			type=int,
			default=None,
			help="Min-samples value [default: None]")
		self.parser.add_argument(
			"--metric",
			type=str,
			default='euclidean',
			help="Distance metric in DBSCAN")
		self.parser.add_argument(
			"--min-face-size",
			type=int,
			default=0,
			help="Min area of face bounding box so the face is kept")
		self.parser.add_argument(
			"--nthneigh",
			type=int,
			default=2,
			help="N-th neighbors (including oneself) to compute distance to [default: 2]")

	def _correct(self):
		assert os.path.isfile(self.args.program_csv)
		self.args.output_dir = os.path.dirname(
			self.args.program_csv)
		if self.args.eps:
			assert isinstance(self.args.eps, float)
		if self.args.min_samples:
			assert isinstance(self.args.min_samples, int)
		assert isinstance(self.args.metric, str)
		dists = ['euclidean', 'cosine', 'manhattan', 'l1', 'l2']
		if self.args.metric not in dists:
			raise NotImplementedError('Pairwise distance not implemented')
		if self.args.min_face_size < 0:
			self.args.min_face_size = 0





