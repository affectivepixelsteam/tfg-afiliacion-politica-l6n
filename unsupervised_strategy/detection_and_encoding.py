"""
	Face detection and encoding

	Our approach is based on visual information. 
	Run through the frames of a program, and detect as many faces as
	possible using MTCNN [1].
	For each detected face, encode its features as a vector
	embedding, thanks to the Facenet model [2].
	That way, each face, no matter from whom, available in a broadcast
	will be accesible as a rich latent representation.

	author: Ricardo Kleinlein
	date: 02/2020

	Usage:
		python detection_and_encoding.py <video-dir>

	Options:
		--encoding-model	Path to the encoding model pretrained
		--face-size	Minimal area of a face (bounding box around it)
		--save-bb	Save images of the bounding boxes
		--output-dir	Directory to save results in
		--quiet	Hide visual information
		-h, --help	Display script additional help
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	# Shut up TF
import PIL
import numpy as np
import pandas as pd
import tensorflow as tf

from os.path import join
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from arguments import FaceDetEncArgs


fix_coord = lambda x: 0 if x <0 else x


def update(program_info, frame_info, output_dir='results', save_bb=False):
	"""Append the information of the current frame to the
	program overall knowledge.

	Args:
		program_info (dict, list): Paths and features of each face
		frame_info (dict, list): Vectors and features of each face
		output_dir (str, optional): Output directory
		save_bb (bool, optional): Whether or not save a copy of the bbs

		Return:
			an updated version of the program_info
	"""
	init_item = len(program_info['img'])
	if len(frame_info['img']) > 0:
		for item in range(len(frame_info['img'])):
			program_info['size'].append(frame_info['size'][item])

			program_info['confidence'].append(frame_info['confidence'][item])

			img_path = join(
				output_dir, 'boundingbox', 'img_' + str(init_item) + '.png')
			if save_bb:
				program_info['img'].append(img_path)
				img = PIL.Image.fromarray(frame_info['img'][item])
				img.save(img_path)
			else:
				program_info['img'].append('not_saved')

			vector_path = join(
				output_dir, 'embedding', 'embedding_' + str(init_item))
			np.save(vector_path, frame_info['embedding'][item], allow_pickle=True)
			program_info['embedding'].append(vector_path + '.npy')

			init_item += 1

	return summary


def detect(frame, size_threshold, detection_model):
	"""MTCNN Face detection for an image.

	Args:
		frame (float): np.ndarray with the frame pixels
		size_threshold (int): min area of face in pixels
		detection_model (keras.Model): Detection model

	Return:
		dict of lists of the face images detected
		and their sizes and confidence in detection
	"""
	faces = detection_model.detect_faces(frame)
	frame_info = {'img': [], 'size': [], 'confidence': []}
	are_faces = True if len(faces) > 0 else False
	if are_faces:
		for face in faces:
			coord = face['box']	# [x0, y0, width, height]
			coord[0] = fix_coord(coord[0])
			coord[1] = fix_coord(coord[1])
			conf = face['confidence']
			face_size = coord[2] * coord[3]
			if face_size >= size_threshold:
				cropped_face = frame[coord[1]:(coord[1]+coord[3]),
					coord[0]:(coord[0]+coord[2])]
				cropped_face = PIL.Image.fromarray(cropped_face).resize((160, 160))
				cropped_face = np.asarray(cropped_face)

				frame_info['img'].append(cropped_face)
				frame_info['size'].append(face_size)
				frame_info['confidence'].append(conf)
				
	return frame_info


def _encode(face, encoding_model):
	# TODO: Check if this normalization makes sense
	mu = np.mean(face)
	std = np.std(face)
	norm_face = (face - mu) / std
	face_ = np.expand_dims(norm_face, axis=0)
	return encoding_model.predict(face_)[0]


def encode(faces, encoding_model):
	"""Encoding of the faces sucesfylly detected.

	Args:
		faces (float): dict of face features/properties
		encoding_model (keras.Model): Encoding model

	Return:
		a dict with the list of face encodings
	"""
	faces['embedding'] = []
	for face in faces['img']:
		face_vector = _encode(face.astype('float32'),
			encoding_model)
		faces['embedding'].append(face_vector)
	return faces


if __name__ == "__main__":
	args = FaceDetEncArgs().parse()

	frame_list = sorted(os.listdir(args.video_dir))
	detection_model = MTCNN()
	encoding_model = tf.compat.v1.keras.models.load_model(args.encoding_model)

	summary = {'img': [], 'size': [], 'confidence': [], 'embedding': []}
	os.makedirs(join(args.output_dir, 'embedding'), exist_ok=True)
	if args.save_bb:
		os.makedirs(join(args.output_dir, 'boundingbox'), exist_ok=True)

	for frame in tqdm(frame_list, disable=args.quiet):
		framepath = join(args.video_dir, frame)
		frame = np.asarray(PIL.Image.open(framepath).convert('RGB'))

		# Face detection
		faces = detect(frame, args.face_size, detection_model)
		
		# Face encoding 
		faces = encode(faces, encoding_model)

		
		summary = update(program_info=summary, 
			frame_info=faces,
			output_dir=args.output_dir,
			save_bb=args.save_bb)

	pd.DataFrame(summary).to_csv(join(
		args.output_dir, 'detection_and_encoding.csv'), index=None)
