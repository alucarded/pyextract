#!/usr/bin/env python

'''
    File name: autoextract.py
    Author: Tomasz Posluszny <tefposluszny@gmail.com>
    Date created: 3/17/2016
    Date last modified: 3/21/2016
    Python Version: 2.7.6
'''

import argparse
import cv2
import logging
import numpy as np
import os.path
import sys
import time

DEBUG = 0

class GrabCutExtraction:
	BG = 0
	FG = 1
	PR_FG = 3
	PR_BG = 2

	MAX_WIDTH = 480.0
	INIT_BG_STRIPE = 0.1

	def __init__(self, img, resize = False):
		self.in_img = img
		self.out_img = img.copy()
		self.resize = resize
		# Enforce max size for optimal processing time
		if resize:
			_, cols = self.out_img.shape[:2]
			factor = float(self.MAX_WIDTH)/float(cols)
			self.out_img = cv2.resize(self.out_img, (0,0), fx=factor, fy=factor)
		# Blur to reduce noise impact
		self.out_img = cv2.GaussianBlur(self.out_img, (15, 15), 0)
		#Initialize
		self.mask = np.zeros(self.out_img.shape[:2], dtype = np.uint8) # mask initialized to BG
		self.bgdmodel = np.zeros((1,65), np.float64)
		self.fgdmodel = np.zeros((1,65), np.float64)
		rows, cols = self.out_img.shape[:2]
		self.rect_init = (int(self.INIT_BG_STRIPE*cols), int(self.INIT_BG_STRIPE*rows),
			int((1 - 2*self.INIT_BG_STRIPE)*cols), int((1 - 2*self.INIT_BG_STRIPE)*rows))

	def process(self):
		cv2.grabCut(self.out_img, self.mask, self.rect_init, self.bgdmodel, self.fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
		if DEBUG == 1:
			cv2.imshow('Initial mask', self.mask)
			cv2.waitKey()

	def morph_open(self, mask, kernel_size, erode_iter, dilate_iter):
		kernel = np.ones((kernel_size, kernel_size), np.uint8)
		eroded = cv2.erode(mask, kernel, iterations = erode_iter)
		opened = cv2.dilate(eroded, kernel, iterations = dilate_iter)
		return eroded, opened

	def refine(self):
		fg_mask = np.where((self.mask == self.FG) + (self.mask == self.PR_FG), 255, 0).astype('uint8')
		# Get rid of noise using morphological opening
		_, fg_mask = self.morph_open(fg_mask, 3, 3, 3)
		# Refine foreground/background mask
		eroded, opened = self.morph_open(fg_mask, 3, 8, 15)
		if DEBUG == 1:
			cv2.imshow('Eroded', eroded)
			cv2.imshow('Opened', opened)
			cv2.waitKey()
		has_fg = False
		for i in xrange(self.out_img.shape[0]):
			for j in xrange(self.out_img.shape[1]):
				if eroded[i, j] == 255:
					self.mask[i, j] = self.FG
					has_fg = True
				elif opened[i, j] == 255:
					self.mask[i, j] = self.PR_BG
				else:
					self.mask[i, j] = self.BG
		# Terminate if we haven't detected any big enough object.
		if has_fg:
			cv2.grabCut(self.out_img, self.mask, self.rect_init, self.bgdmodel, self.fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
		else:
			logging.error("No foreground object.")
		if self.resize:
			factor = float(self.in_img.shape[1])/float(self.mask.shape[1])
			self.mask = cv2.resize(self.mask, (self.in_img.shape[1], self.in_img.shape[0]), interpolation = cv2.INTER_NEAREST)

	def fill_holes(self, bin_img):
		im_floodfill = bin_img.copy()
		# Mask used to flood filling.
		# Notice the size needs to be 2 pixels than the image.
		h, w = bin_img.shape[:2]
		fill_mask = np.zeros((h+2, w+2), np.uint8)

		# Floodfill from point (0, 0)
		cv2.floodFill(im_floodfill, fill_mask, (0,0), 255);

		# Invert floodfilled image
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		 
		# Combine the two images to get the foreground.
		out = bin_img | im_floodfill_inv
		if DEBUG == 1:
			cv2.imshow('Mask', out)
			cv2.waitKey()
		return out

	def write_output(self, directory, file_name):
		show_mask = np.where((self.mask == self.FG) + (self.mask == self.PR_FG), 255, 0).astype('uint8')
		# Smooth shape using morphological opening
		_, show_mask = self.morph_open(show_mask, 5, 20, 20)
		show_mask = self.fill_holes(show_mask)
		output = np.zeros(self.in_img.shape, np.uint8)
		output = cv2.bitwise_and(self.in_img, self.in_img, mask=show_mask)
		cv2.imwrite(os.path.join(directory, file_name), output)
		if DEBUG == 1:
			cv2.imshow('Output', output)
			cv2.waitKey()

if __name__ == '__main__':
	# Configure logging
	logging.basicConfig(level = logging.INFO)
	# Construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser(description="Extract object from image.")
	ap.add_argument("-i", "--input", required = True,
	    help = "Path to the directory with images")
	ap.add_argument("-o", "--output", default = "results", required = False,
	    help = "Path to the directory where output images should be placed")
	args = vars(ap.parse_args())
	in_dir = os.path.abspath(args["input"])
	out_dir = os.path.abspath(args["output"])

	# Output directory should be created by the program
	if os.path.isdir(out_dir):
	    logging.critical("Output directory %s already exists. Please (re)move it before proceeding.", out_dir)
	    sys.exit()
	else:
	    os.makedirs(out_dir)

	# Process image files in input directory
	in_files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
	for fn in in_files:
		logging.info("Processing %s...", os.path.basename(fn))
		img = cv2.imread(fn)
		if img is None:
			logging.error("Image %s could not be read.", fn)
			continue
		start = time.time()
		extraction = GrabCutExtraction(img, resize = True)
		extraction.process()
		extraction.refine()
		extraction.write_output(out_dir, os.path.basename(fn))
		end = time.time()
		logging.debug("Execution time: %d", end - start)
	
	
	
