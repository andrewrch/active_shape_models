#!/usr/bin/env python

import sys
import os
import glob
import numpy as np

class PointsReader ( object ):
	""" Class to read from files provided on Tim Cootes's website."""
	
	@staticmethod
	def read_points_file(filename):
		""" Read a .pts file, and return a list of points """
		pts = list()
		num_pts = 0
		
		with open(filename) as fh:
			# Get expected number of points from file
			first_line = fh.readline()
			if first_line.startswith("version"):
				# Then it is a newer type of file...
				num_pts = int(fh.readline().split()[1])
				# Drop the {
				fh.readline()
				for i in range(num_pts):
					pts.append(fh.readline().split())
			else:
				# It is an older file...
				num_pts = int(first_line)
				# Read all of the points from the file
				for line in fh:
					pts.append(line.split())					
		if len(pts) != num_pts:
			print "Unexpected number of points in file.  "
			"Expecting %d, got %d" % (num_pts, len(pts))
		return pts
	
	@staticmethod
	def read_directory(dirname):
		""" Reads an entire directory of .pts files and returns
		them as a list (Of list of points)
		"""
		pts = list()
		for file in glob.glob(os.path.join(dirname, "*.pts")):
			pts.append(PointsReader.read_points_file(file))
		return pts
				
class ActiveShapeModel:
	"""
	"""
	def train():
		pass
	
	def createWeightMatrix(self, pts):
		pass
		

def main():
	asm = ActiveShapeModel()
	print "Finished"

if __name__ == "__main__":
    main()
