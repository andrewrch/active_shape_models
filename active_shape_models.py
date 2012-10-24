#!/usr/bin/env python

import sys
import os
import glob
import math
import numpy as np

class Point ( object ):
  """ Class to represent a point in 2d cartesian space """
  def __init__(self, x, y):
    self.x = x
    self.y = y
    
  def __add__(self, p):
    """ Return a new point which is equal to this point added to p
    :param p: The other point
    """
    return Point(self.x + p.x, self.y + p.y)
    
  def __repr__(self):
    """return a string representation of this point. """
    return '(%d, %d)' % (self.x, self.y)
    
  def dist(self, p):
    """ Return the distance of this point to another point
    
    :param p: The other point
    """
    return sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

class Shape ( object ):
  """ Class to represent a shape.  This is essentially a list of Point
  objects
  """
  def __init__(self, pts = list()):
    self.pts = pts
    self.num_pts = len(pts)
  
  def add_point(self, p):
    self.pts.append(p)
    self.num_pts += 1
  
  def align_shape(self, shape, w):
    """ Aligns this shape to the given shape using the weight matrix
    provided.
    
    :param shape: The shape to align to
    :param w: The weight matrix
    :return s: The aligned shape
    """
    pass

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
    
  def __procrustes(self, shapes):
    """ This function aligns all shapes passed as a parameter by using
    Procrustes analysis
    
    :param shapes: A list of Shape objects
    """
    pass
	
	def __create_weight_matrix(self, shapes):
    """ Private method to produce the weight matrix which corresponds
    to the training shapes
    
    :param shapes: A list of Shape objects
    :return w: The matrix of weights produced from the shapes
    """
		pass
		

def main():
	asm = ActiveShapeModel()
	print "Finished"

if __name__ == "__main__":
  main()
