#!/usr/bin/env python

import sys
import os
import cv
import glob
import math
import numpy as np
from random import randint

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
    return '(%f, %f)' % (self.x, self.y)

  def dist(self, p):
    """ Return the distance of this point to another point

    :param p: The other point
    """
    return math.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

class Shape ( object ):
  """ Class to represent a shape.  This is essentially a list of Point
  objects
  """
  def __init__(self, pts = []):
    self.pts = pts
    self.num_pts = len(pts)

  def add_point(self, p):
    self.pts.append(p)
    self.num_pts += 1

  """ Helper methods for shape alignment """
  def __get_X(self, w):
    return sum([w[i]*pt.x for i,pt in enumerate(self.pts)])
  def __get_Y(self, w):
    return sum([w[i]*pt.y for i,pt in enumerate(self.pts)])
  def __get_Z(self, w):
    return sum([w[i]*(pt.x**2+pt.y**2) for i,pt in enumerate(self.pts)])
  def __get_C1(self, w, s):
    return sum([w[i]*(pt.x*self.pts[i].x+pt.y*self.pts[i].y) for \
        i,pt in enumerate(s.pts)])
  def __get_C2(self, w, s):
    return sum([w[i]*(pt.y*self.pts[i].x-pt.x*self.pts[i].y) for \
        i,pt in enumerate(s.pts)])

  def align_to_shape(self, s, w):
    """ Aligns this shape to the given shape using the weight matrix
    w.  This applies a scaling, transformation and rotation to
    each point in the shape to align it as closely as possible to the
    shape.

    This relies on some linear algebra which we use numpy to solve.

    [ X2 -Y2   W   0][ax]   [X1]
    [ Y2  X2   0   W][ay] = [Y1]
    [ Z    0  X2  Y2][tx]   [C1]
    [ 0    Z -Y2  X2][ty]   [C2]

    We want to solve this to find ax, ay, tx, and ty

    :param shape: The shape to align to
    :param w: The weight matrix
    :return s: The aligned shape
    """

    X1 = s.__get_X(w)
    X2 = self.__get_X(w)
    Y1 = s.__get_Y(w)
    Y2 = self.__get_Y(w)
    Z = self.__get_Z(w)
    W = sum(w)
    C1 = self.__get_C1(w, s)
    C2 = self.__get_C2(w, s)

    a = np.array([[ X2, -Y2,  W,  0],
                  [ Y2,  X2,  0,  W],
                  [  Z,   0, X2, Y2],
                  [  0,   Z,-Y2, X2]])

    b = np.array([X1, Y1, C1, C2])
    # Solve equations
    # result is [ax, ay, tx, ty]
    res = np.linalg.solve(a, b)

    print res

    new = Shape()
    # For each point in current shape
    for pt in self.pts:
      new_x = (res[0]*pt.x - res[1]*pt.y) - res[2]
      new_y = (res[1]*pt.x + res[0]*pt.y) - res[3]
      new.add_point(Point(new_x, new_y))
    return new

    # Now apply scaling/rotation/translation to the shape

class ShapeViewer ( object ):
  """ Provides functionality to display a shape in a window
  """
  @staticmethod
  def show_shapes(shapes):
    """ Function to show all of the shapes which are passed to it
    """
    cv.NamedWindow('a_window', cv.CV_WINDOW_AUTOSIZE)
    # Get size for the window
    max_x = int(max([pt.x for shape in shapes for pt in shape.pts]))
    max_y = int(max([pt.y for shape in shapes for pt in shape.pts]))

    i = cv.CreateImage((max_x+20, max_y+20), cv.IPL_DEPTH_8U, 3)
    cv.Set(i, 0)
    for shape in shapes:
      r = randint(0, 255)
      g = randint(0, 255)
      b = randint(0, 255)
      print "(%d, %d, %d)" % (r, g, b)
      for pt in shape.pts:
        cv.Circle(i, (int(pt.x), int(pt.y)), 1, (r, g, b))
    cv.ShowImage("a_window",i)
    cv.WaitKey()

class PointsReader ( object ):
  """ Class to read from files provided on Tim Cootes's website."""
  @staticmethod
  def read_points_file(filename):
    """ Read a .pts file, and returns a Shape object """
    s = Shape([])
    num_pts = 0
    with open(filename) as fh:
      # Get expected number of points from file
      first_line = fh.readline()
      if first_line.startswith("version"):
        # Then it is a newer type of file...
        num_pts = int(fh.readline().split()[1])
        # Drop the {
        fh.readline()
      else:
        # It is an older file...
        num_pts = int(first_line)
      for line in fh:
        if not line.startswith("}"):
          pt = line.strip().split()
          s.add_point(Point(float(pt[0]), float(pt[1])))
    if s.num_pts != num_pts:
      print "Unexpected number of points in file.  "\
      "Expecting %d, got %d" % (num_pts, s.num_pts)
    return s

  @staticmethod
  def read_directory(dirname):
    """ Reads an entire directory of .pts files and returns
    them as a list of shapes
    """
    pts = []
    for file in glob.glob(os.path.join(dirname, "*.pts")):
      pts.append(PointsReader.read_points_file(file))
    return pts

class ActiveShapeModel:
  """
  """
  def __init__(self, shapes = []):
    self.shapes = shapes
    # Make sure the shape list is valid
    self.__check_shapes(shapes)
    self.weights = self.__create_weight_matrix(shapes)

  def __check_shapes(self, shapes):
    """ Method to check that all shapes have the correct number of
    points """
    if shapes:
      num_pts = shapes[0].num_pts
      for shape in shapes:
        if shape.num_pts != num_pts:
          raise Exception("Shape has incorrect number of points")

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
    # Return empty matrix if no shapes
    if not shapes:
      return np.array()
    # First get number of points of each shape
    num_pts = shapes[0].num_pts

    # We need to find the distance of each point to each
    # other point in each shape.
    distances = np.zeros((len(shapes), num_pts, num_pts))
    for s, shape in enumerate(shapes):
      for k in range(num_pts):
        for l in range(num_pts):
          distances[s, k, l] = shape.pts[k].dist(shape.pts[l])

    # Create empty weight matrix
    w = np.zeros(num_pts)
    # calculate range for each point
    for k in range(num_pts):
      for l in range(num_pts):
        # Get the variance in distance of that point to other points
        # for all shapes
        w[k] += np.var(distances[:, k, l])
    # Invert weights
    return 1/w

def main():
  shapes = PointsReader.read_directory(sys.argv[1])
  a = ActiveShapeModel(shapes)
  new = shapes[0].align_to_shape(shapes[1], a.weights)
  #ShapeViewer.show_shapes([new, shapes[0]])
  #ShapeViewer.show_shapes([new, shapes[1]])
  ShapeViewer.show_shapes([new, shapes[0], shapes[1]])

  print "Finished"

if __name__ == "__main__":
  main()
