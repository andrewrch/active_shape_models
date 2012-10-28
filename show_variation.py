#!/usr/bin/env python

from active_shape_models import *

def main():
  shapes = PointsReader.read_directory(sys.argv[1])
  print shapes[0].get_normal_to_point(0)
  a = ActiveShapeModel(shapes)
  ShapeViewer.show_modes_of_variation(a, int(sys.argv[2]))
  print "Finished"

if __name__ == "__main__":
  main()
