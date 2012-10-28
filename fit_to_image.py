#!/usr/bin/env python

from active_shape_models import *

def main():
  shapes = PointsReader.read_directory(sys.argv[1])
  a = ActiveShapeModel(shapes)
  # load the image
  i = cv.LoadImage(sys.argv[2])
  m = ModelFitter(a, i)
  ShapeViewer.draw_model_fitter(m)
  m.do_iteration()

if __name__ == "__main__":
  main()

