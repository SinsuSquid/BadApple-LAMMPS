#ifdef FIX_CLASS
FixStyle(bad_apple, FixBadApple)
#else

#ifndef LMP_FIX_BAD_APPLE_H
#define LMP_FIX_BAD_APPLE_H

#include "fix.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace LAMMPS_NS {

class FixBadApple : public Fix {
 public:
  FixBadApple(class LAMMPS *, int, char **);
  virtual ~FixBadApple();
  int setmask() override;
  void init() override;
  void post_force(int) override;

 private:
  cv::VideoCapture cap;
  int every;                // Update frame every N timesteps
  double k_spring;          // Trapping well force constant
  int width, height;        // Grid resolution
  double x_scale, y_scale;  // Physical dimension scaling
  uchar threshold_val;

  // Cached frame buffers ✨
  cv::Mat binary;
  cv::Mat dist_map;

  bool load_next_frame();
};

} // namespace LAMMPS_NS

#endif
#endif
