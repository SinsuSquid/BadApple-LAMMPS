#include "fix_bad_apple.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "update.h"
#include <cmath>
#include <algorithm>

using namespace LAMMPS_NS;
using namespace FixConst;

FixBadApple::FixBadApple(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  if (narg < 8) error->all(FLERR, "Illegal fix bad_apple command: syntax is 'fix ID group-ID bad_apple video.mp4 N_every k_spring width height'");

  std::string video_file = arg[3];
  every = utils::inumeric(FLERR, arg[4], false, lmp);
  k_spring = utils::numeric(FLERR, arg[5], false, lmp);
  width = utils::inumeric(FLERR, arg[6], false, lmp);
  height = utils::inumeric(FLERR, arg[7], false, lmp);
  threshold_val = 128;

  cap.open(video_file);
  if (!cap.isOpened()) {
    error->all(FLERR, "fix bad_apple: Could not open input video file!");
  }

  x_scale = domain->boxhi[0] / width;
  y_scale = domain->boxhi[1] / height;
}

FixBadApple::~FixBadApple() {
  if (cap.isOpened()) cap.release();
}

int FixBadApple::setmask() {
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

void FixBadApple::init() {
  load_next_frame();
}

bool FixBadApple::load_next_frame() {
  cv::Mat frame, gray, resized;
  if (!cap.read(frame) || frame.empty()) return false;

  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::resize(gray, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
  cv::threshold(resized, binary, threshold_val, 255, cv::THRESH_BINARY);

  // Inverted binary for distanceTransform:
  // distanceTransform calculates distance to zero pixels.
  // So invert binary so active silhouette = 0 (zero distance inside well), background = 255
  cv::Mat inv_binary;
  cv::bitwise_not(binary, inv_binary);
  cv::distanceTransform(inv_binary, dist_map, cv::DIST_L2, 3);

  return true;
}

void FixBadApple::post_force(int /*vflag*/) {
  if (update->ntimestep % every == 0) {
    if (!load_next_frame()) {
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      load_next_frame();
    }
  }

  if (binary.empty() || dist_map.empty()) return;

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; ++i) {
    if (mask[i] & groupbit) {
      int c = std::min(width - 1, std::max(0, static_cast<int>(x[i][0] / x_scale)));
      int r = height - 1 - std::min(height - 1, std::max(0, static_cast<int>(x[i][1] / y_scale)));

      // If atom is outside the white silhouette (on a black pixel)
      if (binary.at<uchar>(r, c) == 0) {
        float d = dist_map.at<float>(r, c);

        float d_right = (c + 1 < width) ? dist_map.at<float>(r, c + 1) : d;
        float d_left  = (c - 1 >= 0)     ? dist_map.at<float>(r, c - 1) : d;
        float d_up    = (r - 1 >= 0)     ? dist_map.at<float>(r - 1, c) : d;
        float d_down  = (r + 1 < height) ? dist_map.at<float>(r + 1, c) : d;

        // Force points towards decreasing distance (back into the potential well!)
        double grad_x = (d_right - d_left) / (2.0 * x_scale);
        double grad_y = -(d_down - d_up)   / (2.0 * y_scale);

        f[i][0] += -k_spring * grad_x;
        f[i][1] += -k_spring * grad_y;
      }

      // Strong planar pin to Z=0
      f[i][2] += -100.0 * x[i][2];
    }
  }
}
