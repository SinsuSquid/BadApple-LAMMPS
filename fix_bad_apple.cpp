#include "fix_bad_apple.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "update.h"
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
   Syntax in LAMMPS script:
   fix ID group-ID bad_apple video.mp4 N_every k_spring width height
   Example:
   fix 1 all bad_apple bad_apple.mp4 100 10.0 480 360
------------------------------------------------------------------------- */

FixBadApple::FixBadApple(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  if (narg < 8) error->all(FLERR, "Illegal fix bad_apple command: syntax is 'fix ID group-ID bad_apple video.mp4 N_every k_spring width height'");

  std::string video_file = arg[3];
  every = utils::inumeric(FLERR, arg[4], false, lmp);
  k_spring = utils::numeric(FLERR, arg[5], false, lmp);
  width = utils::inumeric(FLERR, arg[6], false, lmp);
  height = utils::inumeric(FLERR, arg[7], false, lmp);
  threshold_val = 128;

  // Open OpenCV video stream on Rank 0 (or all ranks)
  cap.open(video_file);
  if (!cap.isOpened()) {
    error->all(FLERR, "fix bad_apple: Could not open input video file!");
  }

  // Set physical domain mapping scale
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
  // Load initial frame
  load_next_frame();
}

bool FixBadApple::load_next_frame() {
  cv::Mat frame, gray, resized, binary;
  if (!cap.read(frame) || frame.empty()) return false;

  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::resize(gray, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
  cv::threshold(resized, binary, threshold_val, 255, cv::THRESH_BINARY);

  target_coords.clear();

  for (int r = 0; r < height; ++r) {
    const uchar *ptr = binary.ptr<uchar>(r);
    double y_real = (height - 1 - r) * y_scale;
    for (int c = 0; c < width; ++c) {
      if (ptr[c] > 0) {
        double x_real = c * x_scale;
        target_coords.emplace_back(x_real, y_real);
      }
    }
  }
  return true;
}

void FixBadApple::post_force(int /*vflag*/) {
  // Advance video frame on scheduled timesteps
  if (update->ntimestep % every == 0) {
    if (!load_next_frame()) {
      // Loop video or hold last frame
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      load_next_frame();
    }
  }

  if (target_coords.empty()) return;

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;

  size_t n_targets = target_coords.size();

  // Apply harmonic steering force to pull atoms into active silhouette points!
  for (int i = 0; i < nlocal; ++i) {
    if (mask[i] & groupbit) {
      // Map atom ID modulo target count to distribute particles over the shape
      size_t target_idx = (tag[i] - 1) % n_targets;
      double tx = target_coords[target_idx].first;
      double ty = target_coords[target_idx].second;

      // F = -k * (x - x_target) - damping
      f[i][0] += -k_spring * (x[i][0] - tx);
      f[i][1] += -k_spring * (x[i][1] - ty);
      f[i][2] += -k_spring * (x[i][2] - 0.0); // Pin to z=0 plane
    }
  }
}
