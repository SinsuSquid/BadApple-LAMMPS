#include "fix_bad_apple_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

template <class DeviceType>
FixBadAppleKokkos<DeviceType>::FixBadAppleKokkos(LAMMPS *lmp, int narg, char **arg)
    : FixBadApple(lmp, narg, arg) {
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;

  // Allocate Kokkos 2D Views for binary mask and distance field (Height x Width)
  d_binary = Kokkos::View<uint8_t**, typename DeviceType::array_layout, DeviceType>(
      "BadApple::d_binary", height, width);
  d_dist_map = Kokkos::View<float**, typename DeviceType::array_layout, DeviceType>(
      "BadApple::d_dist_map", height, width);

  h_binary   = Kokkos::create_mirror_view(d_binary);
  h_dist_map = Kokkos::create_mirror_view(d_dist_map);

  // Initial frame load and upload to device
  load_next_frame();
}

template <class DeviceType>
bool FixBadAppleKokkos<DeviceType>::load_next_frame() {
  cv::Mat frame, gray, resized;
  if (!cap.read(frame) || frame.empty()) return false;

  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::resize(gray, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
  cv::threshold(resized, binary, threshold_val, 255, cv::THRESH_BINARY);

  cv::Mat inv_binary;
  cv::bitwise_not(binary, inv_binary);
  cv::distanceTransform(inv_binary, dist_map, cv::DIST_L2, 3);

  // Copy OpenCV buffers into Host Mirror Views
  for (int r = 0; r < height; ++r) {
    const uint8_t *bin_row = binary.ptr<uint8_t>(r);
    const float   *dst_row = dist_map.ptr<float>(r);
    for (int c = 0; c < width; ++c) {
      h_binary(r, c)   = bin_row[c];
      h_dist_map(r, c) = dst_row[c];
    }
  }

  // Blazingly fast Host -> Device DMA upload
  Kokkos::deep_copy(d_binary, h_binary);
  Kokkos::deep_copy(d_dist_map, h_dist_map);

  return true;
}

template <class DeviceType>
void FixBadAppleKokkos<DeviceType>::post_force(int /*vflag*/) {
  if (update->ntimestep % every == 0) {
    if (!load_next_frame()) {
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      load_next_frame();
    }
  }

  // Sync atom coordinates to Device and declare that we will modify forces on Device
  atomKK->sync(ExecutionSpaceFromDevice<DeviceType>::space, X_MASK | MASK_MASK);
  atomKK->modified(ExecutionSpaceFromDevice<DeviceType>::space, F_MASK);

  auto x    = atomKK->k_x.view<DeviceType>();
  auto f    = atomKK->k_f.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atom->nlocal;

  // Local copies for Lambda capture (avoid capturing 'this' on device!)
  const int gbit     = groupbit;
  const int w        = width;
  const int h        = height;
  const double xs    = x_scale;
  const double ys    = y_scale;
  const double k_sp  = k_spring;
  auto dev_bin       = d_binary;
  auto dev_dst       = d_dist_map;

  // GPU Parallel Loop
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(const int i) {
        if (mask(i) & gbit) {
          int c = static_cast<int>(x(i, 0) / xs);
          c = (c < 0) ? 0 : (c >= w ? w - 1 : c);

          int r = h - 1 - static_cast<int>(x(i, 1) / ys);
          r = (r < 0) ? 0 : (r >= h ? h - 1 : r);

          // Outside silhouette -> compute distance gradient field
          if (dev_bin(r, c) == 0) {
            float d = dev_dst(r, c);

            float d_right = (c + 1 < w) ? dev_dst(r, c + 1) : d;
            float d_left  = (c - 1 >= 0) ? dev_dst(r, c - 1) : d;
            float d_up    = (r - 1 >= 0) ? dev_dst(r - 1, c) : d;
            float d_down  = (r + 1 < h) ? dev_dst(r + 1, c) : d;

            double grad_x = static_cast<double>(d_right - d_left) / (2.0 * xs);
            double grad_y = -static_cast<double>(d_down - d_up)   / (2.0 * ys);

            f(i, 0) += -k_sp * grad_x;
            f(i, 1) += -k_sp * grad_y;
          }

          // Harmonic spring trap to Z=0 plane
          f(i, 2) += -100.0 * x(i, 2);
        }
      });
}

namespace LAMMPS_NS {
template class FixBadAppleKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class FixBadAppleKokkos<LMPHostType>;
#endif
}
