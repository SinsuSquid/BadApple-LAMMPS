#ifndef LMP_FIX_BAD_APPLE_KOKKOS_H
#define LMP_FIX_BAD_APPLE_KOKKOS_H

#include "fix_bad_apple.h"
#include "kokkos_type.h"
#include <opencv2/opencv.hpp>

namespace LAMMPS_NS {

template <class DeviceType>
class FixBadAppleKokkos : public FixBadApple {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixBadAppleKokkos(class LAMMPS *, int, char **);
  virtual ~FixBadAppleKokkos() = default;

  void post_force(int) override;
  bool load_next_frame();

  // Kokkos-accessible member variables
  Kokkos::View<uint8_t**, typename DeviceType::array_layout, DeviceType> d_binary;
  Kokkos::View<float**,   typename DeviceType::array_layout, DeviceType> d_dist_map;

  typename Kokkos::View<uint8_t**, typename DeviceType::array_layout, DeviceType>::HostMirror h_binary;
  typename Kokkos::View<float**,   typename DeviceType::array_layout, DeviceType>::HostMirror h_dist_map;

 private:
  class AtomKokkos *atomKK;
};

} // namespace LAMMPS_NS

#endif
