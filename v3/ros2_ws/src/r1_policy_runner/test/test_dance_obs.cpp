// The runner's dance observation (dance_obs.hpp) against the Python reference
// (v3/dance/dance_obs.py): the frames in test/data/dance_obs_fixture.bin through
// DanceObsBuilder must give the observations stored next to them. Standard library only:
//   g++ -std=c++17 -Wall -Wextra -I include test/test_dance_obs.cpp -o /tmp/t
//   /tmp/t test/data/dance_ref.bin test/data/dance_obs_fixture.bin
#include "r1_policy_runner/dance_obs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace r1_policy_runner;

int main(int argc, char ** argv)
{
  const std::string dir = "test/data/";
  const std::string ref_path = argc > 1 ? argv[1] : dir + "dance_ref.bin";
  const std::string fix_path = argc > 2 ? argv[2] : dir + "dance_obs_fixture.bin";
  int fail = 0;
  try {
    const DanceReference ref(ref_path);
    std::printf("reference: %zu frames at %.0f fps\n", ref.frames(), ref.fps());
    if (ref.frames() != 500 || ref.fps() != 50.0f) {
      std::printf("  FAIL  expected one 10 s loop at 50 fps\n");
      ++fail;
    }
    std::ifstream in(fix_path, std::ios::binary);
    char magic[12];
    uint32_t n = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in || std::string(magic, 12) != "R1DANCEOBS1\n" || n == 0) {
      std::printf("  FAIL  %s is not a fixture\n", fix_path.c_str());
      return 1;
    }
    DanceObsBuilder b(ref);
    std::vector<float> frame(kDanceFrame), want(kDanceObs), got(kDanceObs);
    double worst = 0.0;
    std::size_t worst_step = 0, worst_i = 0;
    for (uint32_t s = 0; s < n; ++s) {
      in.read(reinterpret_cast<char *>(frame.data()), kDanceFrame * 4);
      in.read(reinterpret_cast<char *>(want.data()), kDanceObs * 4);
      if (!in) {
        std::printf("  FAIL  fixture truncated at step %u\n", s);
        return 1;
      }
      b.Build(frame.data(), got.data());
      for (std::size_t i = 0; i < kDanceObs; ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - want[i]);
        if (d > worst) {worst = d; worst_step = s; worst_i = i;}
      }
    }
    const bool ok = worst <= 1e-6;
    std::printf("  %s  %u steps (standby, a dance across the loop's end, a second dance): "
                "max |C++ - Python| %.3g (step %zu, element %zu)\n", ok ? "ok  " : "FAIL",
                n, worst, worst_step, worst_i);
    fail += ok ? 0 : 1;
  } catch (const std::exception & e) {
    std::printf("  FAIL  %s\n", e.what());
    return 1;
  }
  std::printf("%s\n", fail ? "FAILED" : "passed");
  return fail ? 1 : 0;
}
