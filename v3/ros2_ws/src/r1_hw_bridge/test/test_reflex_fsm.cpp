// The bridge's reflex (reflex_fsm.hpp) against the reference: replays every tick of
// v3/reflex/golden.csv -- the inputs reflex.py was given and what it answered -- and
// fails on the first difference in each case. Also checks that this header's default
// configuration is the reference's (the golden file's "# defaults:" line) and that
// ParseProximity reads perception's layout. V3 dance: the same for the conductor
// (conductor.hpp) against v3/reflex/golden_dance.csv (conductor.py).
//
// Standard library only, so it needs neither ROS nor gtest:
//   g++ -std=c++17 -Wall -Wextra -I include test/test_reflex_fsm.cpp -o /tmp/t
//   /tmp/t <golden.csv> [<golden_dance.csv>]
// Without arguments: ../../../reflex/golden.csv and golden_dance.csv next to it.
#include "r1_hw_bridge/conductor.hpp"
#include "r1_hw_bridge/reflex_fsm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace r1_hw_bridge::reflex;

namespace
{

int g_fail = 0, g_pass = 0;

void Check(bool ok, const std::string & name, const std::string & detail = "")
{
  if (ok) {
    ++g_pass;
    std::printf("  ok    %s\n", name.c_str());
  } else {
    ++g_fail;
    std::printf("  FAIL  %s  <- %s\n", name.c_str(), detail.c_str());
  }
}

std::vector<std::string> Split(const std::string & s, char sep)
{
  std::vector<std::string> out;
  std::string cur;
  std::istringstream in(s);
  while (std::getline(in, cur, sep)) {out.push_back(cur);}
  if (!s.empty() && s.back() == sep) {out.push_back("");}
  return out;
}

double Num(const std::string & s) {return s == "nan" ? std::nan("") : std::strtod(s.c_str(), nullptr);}

// Sets one Config field by its reflex.py name; false if the name is unknown.
bool SetField(Config & c, const std::string & k, const std::string & v)
{
  const double x = Num(v);
  std::map<std::string, double *> d{
    {"person_trigger_m", &c.person_trigger_m}, {"obstacle_trigger_m", &c.obstacle_trigger_m},
    {"clear_margin_m", &c.clear_margin_m}, {"retreat_vx", &c.retreat_vx},
    {"retreat_s", &c.retreat_s}, {"window_s", &c.window_s}, {"stale_s", &c.stale_s},
    {"center_valid_min", &c.center_valid_min}, {"release_margin_m", &c.release_margin_m},
    {"person_floor_m", &c.person_floor_m}, {"obstacle_floor_m", &c.obstacle_floor_m},
    {"stop_margin_m", &c.stop_margin_m}, {"latency_s", &c.latency_s}, {"k_lat", &c.k_lat},
    {"k_brake", &c.k_brake}, {"speed_window_s", &c.speed_window_s}};
  if (d.count(k)) {*d[k] = x; return true;}
  if (k == "enable") {c.enable = x != 0.0; return true;}
  if (k == "confirm_frames") {c.confirm_frames = static_cast<int>(x); return true;}
  if (k == "max_retreats") {c.max_retreats = static_cast<int>(x); return true;}
  return false;
}

// "key=value key=value" -> the Config they describe, starting from `base`.
bool ParseCfg(const std::string & s, Config & c, std::string & bad)
{
  for (const auto & kv : Split(s, ' ')) {
    if (kv.empty()) {continue;}
    const auto eq = kv.find('=');
    if (eq == std::string::npos || !SetField(c, kv.substr(0, eq), kv.substr(eq + 1))) {
      bad = kv;
      return false;
    }
  }
  return true;
}

bool SameConfig(const Config & a, const Config & b)
{
  return a.enable == b.enable && a.person_trigger_m == b.person_trigger_m &&
         a.obstacle_trigger_m == b.obstacle_trigger_m && a.clear_margin_m == b.clear_margin_m &&
         a.confirm_frames == b.confirm_frames && a.retreat_vx == b.retreat_vx &&
         a.retreat_s == b.retreat_s && a.max_retreats == b.max_retreats &&
         a.window_s == b.window_s && a.stale_s == b.stale_s &&
         a.center_valid_min == b.center_valid_min && a.release_margin_m == b.release_margin_m &&
         a.person_floor_m == b.person_floor_m && a.obstacle_floor_m == b.obstacle_floor_m &&
         a.stop_margin_m == b.stop_margin_m && a.latency_s == b.latency_s && a.k_lat == b.k_lat &&
         a.k_brake == b.k_brake && a.speed_window_s == b.speed_window_s;
}

// "dance_enable=1 dance_person_trigger_m=0.9 ... dance_bar_ticks=150/150/100/100"
bool SameDanceDefaults(const std::string & s, std::string & bad)
{
  const DanceConfig d;
  for (const auto & kv : Split(s, ' ')) {
    if (kv.empty()) {continue;}
    const auto eq = kv.find('=');
    const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
    bool ok = true;
    if (k == "dance_enable") {ok = (Num(v) != 0.0) == d.enable;}
    else if (k == "dance_person_trigger_m") {ok = Num(v) == d.person_trigger_m;}
    else if (k == "dance_obstacle_trigger_m") {ok = Num(v) == d.obstacle_trigger_m;}
    else if (k == "dance_clear_m") {ok = Num(v) == d.clear_m;}
    else if (k == "dance_still_s") {ok = Num(v) == d.still_s;}
    else if (k == "dance_heartbeat_s") {ok = Num(v) == d.heartbeat_s;}
    else if (k == "dance_max_s") {ok = Num(v) == d.max_s;}
    else if (k == "dance_bar_ticks") {
      const auto parts = Split(v, '/');
      ok = parts.size() == d.bar_ticks.size();
      for (std::size_t i = 0; ok && i < parts.size(); ++i) {ok = std::stoi(parts[i]) == d.bar_ticks[i];}
    } else {ok = false;}
    if (!ok) {bad = kv; return false;}
  }
  return true;
}

// Every tick of golden_dance.csv through the conductor; the number of ticks compared.
long ReplayDance(const std::string & path)
{
  std::ifstream in(path);
  if (!in) {
    Check(false, "golden_dance.csv", "cannot open " + path);
    return 0;
  }
  std::printf("== conductor golden cases (%s) ==\n", path.c_str());
  std::string line, cur_case;
  Config cfg_next;
  bool cfg_pending = false, defaults_seen = false, case_failed = false;
  Conductor cd;
  long rows = 0, case_rows = 0;
  std::map<std::string, int> col;
  auto finish_case = [&]() {
      if (!cur_case.empty() && !case_failed) {
        Check(true, cur_case + " (" + std::to_string(case_rows) + " ticks)");
      }
    };
  while (std::getline(in, line)) {
    if (line.rfind("# dance defaults: ", 0) == 0) {
      std::string bad;
      Check(SameDanceDefaults(line.substr(18), bad), "conductor.hpp's defaults are conductor.py's", bad);
      defaults_seen = true;
      continue;
    }
    if (line.rfind("#cfg ", 0) == 0) {
      const auto rest = line.substr(5);
      const auto sp = rest.find(' ');
      cfg_next = Config();
      std::string bad;
      if (!ParseCfg(sp == std::string::npos ? "" : rest.substr(sp + 1), cfg_next, bad)) {
        Check(false, "#cfg line", "unknown key " + bad);
      }
      cfg_pending = true;
      continue;
    }
    if (line.empty() || line[0] == '#') {continue;}
    const auto c = Split(line, ',');
    if (col.empty()) {
      for (std::size_t i = 0; i < c.size(); ++i) {col[c[i]] = static_cast<int>(i);}
      continue;
    }
    auto get = [&](const char * name) {return c.at(col.at(name));};
    if (get("case") != cur_case) {
      finish_case();
      cur_case = get("case");
      cd = Conductor(cfg_pending ? cfg_next : Config());
      cfg_pending = false;
      case_failed = false;
      case_rows = 0;
    }
    ++rows;
    ++case_rows;
    if (case_failed) {continue;}
    Frame fr;
    const bool has_frame = get("frame") == "1";
    if (has_frame) {
      fr.seq = std::stol(get("seq"));
      fr.t_cap = Num(get("t_cap"));
      const double pm = Num(get("person_m")), om = Num(get("obstacle_m"));
      if (!std::isnan(pm)) {fr.person_m = pm;}
      if (!std::isnan(om)) {fr.obstacle_m = om;}
      fr.person_near_box = get("person_near_box") == "1";
      fr.depth_ok = get("depth_ok") == "1";
      fr.center_valid = Num(get("center_valid"));
    }
    const std::string dm = get("dance");
    const DanceMsg msg = dm == "1" ? DanceMsg::Dance : (dm == "0" ? DanceMsg::End : DanceMsg::None);
    const ConductorOutput o = cd.Step(Num(get("t")), Num(get("cmd_vx")), Num(get("cmd_vy")),
                                      Num(get("cmd_wz")), has_frame ? &fr : nullptr,
                                      get("reset") == "1", get("running") == "1", msg,
                                      get("dance_ok") == "1");
    const Output & r = o.reflex;
    const int want_state = std::stoi(get("out_state"));
    const double wvx = Num(get("out_vx")), wvy = Num(get("out_vy")), wwz = Num(get("out_wz"));
    const int want_reason = std::stoi(get("out_reason"));
    const bool want_blocked = get("out_blocked") == "1", want_refused = get("out_refused") == "1";
    const int want_policy = std::stoi(get("policy")), want_dstate = std::stoi(get("dance_state"));
    const long want_tick = std::stol(get("dance_tick"));
    const int want_dreason = std::stoi(get("dance_reason"));
    if (static_cast<int>(r.state) != want_state || r.vx != wvx || r.vy != wvy || r.wz != wwz ||
      r.reason != want_reason || r.blocked != want_blocked || r.reset_refused != want_refused ||
      static_cast<int>(o.policy) != want_policy || static_cast<int>(o.dance_state) != want_dstate ||
      o.dance_tick != want_tick || o.dance_reason != want_dreason)
    {
      char buf[500];
      std::snprintf(buf, sizeof(buf),
        "tick %s t=%s: want %s vx %.17g reason %d | policy %d tick %ld dance %d reason %d; "
        "got %s vx %.17g reason %d | policy %d tick %ld dance %d reason %d",
        get("k").c_str(), get("t").c_str(), StateName(static_cast<State>(want_state)), wvx,
        want_reason, want_policy, want_tick, want_dstate, want_dreason, StateName(r.state), r.vx,
        r.reason, static_cast<int>(o.policy), o.dance_tick, static_cast<int>(o.dance_state),
        o.dance_reason);
      Check(false, cur_case, buf);
      case_failed = true;
    }
  }
  finish_case();
  Check(defaults_seen, "golden_dance.csv states the conductor's defaults");
  return rows;
}

}  // namespace

int main(int argc, char ** argv)
{
  const std::string path = argc > 1 ? argv[1] : "../../../reflex/golden.csv";
  const std::string dance_path = argc > 2 ? argv[2] :
    path.substr(0, path.find_last_of('/') + 1) + "golden_dance.csv";
  std::ifstream in(path);
  if (!in) {
    std::printf("cannot open %s\n", path.c_str());
    return 2;
  }

  std::printf("== stop distance ==\n");
  const Config dc;
  Check(std::round(StopDistance(dc.obstacle_trigger_m, dc.obstacle_floor_m, 0.3, dc) * 100) == 87 &&
        std::round(StopDistance(dc.obstacle_trigger_m, dc.obstacle_floor_m, 1.0, dc) * 100) == 154 &&
        std::round(StopDistance(dc.person_trigger_m, dc.person_floor_m, 1.0, dc) * 100) == 169 &&
        StopDistance(dc.person_trigger_m, dc.person_floor_m, 0.0, dc) == 1.2,
        "the table: obstacle 0.87 m at 0.3 m/s, 1.54 at 1.0; person 1.69 at 1.0, 1.2 standing");

  std::printf("== proximity message ==\n");
  std::vector<double> msg(kProxLen, 0.0);
  msg[kProxVersionIdx] = 1.0; msg[kProxSeq] = 9; msg[kProxTCap] = 1000.25;
  msg[kProxPersonM] = std::nan(""); msg[kProxObstacleM] = 0.5; msg[kProxDepthOk] = 1.0;
  msg[kProxCenterValid] = 0.8;
  const auto f = ParseProximity(msg);
  Check(f && f->seq == 9 && f->t_cap == 1000.25 && !f->person_m && f->obstacle_m &&
        *f->obstacle_m == 0.5 && f->depth_ok && f->center_valid == 0.8,
        "a frame is read; NaN is no reading");
  auto wrong = msg;
  wrong[kProxVersionIdx] = 2.0;
  Check(!ParseProximity(wrong) && !ParseProximity(std::vector<double>(kProxLen - 1, 1.0)),
        "a wrong version or length is refused");

  std::printf("== golden cases (%s) ==\n", path.c_str());
  std::string line, header, cur_case;
  Config cfg_next;
  bool cfg_pending = false, defaults_seen = false, case_failed = false;
  Fsm fsm;
  long rows = 0, cases = 0, case_rows = 0;
  std::map<std::string, int> col;
  auto finish_case = [&]() {
      if (cur_case.empty()) {return;}
      ++cases;
      if (!case_failed) {Check(true, cur_case + " (" + std::to_string(case_rows) + " ticks)");}
    };
  while (std::getline(in, line)) {
    if (line.rfind("# defaults: ", 0) == 0) {
      Config py;
      std::string bad;
      const bool ok = ParseCfg(line.substr(12), py, bad);
      Check(ok && SameConfig(py, Config()),
            "reflex_fsm.hpp's defaults are reflex.py's", ok ? "values differ" : "unknown key " + bad);
      defaults_seen = true;
      continue;
    }
    if (line.rfind("#cfg ", 0) == 0) {
      const auto rest = line.substr(5);
      const auto sp = rest.find(' ');
      cfg_next = Config();
      std::string bad;
      if (!ParseCfg(sp == std::string::npos ? "" : rest.substr(sp + 1), cfg_next, bad)) {
        Check(false, "#cfg line", "unknown key " + bad);
      }
      cfg_pending = true;
      continue;
    }
    if (line.empty() || line[0] == '#') {continue;}
    const auto c = Split(line, ',');
    if (col.empty()) {
      for (std::size_t i = 0; i < c.size(); ++i) {col[c[i]] = static_cast<int>(i);}
      continue;
    }
    auto get = [&](const char * name) {return c.at(col.at(name));};
    if (get("case") != cur_case) {
      finish_case();
      cur_case = get("case");
      fsm = Fsm(cfg_pending ? cfg_next : Config());
      cfg_pending = false;
      case_failed = false;
      case_rows = 0;
    }
    ++rows;
    ++case_rows;
    if (case_failed) {continue;}
    Frame fr;
    const bool has_frame = get("frame") == "1";
    if (has_frame) {
      fr.seq = std::stol(get("seq"));
      fr.t_cap = Num(get("t_cap"));
      const double pm = Num(get("person_m")), om = Num(get("obstacle_m"));
      if (!std::isnan(pm)) {fr.person_m = pm;}
      if (!std::isnan(om)) {fr.obstacle_m = om;}
      fr.person_near_box = get("person_near_box") == "1";
      fr.depth_ok = get("depth_ok") == "1";
      fr.center_valid = Num(get("center_valid"));
    }
    const Output o = fsm.Step(Num(get("t")), Num(get("cmd_vx")), Num(get("cmd_vy")),
                              Num(get("cmd_wz")), has_frame ? &fr : nullptr,
                              get("reset") == "1", get("running") == "1");
    const int want_state = std::stoi(get("out_state"));
    const double wvx = Num(get("out_vx")), wvy = Num(get("out_vy")), wwz = Num(get("out_wz"));
    const int want_reason = std::stoi(get("out_reason"));
    const bool want_blocked = get("out_blocked") == "1", want_refused = get("out_refused") == "1";
    if (static_cast<int>(o.state) != want_state || o.vx != wvx || o.vy != wvy || o.wz != wwz ||
      o.reason != want_reason || o.blocked != want_blocked || o.reset_refused != want_refused)
    {
      char buf[400];
      std::snprintf(buf, sizeof(buf),
        "tick %s t=%s: want %s (%.17g, %.17g, %.17g) reason %d blocked %d refused %d; "
        "got %s (%.17g, %.17g, %.17g) reason %d blocked %d refused %d",
        get("k").c_str(), get("t").c_str(), StateName(static_cast<State>(want_state)), wvx, wvy,
        wwz, want_reason, want_blocked, want_refused, StateName(o.state), o.vx, o.vy, o.wz,
        o.reason, o.blocked, o.reset_refused);
      Check(false, cur_case, buf);
      case_failed = true;
    }
  }
  finish_case();
  Check(defaults_seen, "the golden file states the reference's defaults");
  Check(cases > 0 && rows > 0, "the golden file has cases",
        std::to_string(cases) + " cases, " + std::to_string(rows) + " ticks");

  std::printf("== the conductor's bars ==\n");
  const Conductor cbars;
  Check(cbars.loop_ticks() == 500 && cbars.AtBoundary(0) && cbars.AtBoundary(150) &&
        cbars.AtBoundary(300) && cbars.AtBoundary(400) && cbars.AtBoundary(650) &&
        !cbars.AtBoundary(149), "10 s loop, bar boundaries at 0 / 3 / 6 / 8 s");
  const long dance_rows = ReplayDance(dance_path);

  std::printf("--------------------------------------------------\n");
  std::printf("pass %d  fail %d  (%ld + %ld ticks compared exactly)\n", g_pass, g_fail, rows,
              dance_rows);
  return g_fail ? 1 : 0;
}
