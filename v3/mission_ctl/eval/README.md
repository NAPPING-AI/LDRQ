# eval/ · the offline English eval for `ask`

`run_nl_eval.py` sends every case through exactly what `r1_mission_cli.py ask` runs
(normalize → model → translate → `plan_from_json`), and scores **what the robot would
do**. Results, method and the model choice: [docs/stageC_nl_eval.md](../../../docs/stageC_nl_eval.md).

| file | cases | role |
|---|---|---|
| `nl_dev.jsonl` | 20 | prompt design, and nothing else |
| `nl_test.jsonl` | 40 | the plan's four categories; first scored in run 1 |
| `nl_test2.jsonl` | 20 | frozen before run 2 |
| `nl_test3.jsonl` | 20 | frozen before run 3 |
| `nl_dev_objects.jsonl` | 8 | stage E prompt design (turn_to / walk_to), and nothing else |
| `nl_test4.jsonl` | 20 | stage E: object instructions, frozen before the stage E prompt was scored |
| `nl_test5.jsonl` | 20 | v2 policy: backward, sideways and faster turns, frozen before it was scored; judged against the v2 envelope (`SET_ENVELOPE` in `run_nl_eval.py`), not the shipped one |
| `sim_object_trials.py` | — | the stage E trial protocol (10 × turn_to, 10 × walk_to) in a simulated room |

```bash
python3 run_nl_eval.py --check --set all                 # no model: every case vs the shipped config
python3 run_nl_eval.py --model qwen3:1.7b                # the 80 held-out cases (default --set heldout)
python3 run_nl_eval.py --model qwen3:1.7b --set stageE --options '{"num_gpu": 0, "num_thread": 4}'   # + objects, on the CPU
python3 run_nl_eval.py --model qwen3:1.7b --set v2 --options '{"num_gpu": 0, "num_thread": 4}'   # the v2 directions
python3 run_nl_eval.py --model qwen3:1.7b --set dev -v   # prompt work; -v prints the wrong answers
python3 run_nl_eval.py --model qwen3:1.7b --options '{"num_gpu": 0}' --tag cpu   # CPU only
python3 run_nl_eval.py --table results/*.json            # markdown comparison
```

A case lists acceptable transcriptions (`expect`, a list of alternatives) and the
outcome they lead to under the shipped config (`outcome`). A case whose ideal answer the
deterministic layer cannot produce carries a `limit` note: it is scored as usual, so it
shows up as a miss, and `--check` skips it. **Do not tune the prompt against the test
sets.** `tests/test_nl.py` fails if the shipped prompt id is not the one the eval ran.
