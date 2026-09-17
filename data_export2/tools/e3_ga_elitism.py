"""E3: GA with elitism = 1, on the two circuits where the GA underperforms.

ga_agent.py is NOT modified. GA_Agent is subclassed and only run() is
reimplemented, identically to ga_agent.py:271-337 except that the best
individual of each generation is carried unchanged into the next population
(replacing the last of the num_ind offspring, so the population stays 8).

Everything else is the paper's configuration: population 8, roulette selection,
one-point crossover p=0.87, mutation p=0.7, patience 50, no cache.

Note for the write-up: even without elitism the *reported* GA number is the
best individual ever evaluated -- best_fit_global starts at inf
(ga_agent.py:278), is only ever lowered (ga_agent.py:291-292) and is what run()
returns (ga_agent.py:312, :337) -- so the incumbent is never lost from the
answer; elitism can only change search progress.

Usage (inside distrobox ubuntu-work):
    python e3_ga_elitism.py CIRCUIT SEED
"""
import random
import sys
from pathlib import Path

import numpy as np

from elib import *  # noqa: F401,F403

GA_DIR = Path("/home/digital-2/workspace/vtr_exp")
sys.path.insert(0, str(GA_DIR))
from ga_agent import GA_Agent  # noqa: E402


class ElitistGA(GA_Agent):
    def run(self, log_suffix=None):
        suffix = log_suffix if log_suffix is not None else self.benchmark_name
        out_path = GA_DIR / f"ga_output_{suffix}.txt"

        pop_bag = self.initiate()
        t_con = [0 for _ in range(self.num_gen)]
        g = 0
        best_fit_global, best_sol_global, best_metrics_global = float("inf"), None, None

        for gen in range(self.num_gen):
            print(f"--- Generation {gen} ---", flush=True)
            pop_bag_fit = self.eval_fit_pop(pop_bag)

            best_fit = np.min(pop_bag_fit["fit_val"])
            i = pop_bag_fit["fit_val"].index(best_fit)
            gen_best_sol = pop_bag_fit["sol"][i]

            if best_fit < best_fit_global:
                best_fit_global = best_fit
                best_sol_global = gen_best_sol
                best_metrics_global = pop_bag_fit["metrics"][i]
                print(f"*** NEW BEST GLOBAL FITNESS: {best_fit_global} ***", flush=True)

            with open(out_path, "a+") as f:
                ar = best_sol_global.get("aspect_ratio", 1.0) if best_sol_global else "N/A"
                f.write(f"Generation {gen} | Best Fitness: {best_fit_global} | "
                        f"Metrics: {best_metrics_global} | Aspect Ratio: {ar} | "
                        f"Layout: {best_sol_global}\n")

            t_con[g] = best_fit_global
            if g >= self.patience:
                if sum(1 for k in range(g - self.patience, g + 1)
                       if t_con[k] == best_fit_global) >= self.patience + 1:
                    print("Breaking condition reached.", flush=True)
                    return best_fit_global, best_sol_global, best_metrics_global

            new_pop_bag = []
            for _ in range(self.num_ind - 1):          # ELITISM: one slot reserved
                pA = self.pick(pop_bag, pop_bag_fit)
                pB = self.pick(pop_bag, pop_bag_fit)
                el = pA
                if random.random() <= self.cross_prob:
                    el = self.crossover(pA, pB)
                if random.random() <= self.mut_prob:
                    el = self.mutation(el)
                if len(el["dsps"]) < self.req_dsp or len(el["brams"]) < self.req_bram:
                    el = self.generate_random_solution()
                new_pop_bag.append(el)
            new_pop_bag.append({"dsps": list(gen_best_sol["dsps"]),
                                "brams": list(gen_best_sol["brams"]),
                                "aspect_ratio": gen_best_sol["aspect_ratio"]})  # elite, unchanged
            pop_bag = new_pop_bag
            g += 1
        return best_fit_global, best_sol_global, best_metrics_global


if __name__ == "__main__":
    circuit, seed = sys.argv[1], int(sys.argv[2])
    _, res = baseline(circuit)                      # noqa: F405
    w, h = res["fpga_size"]
    random.seed(seed)
    np.random.seed(seed)
    agent = ElitistGA(benchmark_name=circuit, width=w, height=h,
                      req_dsp=res["requirements"]["dsp"], req_bram=res["requirements"]["bram"],
                      num_ind=8, num_gen=500, patience=50)
    print(f"E3 elitist GA: {circuit} seed {seed} grid {w}x{h} "
          f"dsp={res['requirements']['dsp']} bram={res['requirements']['bram']}", flush=True)
    print(agent.run(log_suffix=f"{circuit}_elitism_seed{seed}")[0], flush=True)
