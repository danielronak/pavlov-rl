import jax, numpy as np
from pavlov.physiology import run_open_loop
def lifetime_returns(sched, actions, pc):
    f = jax.vmap(lambda fo, a, fl: run_open_loop(fo, a, fl, pc)["reward"].sum())
    return np.asarray(f(sched.food_onset, actions, sched.params.flipped))
