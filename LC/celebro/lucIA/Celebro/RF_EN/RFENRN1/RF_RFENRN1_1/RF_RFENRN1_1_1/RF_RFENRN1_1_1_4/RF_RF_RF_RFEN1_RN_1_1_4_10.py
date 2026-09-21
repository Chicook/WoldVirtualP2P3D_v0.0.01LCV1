from __future__ import annotations

from typing import Tuple

from . import ENRN_TKNZ, ENRN_TRNS, SLRN_TTS, RF_ENV, RF_CMND, RFENRN_OPT
from .RF_RF_RF_RFEN1_RN_1_1_4_8 import PPOAgent
from .RF_RF_RF_RFEN1_RN_1_1_4_9 import OpenSimTerminal


def run_demo(goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    tok = ENRN_TKNZ()
    tr = ENRN_TRNS()
    tts = SLRN_TTS()
    env = RF_ENV()
    cmd = RF_CMND()
    opt = RFENRN_OPT()
    ppo = PPOAgent()
    term = OpenSimTerminal()

    # Text pipeline
    tok.fit_bpe(["hola lucia", "mueve el avatar", "crea un prim"], vocab_size=128)
    ids = tok.encode("hola lucia")
    out = tr.generate(ids, max_new_tokens=8)
    audio = tts.synthesize(out[:16], duration_sec=0.1)

    # RL loop (single step demo)
    logs = term.read_log()
    state = {}
    for line in logs:
        state = env.reduce_state(state, env.parse_log_line(line))
    action, _ = ppo.select_action(state)
    command = cmd.apply_policy(state)
    term.send_command(command)
    new_logs = term.read_log()
    for line in new_logs:
        state = env.reduce_state(state, env.parse_log_line(line))
    reward = env.compute_reward(state, goal)
    ppo.remember(state, action, reward)
    ppo.update()

    # Resource optimization sample
    res = opt.read_resources()
    _ = opt.adjust_pruning(res["cpu_util"], res["ram_util"])
    _ = opt.adjust_p2p_bandwidth(res["cpu_util"], res["ram_util"])

    # Side effects prevent lint warnings
    _ = len(audio) + len(ids) + len(out) + len(new_logs) + len(command)
