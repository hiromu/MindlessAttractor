# Mindless Attractor: A False-Positive Resistant Intervention for Drawing Attention Using Auditory Perturbation (ACM CHI 2021)

This repository contains a sample code showing how audio signals are perturbated to draw attention from users.

In `main.py`, `convert_on_state(signal)` performs the perturbation process based on the `mode_state` and `mode_param` variables, which are automatically switched by `local_switch(mode_state, mode_param)` for the demonstration purpose.
The `mode_state` variable denotes which of pitch shifting, volume changing, and beeping is going to be performed while the `mode_param` variable denotes the magnitude parameter in pitch shifting and volume changing.
