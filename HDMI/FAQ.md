# FAQ

1. How are object trajectories obtained from demonstration videos?

   We manually annotate object trajectories and desired contact locations for each clip, then run a video-specific post-processing script. For instance, in the suitcase demo we mark the pickup and drop-off frames, align the suitcase with the wrist while it is held, place it on the ground otherwise, and interpolate with smoothing to produce the final trajectory. Future releases may integrate automated pipelines such as OmniRetarget to reduce this manual effort.

2. What does `roa` refer to in `active_adaptation/learning/ppo/ppo_roa.py`?

   `roa` stands for Regularized Online Adaptation, introduced in https://arxiv.org/abs/2210.10044 and extended in https://arxiv.org/abs/2505.06883. Our implementation drops the explicit regularization term but keeps the latent-space behavior cloning between the teacher and student encoders during teacher training. This distillation exposes the student to richer teacher information, accelerating training and aiding exploration, although training the student policy directly is also possible.

3. Were all benchmark tasks deployed on real hardware?

   No. We successfully deployed 6 of the 14 tasks. The remaining tasks were limited by hardware constraints-for example, we could not attach motion-capture markers to props like balls, and we lacked a folding chair with a rigid back. The simulation policies for those tasks remain available for further experimentation.

4. How does HDMI handle robustness to different initial poses and headings?

   We randomize initial positions and headings during training. Door-opening starts roughly within a 20 cm translational range, while bread box transport uses about a 10 cm range. 

5. Why link_pos/quat_w and com_lin/ang_vel for motion tracking rewards?

   Getting link lin/ang vel from isaacsim is significantly slower than com lin/ang vel, so I use the latter for efficiency.
