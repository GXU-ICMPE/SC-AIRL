# SC-AIRL in PyTorch
This is a PyTorch implementation of SC-AIRL


## Example

### Train expert
You can train experts using Soft Actor-Critic(SAC). 
```bash
python train_expert.py --cuda --env_id your id --num_steps your num_steps --seed 0
```

### Collect demonstrations
You need to collect demonstraions using trained expert's weight. 
```bash
python collect_demo.py 
```

### Train Imitation Learning
You can train IL using demonstrations.

```bash
python train_imitation.py 
```
### Test Demo
You can render the task with our prepared weight ( ).

```bash
python test_demo.py 
weight0 UR5-AIRL-master/logs/panda_stack_sc-airl/actor.pth
weight1 UR5-AIRL-master/logs/panda_stack_sc-airl/actor1.pth
```