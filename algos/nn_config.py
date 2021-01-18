## agent

ppo_lagrange_kwargs = dict(
    reward_penalized=False, objective_penalized=True,
    learn_penalty=True, penalty_param_loss=True  # Irrelevant in unconstrained
)

## nn
scarlet_config = dict(penalty_lr=0.025,
                        cost_lim=25,
                        gamma=0.985,
                        lam=0.98,
                        seed=0,
                        steps=8000,
                        hid=128,
                        l=2)

lemon_config = dict(penalty_lr=0.025,
                      cost_lim=25,
                      gamma=0.99,
                      lam=0.98,
                      seed=0,
                      steps=8000,
                      hid=128,
                      l=2)

### cyan
cyan_config = dict(penalty_lr=0.025,
                     cost_lim=25,
                     gamma=0.99,
                     lam=0.98,
                     seed=0,
                     steps=8000,
                     hid=128,
                     l=4)

### navy
navy_config = dict(penalty_lr=0.025,
                     cost_lim=25,
                     gamma=0.99,
                     lam=0.98,
                     seed=0,
                     steps=8000,
                     hid=128,
                     l=4)