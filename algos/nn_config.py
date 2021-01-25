## agent

ppo_lagrange_kwargs = dict(
    reward_penalized=False, objective_penalized=True,
    learn_penalty=True, penalty_param_loss=True  # Irrelevant in unconstrained
)


scarlet_config = dict(name='scarlet',penalty_lr=0.025, cost_lim=25, gamma=0.985, lam=0.98, seed=0, steps=8000, hid=128, l=2)

lemon_config    = dict(name='lemon', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=0, steps=8000, hid=128, l=2)

cyan_config     = dict(name='cyan',  penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=0, steps=8000, hid=128, l=4)

navy_config      = dict(name='navy', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=0, steps=8000, hid=128, l=4)

rose_config      = dict(name='rose', penalty_lr=0.025, cost_lim=0, gamma=0.99, lam=0.98, seed=0, steps=20000, hid=128, l=4)

hyacinth_config= dict(name='hyacinth', penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.98, seed=0, steps=20000, hid=128, l=4)

violet_config   = dict(name='violet', penalty_lr=0.025, cost_lim=0, gamma=0.999, lam=0.98, seed=0, steps=20000, hid=128, l=4)

marigold_config = dict(name='marigold', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.98, seed=0, steps=20000, hid=128, l=4)

marigold2_config = dict(name='marigold2', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)

lilly_config    = dict(name='lilly', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)

peony_config    = dict(name='peony' ,penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)
