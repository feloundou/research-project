# ## agent
#
# ppo_lagrange_kwargs = dict(
#     reward_penalized=False, objective_penalized=True,
#     learn_penalty=True, penalty_param_loss=True  # Irrelevant in unconstrained
# )

buttercup_config  = dict(name='buttercup', penalty_lr=0.001, cost_lim=0, gamma=1, lam=0.95, seed=0, steps=20000, hid=256, l=4)
brownsugar_config = dict(name='brownsugar', penalty_lr=0.001, cost_lim=0, gamma=1, lam=0.95, seed=4444, steps=20000, hid=256, l=4)

cyan_config       = dict(name='cyan',  penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=0, steps=8000, hid=128, l=4)
chrome_config     = dict(name='chrome',  penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=4444, steps=8000, hid=128, l=4)

hyacinth_config= dict(name='hyacinth', penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.98, seed=0, steps=20000, hid=128, l=4)
heather_config= dict(name='heather', penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.98, seed=4444, steps=20000, hid=128, l=4)

lemon_config    = dict(name='lemon', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=0, steps=8000, hid=128, l=2)
leaf_config    = dict(name='leaf', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=4444, steps=8000, hid=128, l=2)

lilly_config    = dict(name='lilly', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)
lavender_config = dict(name='lavender', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.95, seed=4444, steps=20000, hid=128, l=4)

navy_config      = dict(name='navy', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=0, steps=8000, hid=128, l=4)
nova_config      = dict(name='nova', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=4444, steps=8000, hid=128, l=4)

marigold_config = dict(name='marigold', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.98, seed=0, steps=20000, hid=128, l=4)
magenta_config  = dict(name='magenta', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.98, seed=4444, steps=20000, hid=128, l=4)

petrichor_config = dict(name='petrichor', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)
polenta_config = dict(name='polenta', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.95, seed=4444, steps=20000, hid=128, l=4)

peony_config    = dict(name='peony' ,penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)
prance_config    = dict(name='prance' ,penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.95, seed=4444, steps=20000, hid=128, l=4)

query_config     = dict(name='query', penalty_lr=0.01, cost_lim=0.5, gamma=1, lam=0.98, seed=123, steps=20000, hid=128, l=4)
quandary_config = dict(name='quandary', penalty_lr=0.01, cost_lim=0.5, gamma=1, lam=0.98, seed=123, steps=20000, hid=128, l=4)

rose_config      = dict(name='rose', penalty_lr=0.025, cost_lim=0, gamma=0.99, lam=0.98, seed=0, steps=20000, hid=128, l=4)
rhizome_config   = dict(name='rhizome', penalty_lr=0.025, cost_lim=0, gamma=0.99, lam=0.98, seed=4444, steps=20000, hid=128, l=4)

scarlet_config = dict(name='scarlet',penalty_lr=0.025, cost_lim=25, gamma=0.985, lam=0.98, seed=0, steps=8000, hid=128, l=2)
soprano_config = dict(name='soprano',penalty_lr=0.025, cost_lim=25, gamma=0.985, lam=0.98, seed=4444, steps=8000, hid=128, l=2)

standard_config = dict(name='standard', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.98, seed=0, steps=5000, hid=128, l=4)
status_config = dict(name='status', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.98, seed=4444, steps=5000, hid=128, l=4)

ukelele_config = dict(name='ukelele', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)
uniform_config = dict(name='uniform', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.95, seed=4444, steps=20000, hid=128, l=4)

violet_config = dict(name='violet', penalty_lr=0.025, cost_lim=0, gamma=0.999, lam=0.95, seed=0, steps=20000, hid=128, l=4)
viola_config = dict(name='viola', penalty_lr=0.025, cost_lim=0, gamma=0.999, lam=0.95, seed=4444, steps=20000, hid=128, l=4)

zebra_config = dict(name='zebra', penalty_lr=0.025, cost_lim=0, gamma=0.999, lam=0.95, seed=0, steps=10000, hid=128, l=4)
zany_config = dict(name='zany', penalty_lr=0.025, cost_lim=0, gamma=0.999, lam=0.95, seed=4444, steps=20000, hid=128, l=4)


CONFIG_LIST = [buttercup_config, brownsugar_config,
               cyan_config, chrome_config,
               hyacinth_config, heather_config,
               lemon_config, leaf_config,
               lilly_config, lavender_config,
               navy_config, nova_config,
               marigold_config, magenta_config,
               petrichor_config, polenta_config,
               peony_config, prance_config,
               query_config, quandary_config,
               rose_config, rhizome_config,
               scarlet_config, soprano_config,
               standard_config, status_config,
               ukelele_config, uniform_config,
               violet_config, viola_config,
               zebra_config, zany_config]

