# ## agent
#
# ppo_lagrange_kwargs = dict(
#     reward_penalized=False, objective_penalized=True,
#     learn_penalty=True, penalty_param_loss=True  # Irrelevant in unconstrained
# )


#
# buttercup_config  = dict(name='buttercup', penalty_lr=0.001, cost_lim=0, gamma=1, lam=0.95, seed=0, steps=20000, hid=256, l=4)
# brownsugar_config = dict(name='brownsugar', penalty_lr=0.001, cost_lim=0, gamma=1, lam=0.95, seed=4444, steps=20000, hid=256, l=4)
#
# cyan_config       = dict(name='cyan',  penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=0, steps=8000, hid=128, l=4)
# chrome_config     = dict(name='chrome',  penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=4444, steps=8000, hid=128, l=4)
#
# hyacinth_config= dict(name='hyacinth', penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.98, seed=0, steps=20000, hid=128, l=4)
# heather_config= dict(name='heather', penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.98, seed=4444, steps=20000, hid=128, l=4)
#
# lemon_config    = dict(name='lemon', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=0, steps=8000, hid=128, l=2)
# leaf_config    = dict(name='leaf', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=4444, steps=8000, hid=128, l=2)
#
# lilly_config    = dict(name='lilly', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)
# lavender_config = dict(name='lavender', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.95, seed=4444, steps=20000, hid=128, l=4)
#
# navy_config      = dict(name='navy', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=0, steps=8000, hid=128, l=4)
# nova_config      = dict(name='nova', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.98, seed=4444, steps=8000, hid=128, l=4)
#
# marigold_config = dict(name='marigold', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.98, seed=0, steps=20000, hid=128, l=4)
# magenta_config  = dict(name='magenta', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.98, seed=4444, steps=20000, hid=128, l=4)
#
# marigold2_config = dict(name='marigold2', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.95, seed=0, steps=8000, hid=128, l=4)
#
#
# petrichor_config = dict(name='petrichor', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)
# polenta_config = dict(name='polenta', penalty_lr=0.025, cost_lim=25, gamma=0.99, lam=0.95, seed=4444, steps=20000, hid=128, l=4)
#
# peony_config    = dict(name='peony' ,penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)
# prance_config    = dict(name='prance' ,penalty_lr=0.025, cost_lim=10, gamma=0.99, lam=0.95, seed=4444, steps=20000, hid=128, l=4)
#
# query_config     = dict(name='query', penalty_lr=0.01, cost_lim=0.5, gamma=1, lam=0.98, seed=123, steps=20000, hid=128, l=4)
# quandary_config = dict(name='quandary', penalty_lr=0.01, cost_lim=0.5, gamma=1, lam=0.98, seed=123, steps=20000, hid=128, l=4)
#
# rose_config      = dict(name='rose', penalty_lr=0.025, cost_lim=0, gamma=0.99, lam=0.98, seed=0, steps=20000, hid=128, l=4)
# rhizome_config   = dict(name='rhizome', penalty_lr=0.025, cost_lim=0, gamma=0.99, lam=0.98, seed=4444, steps=20000, hid=128, l=4)
#
# scarlet_config = dict(name='scarlet',penalty_lr=0.025, cost_lim=25, gamma=0.985, lam=0.98, seed=0, steps=8000, hid=128, l=2)
# soprano_config = dict(name='soprano',penalty_lr=0.025, cost_lim=25, gamma=0.985, lam=0.98, seed=4444, steps=8000, hid=128, l=2)
#
# standard_config = dict(name='standard', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.98, seed=0, steps=5000, hid=128, l=4)
# status_config = dict(name='status', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.98, seed=4444, steps=5000, hid=128, l=4)
#
# ukelele_config = dict(name='ukelele', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.95, seed=0, steps=20000, hid=128, l=4)
# uniform_config = dict(name='uniform', penalty_lr=0.025, cost_lim=50, gamma=0.99, lam=0.95, seed=4444, steps=20000, hid=128, l=4)
#
# violet_config = dict(name='violet', penalty_lr=0.025, cost_lim=0, gamma=0.999, lam=0.95, seed=0, steps=20000, hid=128, l=4)
# viola_config = dict(name='viola', penalty_lr=0.025, cost_lim=0, gamma=0.999, lam=0.95, seed=4444, steps=20000, hid=128, l=4)
#
# zebra_config = dict(name='zebra', penalty_lr=0.025, cost_lim=0, gamma=0.999, lam=0.95, seed=0, steps=10000, hid=128, l=4)
# zany_config = dict(name='zany', penalty_lr=0.025, cost_lim=0, gamma=0.999, lam=0.95, seed=4444, steps=20000, hid=128, l=4)


amaranth_config = dict(name='amaranth-80s-20cpu', penalty_lr=0.01, cost_lim=0, gamma=0.99, lam=0.95, seed=0, steps=80000, hid=128, l=4)
aster_config = dict(name='aster-80s-20cpu', penalty_lr=0.01, cost_lim=0, gamma=0.99, lam=0.95, seed=123, steps=80000, hid=128, l=4)
armadillo_config = dict(name='armadillo-80s-20cpu', penalty_lr=0.01, cost_lim=0, gamma=0.99, lam=0.95, seed=4444, steps=80000, hid=128, l=4)
acacia_config = dict(name='acacia-80s-20cpu', penalty_lr=0.01, cost_lim=0, gamma=0.99, lam=0.95, seed=1248, steps=80000, hid=128, l=4)
amaryllis_config = dict(name='amaryllis-80s-20cpu', penalty_lr=0.01, cost_lim=0, gamma=0.99, lam=0.95, seed=2496, steps=80000, hid=128, l=4)


chrysanthemum_config = dict(name='chrysanthemum-80s-20cpu', penalty_lr=0.01, cost_lim=10, gamma=0.99, lam=0.95, seed=0, steps=80000, hid=128, l=4)
chlorophyll_config = dict(name='chlorophyll-80s-20cpu', penalty_lr=0.01, cost_lim=10, gamma=0.99, lam=0.95, seed=123, steps=80000, hid=128, l=4)
carnation_config = dict(name='carnation-80s-20cpu', penalty_lr=0.01, cost_lim=10, gamma=0.99, lam=0.95, seed=4444, steps=80000, hid=128, l=4)
camellia_config = dict(name='camellia-80s-20cpu', penalty_lr=0.01, cost_lim=10, gamma=0.99, lam=0.95, seed=1248, steps=80000, hid=128, l=4)
carapace_config = dict(name='carapace-80s-20cpu', penalty_lr=0.01, cost_lim=10, gamma=0.99, lam=0.95, seed=2496, steps=80000, hid=128, l=4)


dahlia_config = dict(name='dahlia-80s-20cpu', penalty_lr=0.01, cost_lim=20, gamma=0.99, lam=0.95, seed=0, steps=80000, hid=128, l=4)
daffodil_config = dict(name='daffodil-80s-20cpu', penalty_lr=0.01, cost_lim=20, gamma=0.99, lam=0.95, seed=123, steps=80000, hid=128, l=4)
daisy_config = dict(name='daisy-80s-20cpu', penalty_lr=0.01, cost_lim=20, gamma=0.99, lam=0.95, seed=4444, steps=80000, hid=128, l=4)
daphne_config = dict(name='daphne-80s-20cpu', penalty_lr=0.01, cost_lim=20, gamma=0.99, lam=0.95, seed=1248, steps=80000, hid=128, l=4)
durum_config = dict(name='durum-80s-20cpu', penalty_lr=0.01, cost_lim=20, gamma=0.99, lam=0.95, seed=2496, steps=80000, hid=128, l=4)


kilo_config = dict(name='kilo-80s-20cpu', penalty_lr=0.01, cost_lim=30, gamma=0.99, lam=0.95, seed=0, steps=80000, hid=128, l=4)
kefir_config = dict(name='kefir-80s-20cpu', penalty_lr=0.01, cost_lim=30, gamma=0.99, lam=0.95, seed=123, steps=80000, hid=128, l=4)
kompromat_config = dict(name='kompromat-80s-20cpu', penalty_lr=0.01, cost_lim=30, gamma=0.99, lam=0.95, seed=4444, steps=80000, hid=128, l=4)
kamikaze_config = dict(name='kamikaze-80s-20cpu', penalty_lr=0.01, cost_lim=30, gamma=0.99, lam=0.95, seed=1248, steps=80000, hid=128, l=4)
kennel_config = dict(name='kennel-80s-20cpu', penalty_lr=0.01, cost_lim=30, gamma=0.99, lam=0.95, seed=2496, steps=80000, hid=128, l=4)


paisley_config = dict(name='paisley-80s-20cpu', penalty_lr=0.01, cost_lim=40, gamma=0.99, lam=0.95, seed=0, steps=80000, hid=128, l=4)
pulchritude_config = dict(name='pulchritude-80s-20cpu', penalty_lr=0.01, cost_lim=40, gamma=0.99, lam=0.95, seed=123, steps=80000, hid=128, l=4)
pourpre_config = dict(name='pourpre-80s-20cpu', penalty_lr=0.01, cost_lim=40, gamma=0.99, lam=0.95, seed=4444, steps=80000, hid=128, l=4)
porpoise_config = dict(name='porpoise-80s-20cpu', penalty_lr=0.01, cost_lim=40, gamma=0.99, lam=0.95, seed=1248, steps=80000, hid=128, l=4)
pallid_config = dict(name='pallid-80s-20cpu', penalty_lr=0.01, cost_lim=40, gamma=0.99, lam=0.95, seed=2496, steps=80000, hid=128, l=4)


sage_config = dict(name='sage-80s-20cpu', penalty_lr=0.01, cost_lim=50, gamma=0.99, lam=0.95, seed=0, steps=80000, hid=128, l=4)
sorrel_config = dict(name='sorrel-80s-20cpu', penalty_lr=0.01, cost_lim=50, gamma=0.99, lam=0.95, seed=123, steps=80000, hid=128, l=4)
senna_config = dict(name='senna-80s-20cpu', penalty_lr=0.01, cost_lim=50, gamma=0.99, lam=0.95, seed=4444, steps=80000, hid=128, l=4)
sable_config = dict(name='sable-80s-20cpu', penalty_lr=0.01, cost_lim=50, gamma=0.99, lam=0.95, seed=1248, steps=80000, hid=128, l=4)
scepter_config = dict(name='scepter-80s-20cpu', penalty_lr=0.01, cost_lim=50, gamma=0.99, lam=0.95, seed=2496, steps=80000, hid=128, l=4)


CONFIG_LIST = [
               # amaranth_config, 
                aster_config, armadillo_config, acacia_config, amaryllis_config,
                chrysanthemum_config, chlorophyll_config, carnation_config, camellia_config, carapace_config,
                dahlia_config, daffodil_config, daisy_config, daphne_config, durum_config,
                paisley_config, pulchritude_config, pourpre_config, porpoise_config, pallid_config,
                kilo_config, kefir_config, kompromat_config, kamikaze_config, kennel_config,
                sage_config, sorrel_config, senna_config, sable_config, scepter_config,
               ]


