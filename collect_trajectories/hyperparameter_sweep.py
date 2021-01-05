from run_policy_agent import *


# Experimentation
# We have two experimentation options: Experiment Grid and wandb hyperparameter sweep
hyperparameter_defaults = dict(
    hid = 64,
    l = 2,
    gamma = 0.99,
    cost_gamma = 0.99,
    seed = 0,
    cost_lim = 10,
    steps = 4000,
    epochs = 50,
    )


sweep_config = {
  "name": "First Sweep",
  "method": "grid",  # think about switching to bayes
  "parameters": {
        "hid": {
            "values": [64, 128, 256]
        },
        "l": {
            "values" : [1, 2, 3, 4]
        },
        "gamma": {
            "values": [ 0.98, 0.985, 0.99, 0.995]
            # "min": 0.98,
            # "max": 0.995
        },
        "seed": {
            "values" : [0, 99, 123, 456, 999]
        },
        # "cost_lim": {
        #     "min" : 0,
        #     "max" : 10
        # },
        "epochs": {
            "values" : [50, 100, 200]
        }
    }
}

def safe_ppo_train():
    run = wandb.init(project="safe-ppo-agent", config=hyperparameter_defaults)

    ppo(lambda: gym.make('Safexp-PointGoal1-v0'),
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[run.config.hid] * run.config.l),
        gamma=run.config.gamma, seed=run.config.seed,
        steps_per_epoch=4000,
        epochs=run.config.epochs,
        logger_kwargs=logger_kwargs)

    print("config:", dict(run.config))



sweep_id = wandb.sweep(sweep_config, entity="feloundou", project="safe-ppo-agent")
wandb.agent(sweep_id, function= safe_ppo_train)

wandb.finish()


# Experiment Grid
def test_eg():
    eg = ExperimentGrid()
    eg.add('test:a', [1, 2, 3], 'ta', True)
    eg.add('test:b', [1, 2, 3])
    eg.add('some', [4, 5])
    eg.add('why', [True, False])
    eg.add('huh', 5)
    eg.add('no', 6, in_name=True)
    return eg.variants()