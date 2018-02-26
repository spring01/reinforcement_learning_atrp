
from gym.envs.registration import register


def register_targets(kwargs_common, set_name, target_list):
    for name, target in target_list:
        kwargs_target = kwargs_common.copy()
        kwargs_target['target'] = target
        register(
            id='ATRP-' + set_name + '-' + name + '-v0',
            entry_point='simatrp:ATRPTargetDistribution',
            max_episode_steps=100000,
            kwargs=kwargs_target
        )
