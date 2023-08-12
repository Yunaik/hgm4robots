import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 2] # 0 is x, 1 is y, 2 is z
        
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * (height > 0.8) 

        done = ~not_done
        done = done[:,None]
        return done
