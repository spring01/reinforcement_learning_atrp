'''
Example: python sample_test.py atrpenv_gv.py ATRP-psnt-td-gv24-v0 ./output/ATRP-psnt-td-gv24-v0-run1/model_0.h5
'''
import os; os.environ['OMP_NUM_THREADS'] = '1'
import sys
from sample_train import make_env, state_to_input
from drlbox.evaluator import make_evaluator



def main():
    evaluator = make_evaluator('ac', env_maker=lambda: make_env(*sys.argv[1:3]),
                               state_to_input=state_to_input,
                               load_model=sys.argv[3],
                               num_episodes=20,
                               render_timestep=None,
                               render_end=False,
                               verbose=True)
    evaluator.run()



if __name__ == '__main__':
    main()
