import unittest
from jlab_opt_control.drivers.run_continuous import main

#  python jlab_opt_control/drivers/run_continuous.py --env "DnC2s-Circle2D-Statefull-v0" --nsteps 1 --nepisodes 10000
#  python jlab_opt_control/drivers/run_continuous.py --env "HalfCheetah-v4" --nsteps 1000 --nepisodes 10000
#  python jlab_opt_control/drivers/run_continuous.py --env "Pendulum-v1" --nsteps 100 --nepisodes 10000

class MyTestCase(unittest.TestCase):
    def test_env_performance_circle2d(self):
        args = ['--env','DnC2s-Circle2D-Statefull-v0','--nepisodes','100']
        main(args)
    
    def test_ER(self):
        args = ['--env','Pendulum-v1','--nepisodes','100', '--nsteps','100', '--btype', 'ER-v0', '--bsize', '100000']
        main(args)

    def test_PER(self):
        args = ['--env','Pendulum-v1','--nepisodes','100', '--nsteps','100', '--btype', 'PER-v0', '--bsize', '100000']
        main(args)

    def test_env_performance_halfcheetah(self):
        args = ['--env','HalfCheetah-v4','--nepisodes','100']
        main(args)

    def test_env_performance_halfcheetah_PER(self):
        args = ['--env','HalfCheetah-v4','--nepisodes','100', '--btype', 'PER-v0']
        main(args)
    
if __name__ == '__main__':
    unittest.main()
