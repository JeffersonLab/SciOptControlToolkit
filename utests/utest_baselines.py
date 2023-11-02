import unittest
from jlab_opt_control.drivers.run_openai import main

#  python jlab_opt_control/drivers/run_openai.py --env "DnC2s-Circle2D-Statefull-v0" --nsteps 1 --nepisodes 10000
#  python jlab_opt_control/drivers/run_openai.py --env "HalfCheetah-v4" --nsteps 1000 --nepisodes 10000
#  python jlab_opt_control/drivers/run_openai.py --env "Pendulum-v1" --nsteps 100 --nepisodes 10000

class MyTestCase(unittest.TestCase):
    def test_env_performance(self):
        #args = parser.parse_args(['--env',"DnC2s-Circle2D-Statefull-v0"])
        main()


if __name__ == '__main__':
    unittest.main()
