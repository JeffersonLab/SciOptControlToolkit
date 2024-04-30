import unittest
from jlab_opt_control.drivers.run_continuous import main

class MyTestCase(unittest.TestCase):

    def test_sindy_td3(self):
        args = ['--agent', 'KerasSINDyTD3-v0']
        main(args)

if __name__ == '__main__':
    unittest.main()