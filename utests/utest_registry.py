import unittest2 as unittest
import jlab_rl.agents as agents

class RegistryTests(unittest.TestCase):
    """
    Registry Test class to test all the registered modules are loaded properly.
    """

    def test_agents(self):
        """
        """
        registered_agents = agents.list_registered_modules()
        print('registered_agents:', registered_agents)

if __name__ == '__main__':
    unittest.main()