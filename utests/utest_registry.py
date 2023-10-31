import unittest as unittest
import jlab_opt_control.agents as agents
import gymnasium as gym


class RegistryTests(unittest.TestCase):
    """
    Registry Test class to test all the registered modules are loaded properly.
    """

    def test_agents(self):
        """
        Test each agent using a OpenAI gym env
        :return: No return value
        """
        env = gym.make('MountainCarContinuous-v0')
        registered_agents = agents.list_registered_modules()
        for agent_id in registered_agents:
            print('Testing agent:', agent_id)
            agents.make(agent_id, env=env)

    def test_registered_agents(self):
        """
        """
        registered_agents = agents.list_registered_modules()
        print('Registered agents:', registered_agents)

if __name__ == '__main__':
    unittest.main()