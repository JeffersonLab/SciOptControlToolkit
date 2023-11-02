# Copyright (c) 2020, Jefferson Science Associates, LLC. All Rights Reserved. Redistribution
# and use in source and binary forms, with or without modification, are permitted as a
# licensed user provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# This material resulted from work developed under a United States Government Contract.
# The Government retains a paid-up, nonexclusive, irrevocable worldwide license in such
# copyrighted data to reproduce, distribute copies to the public, prepare derivative works,
# perform publicly and display publicly and to permit others to do so.
#
# THIS SOFTWARE IS PROVIDED BY JEFFERSON SCIENCE ASSOCIATES LLC "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
# JEFFERSON SCIENCE ASSOCIATES, LLC OR THE U.S. GOVERNMENT BE LIABLE TO LICENSEE OR ANY
# THIRD PARTES FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import unittest as unittest
import jlab_opt_control.agents as agents
import jlab_opt_control.envs as envs
import gymnasium as gym


class RegistryTests(unittest.TestCase):
    """
    Registry Test class to test all the registered modules are loaded properly.
    """

    def test_env(self):
        """
        Test each agent using a OpenAI gym env
        :return: No return value
        """
        registered_envs = envs.list_registered_modules()
        for env_id in registered_envs:
            print('Testing env:', env_id)
            env = envs.make(env_id)

    def test_continuous_agents(self):
        """
        Test each agent using a OpenAI gym env
        :return: No return value
        """
        env = gym.make('MountainCarContinuous-v0')
        registered_agents = agents.list_registered_modules()
        for agent_id in registered_agents:
            print('Continuous env test agent:', agent_id)
            agents.make(agent_id, env=env, logdir='./')

    def test_discrete_agents(self):
        """
        Test each agent using a OpenAI gym env
        :return: No return value
        """
        env = gym.make('CartPole-v0')
        registered_agents = agents.list_registered_modules()
        for agent_id in registered_agents:
            print('Discrete env test agent:', agent_id)
            agents.make(agent_id, env=env, logdir='./')


    def test_registered_agents(self):
        """
        """
        registered_agents = agents.list_registered_modules()
        print('Registered agents:', registered_agents)

    def test_registered_envs(self):
        """
        """
        registered_envs = envs.list_registered_modules()
        print('Registered envs:', registered_envs)


if __name__ == '__main__':
    unittest.main()
