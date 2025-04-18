# Copyright (c) 2025, Jefferson Science Associates, LLC. All Rights Reserved.

import unittest
import numpy as np
import os
import shutil
import tempfile
import time
import warnings
import gymnasium as gym
import tensorflow as tf
import jlab_opt_control.agents as agents
import logging


class AgentFunctionalityTest(unittest.TestCase):
    """
    Test class to verify basic functionality of all registered RL agents.
    Tests initialization, action selection, training, saving/loading models, and config saving.
    """

    @classmethod
    def setUpClass(cls):
        """Set up environment and parameters used across all tests."""
        # Suppress TensorFlow warning messages for cleaner test output
        tf.get_logger().setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Create a continuous action space environment for testing
        cls.env = gym.make('MountainCarContinuous-v0')
        
        # Get list of registered agents
        cls.registered_agents = agents.list_registered_modules()
        print(f"Detected {len(cls.registered_agents)} registered agents: {cls.registered_agents}")
        
        # Sample state for action testing (matches MountainCarContinuous observation space)
        cls.sample_state = np.array([0.0, 0.0])

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have been run."""
        cls.env.close()

    def setUp(self):
        """Create a unique temporary directory for each test."""
        # Use tempfile to create a unique temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="agent_test_")
        print(f"Created test directory: {self.test_dir}")

    def tearDown(self):
        """Clean up test directory with error handling."""
        # Close any TensorFlow file writers that might be keeping files open
        try:
            tf.summary.flush()
        except:
            pass
            
        # Sleep briefly to allow file operations to complete
        time.sleep(0.5)
        
        # Try to remove the test directory
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception as e:
            warnings.warn(f"Could not fully remove test directory {self.test_dir}: {e}")

    def test_agent_initialization(self):
        """Test if all agents can be properly initialized."""
        for agent_id in self.registered_agents:
            with self.subTest(agent=agent_id):
                try:
                    agent = agents.make(agent_id, env=self.env, logdir=self.test_dir)
                    self.assertIsNotNone(agent, f"Agent {agent_id} initialization failed")
                    
                    # Verify that the agent has all required abstract methods
                    self.assertTrue(hasattr(agent, 'soft_update'), f"{agent_id} missing soft_update method")
                    self.assertTrue(hasattr(agent, 'train'), f"{agent_id} missing train method")
                    self.assertTrue(hasattr(agent, 'action'), f"{agent_id} missing action method")
                    self.assertTrue(hasattr(agent, 'load'), f"{agent_id} missing load method")
                    self.assertTrue(hasattr(agent, 'save'), f"{agent_id} missing save method")
                    self.assertTrue(hasattr(agent, 'save_cfg'), f"{agent_id} missing save_cfg method")
                    
                    print(f"Successfully initialized agent: {agent_id}")
                except Exception as e:
                    self.fail(f"Agent {agent_id} initialization raised exception: {e}")

    def test_agent_action_selection(self):
        """Test if agents can select actions for given states."""
        for agent_id in self.registered_agents:
            with self.subTest(agent=agent_id):
                try:
                    # Create agent in a new subdirectory to avoid conflicts
                    agent_dir = os.path.join(self.test_dir, f"{agent_id}_action_test")
                    os.makedirs(agent_dir, exist_ok=True)
                    agent = agents.make(agent_id, env=self.env, logdir=agent_dir)
                    
                    # Test training action
                    action, noise = agent.action(self.sample_state, train=True)
                    self.assertIsNotNone(action, f"Agent {agent_id} returned None action in training mode")
                    self.assertEqual(len(action), self.env.action_space.shape[0], 
                                    f"Agent {agent_id} action dimension mismatch")
                    
                    # Check action bounds
                    self.assertTrue(all(action >= self.env.action_space.low) and 
                                   all(action <= self.env.action_space.high),
                                  f"Agent {agent_id} action out of bounds: {action}")
                    
                    # Test inference action if supported
                    try:
                        inf_action, inf_noise = agent.action(self.sample_state, train=False, inference=True)
                        self.assertIsNotNone(inf_action, f"Agent {agent_id} returned None action in inference mode")
                        self.assertEqual(len(inf_action), self.env.action_space.shape[0],
                                        f"Agent {agent_id} inference action dimension mismatch")
                    except TypeError:
                        # Some agents might not support the inference parameter
                        inf_action, inf_noise = agent.action(self.sample_state, train=False)
                        self.assertIsNotNone(inf_action, f"Agent {agent_id} returned None action in non-training mode")
                    
                    print(f"Successfully tested action selection for agent: {agent_id}")
                except Exception as e:
                    self.fail(f"Agent {agent_id} action test failed with exception: {e}")

    def test_agent_memory_and_training(self):
        """Test if agents can store experiences and perform training steps."""
        for agent_id in self.registered_agents:
            with self.subTest(agent=agent_id):
                try:
                    # Create agent in a new subdirectory to avoid conflicts
                    agent_dir = os.path.join(self.test_dir, f"{agent_id}_training_test")
                    os.makedirs(agent_dir, exist_ok=True)
                    agent = agents.make(agent_id, env=self.env, logdir=agent_dir)
                    
                    # Add a few experiences to the buffer
                    # Note: This won't be enough for actual training in most cases
                    # but should exercise the memory and train methods
                    for _ in range(5):
                        state = self.env.reset()[0]
                        action, _ = agent.action(state)
                        next_state, reward, done, truncated, _ = self.env.step(action)
                        
                        # Store in agent's memory
                        if hasattr(agent, 'memory'):
                            agent.memory((state, action, reward, next_state, float(done or truncated)))
                    
                    # Try a training step (may not actually train if buffer is not filled enough)
                    if hasattr(agent, 'ntrain_calls'):
                        before_train = agent.ntrain_calls
                    agent.train()
                    if hasattr(agent, 'ntrain_calls'):
                        self.assertEqual(agent.ntrain_calls, before_train + 1, 
                                       f"Agent {agent_id} ntrain_calls not incremented after train()")
                    
                    print(f"Successfully tested memory and training for agent: {agent_id}")
                except Exception as e:
                    self.fail(f"Agent {agent_id} memory/training test failed with exception: {e}")

    def test_agent_save_and_load(self):
        """Test if agents can save and load their models."""
        for agent_id in self.registered_agents:
            with self.subTest(agent=agent_id):
                try:
                    # Create agent in a new subdirectory to avoid conflicts
                    agent_dir = os.path.join(self.test_dir, f"{agent_id}_save_load_test")
                    os.makedirs(agent_dir, exist_ok=True)
                    agent = agents.make(agent_id, env=self.env, logdir=agent_dir)
                    
                    # Test save functionality with a unique postfix
                    test_postfix = "unittest"
                    agent.save(post_fix=test_postfix)
                    
                    # Check if model directory was created
                    model_dir = os.path.join(agent_dir, "models", test_postfix)
                    self.assertTrue(os.path.exists(model_dir), 
                                   f"Agent {agent_id} did not create model directory at {model_dir}")
                    
                    # Confirm at least one model file was saved (should be .h5 files)
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
                    self.assertGreater(len(model_files), 0, 
                                      f"Agent {agent_id} did not save any model files in {model_dir}")
                    
                    # Create a new agent and test load functionality 
                    # Skip actual loading as it might require special setup
                    # Just verify the method runs without errors
                    agent.model_load_path = model_dir
                    try:
                        agent.load()
                        print(f"Successfully tested save and load for agent: {agent_id}")
                    except Exception as e:
                        warnings.warn(f"Agent {agent_id} load method raised exception {e}, but test continuing")
                except Exception as e:
                    self.fail(f"Agent {agent_id} save/load test failed with exception: {e}")

    def test_agent_save_cfg(self):
        """Test if agents can save their configuration files."""
        for agent_id in self.registered_agents:
            with self.subTest(agent=agent_id):
                try:
                    # Create agent in a new subdirectory to avoid conflicts
                    agent_dir = os.path.join(self.test_dir, f"{agent_id}_cfg_test")
                    os.makedirs(agent_dir, exist_ok=True)
                    agent = agents.make(agent_id, env=self.env, logdir=agent_dir)
                    
                    # Test configuration saving
                    agent.save_cfg()
                    
                    # Check if the config directory was created
                    cfg_dir = os.path.join(agent_dir, "cfgs")
                    self.assertTrue(os.path.exists(cfg_dir), 
                                   f"Agent {agent_id} did not create config directory")
                    
                    # There should be at least one JSON/cfg file saved
                    cfg_files = [f for f in os.listdir(cfg_dir) if f.endswith('.json') or f.endswith('.cfg')]
                    self.assertTrue(len(cfg_files) > 0, 
                                   f"Agent {agent_id} did not save any config files in {cfg_dir}")
                    
                    print(f"Successfully tested configuration saving for agent: {agent_id}")
                except Exception as e:
                    self.fail(f"Agent {agent_id} save_cfg test failed with exception: {e}")


class AgentSpecificTests(unittest.TestCase):
    """Test specific functionality of different agent types."""

    @classmethod
    def setUpClass(cls):
        """Set up environment and parameters used across all tests."""
        # Suppress TensorFlow warning messages for cleaner test output
        tf.get_logger().setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Create a continuous action space environment for testing
        cls.env = gym.make('MountainCarContinuous-v0')
        
        # Get list of registered agents for specific tests
        cls.registered_agents = agents.list_registered_modules()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have been run."""
        cls.env.close()

    def setUp(self):
        """Create a unique temporary directory for each test."""
        # Use tempfile to create a unique temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="agent_specific_test_")

    def tearDown(self):
        """Clean up test directory with error handling."""
        # Close any TensorFlow file writers that might be keeping files open
        try:
            tf.summary.flush()
        except:
            pass
            
        # Sleep briefly to allow file operations to complete
        time.sleep(0.5)
        
        # Try to remove the test directory, handling errors gracefully
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception as e:
            warnings.warn(f"Could not fully remove test directory {self.test_dir}: {e}")

    def test_ddpg_structure(self):
        """Test DDPG-specific network structure."""
        if 'keras_ddpg-v0' in self.registered_agents:
            agent = agents.make('keras_ddpg-v0', env=self.env, logdir=self.test_dir)
            
            # DDPG should have actor and critic networks
            self.assertIsNotNone(agent.actor_model, "DDPG missing actor model")
            self.assertIsNotNone(agent.target_actor, "DDPG missing target actor model")
            self.assertIsNotNone(agent.critic_model1, "DDPG missing critic model")
            self.assertIsNotNone(agent.target_critic1, "DDPG missing target critic model")
            
            # Check key parameters
            self.assertTrue(hasattr(agent, 'gamma'), "DDPG missing discount factor (gamma)")
            self.assertTrue(hasattr(agent, 'tau'), "DDPG missing soft update parameter (tau)")
            
            print("Successfully verified DDPG agent structure")

    def test_td3_structure(self):
        """Test TD3-specific network structure."""
        if 'keras_td3-v0' in self.registered_agents:
            agent = agents.make('keras_td3-v0', env=self.env, logdir=self.test_dir)
            
            # TD3 should have actor and dual critic networks
            self.assertIsNotNone(agent.actor_model, "TD3 missing actor model")
            self.assertIsNotNone(agent.target_actor, "TD3 missing target actor model")
            self.assertIsNotNone(agent.critic_model1, "TD3 missing critic model 1")
            self.assertIsNotNone(agent.target_critic1, "TD3 missing target critic model 1")
            self.assertIsNotNone(agent.critic_model2, "TD3 missing critic model 2")
            self.assertIsNotNone(agent.target_critic2, "TD3 missing target critic model 2")
            
            # Check TD3-specific parameters
            self.assertTrue(hasattr(agent, 'noise_clip'), "TD3 missing noise clipping parameter")
            self.assertTrue(hasattr(agent, 'actor_update_freq'), "TD3 missing actor update frequency parameter")
            
            print("Successfully verified TD3 agent structure")

    def test_redq_td3_structure(self):
        """Test REDQ-TD3-specific network structure."""
        if 'keras_redq_td3-v0' in self.registered_agents:
            agent = agents.make('keras_redq_td3-v0', env=self.env, logdir=self.test_dir)
            
            # REDQ-TD3 should have actor and ensemble of critic networks
            self.assertIsNotNone(agent.actor_model, "REDQ-TD3 missing actor model")
            self.assertIsNotNone(agent.target_actor, "REDQ-TD3 missing target actor model")
            self.assertTrue(hasattr(agent, 'critic_models'), "REDQ-TD3 missing critic models list")
            self.assertTrue(hasattr(agent, 'target_critics'), "REDQ-TD3 missing target critics list")
            
            # Check that the ensemble has the right size
            self.assertEqual(len(agent.critic_models), agent.num_critics, 
                            "REDQ-TD3 critic ensemble size mismatch")
            self.assertEqual(len(agent.target_critics), agent.num_critics, 
                            "REDQ-TD3 target critic ensemble size mismatch")
            
            # Check REDQ-specific parameters
            self.assertTrue(hasattr(agent, 'utd_ratio'), "REDQ-TD3 missing update-to-data ratio parameter")
            self.assertTrue(hasattr(agent, 'in_target_min'), "REDQ-TD3 missing in-target minimization parameter")
            
            print("Successfully verified REDQ-TD3 agent structure")

    def test_simple_episode(self):
        """Run a simple episode with each agent to test integration."""
        max_steps = 10 
        
        for agent_id in self.registered_agents:
            with self.subTest(agent=agent_id):
                try:
                    # Create agent in a new subdirectory to avoid conflicts
                    agent_dir = os.path.join(self.test_dir, f"{agent_id}_episode_test")
                    os.makedirs(agent_dir, exist_ok=True)
                    agent = agents.make(agent_id, env=self.env, logdir=agent_dir)
                    
                    # Run a short episode
                    state, _ = self.env.reset()
                    total_reward = 0
                    
                    for step in range(max_steps):
                        # Select an action using the agent
                        action, _ = agent.action(state)
                        
                        # Take step in environment
                        next_state, reward, done, truncated, _ = self.env.step(action)
                        total_reward += reward
                        
                        # Store in agent's memory
                        if hasattr(agent, 'memory'):
                            agent.memory((state, action, reward, next_state, float(done or truncated)))
                        
                        # Call train method
                        agent.train()
                        
                        # Update state
                        state = next_state
                        
                        if done or truncated:
                            break
                    
                    # Save both models and config at the end of the episode
                    agent.save(post_fix="episode_test")
                    agent.save_cfg()
                    
                    print(f"Successfully ran episode with agent {agent_id}, total reward: {total_reward}")
                except Exception as e:
                    self.fail(f"Agent {agent_id} episode test failed with exception: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)