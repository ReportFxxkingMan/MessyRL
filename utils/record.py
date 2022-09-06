import gym
import tensorflow as tf
from gym.wrappers.monitoring.video_recorder import VideoRecorder

env = gym.make()

class RecordEpisode():

    def __init__(
        self,
        env,
        actor
    ):  
        """        
        Args:
            env (gym.Env): open-ai gym environment ex) bipedalwalkr-v3
            actor (tf.keras.models.Model): agent actor model has state as input and action array as output ex) pdpg.actor
        """
    
        self.env = env 
        self.actor = actor
    
    def record(self, path):
        """
        Recording an episode video 

        Args:
            path (str): Directory path for recorded video saving
        """
        state = env.reset()

        video_recorder = VideoRecorder(self.env, path, enabled=True)
        while not done:
            
            env.render()
            video_recorder.capture_frame()
            action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
            action = action.numpy()[0]
            
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            episode_reward += reward
            time += 1  

        video_recorder.close()
        video_recorder.enabled = False
        env.close()        
        
