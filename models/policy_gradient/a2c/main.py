import gym
from models.policy_gradient.a2c.learn import A2Cagent


def main():
    """
    에이전트를 학습하고 결과를 Visualization
    """
    max_episode_num = 1000  # 최대 에피소드 설정
    env_name = "Pendulum-v1"
    env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
    agent = A2Cagent(env)  # A2C 에이전트 객체

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()


if __name__ == "__main__":
    main()
