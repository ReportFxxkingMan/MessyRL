from models.qr_dqn.main import qrdqn_main


game_name = "CartPole-v1"
params_dict = {
    "gamma" : 0.99,
    "batch_size" : 8,
    "lr" : 1e-4,
    "atoms" : 8,
}

qrdqn_main(
    game_name = game_name, 
    params_dict = params_dict,
)
