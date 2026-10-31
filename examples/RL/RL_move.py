from RL_base import run, visualise, check_prior
 
ENV_NAME = "Urchin"
MODEL_NAME = "urchin_v2"      # -> models/urchin_v2.zip
LOG_NAME = "urchin_v2"        # -> logs/urchin_v2/progress.csv
STEPS = 5_000_000
MAX_SPEED = 0.05              # m/s - set this from check_prior()'s output
 
if __name__ == "__main__":
    check_prior(ENV_NAME)     # quick: does the untrained sparse gait roll the right way?
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, max_speed=MAX_SPEED, record_video=True)
    visualise(ENV_NAME, MODEL_NAME)
 