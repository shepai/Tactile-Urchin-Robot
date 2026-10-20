from RL_base import run, visualise
 
ENV_NAME = "Urchin"
MODEL_NAME = "urchin_v2"      # -> models/urchin_v1.zip
LOG_NAME = "urchin_v2"        # -> logs/urchin_v1/progress.csv
STEPS = 1_000_000
 
if __name__ == "__main__":
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, record_video=True)
    visualise(ENV_NAME, MODEL_NAME)
