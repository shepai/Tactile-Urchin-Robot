import time
import math
import mujoco
import mujoco.viewer

def main():
    # 1. Load the XML model file
    model_path = "/home/dexter/Documents/GitHub/Tactile-Urchin-Robot/urchin/icosabot.xml"
    print(f"Loading model from {model_path}...")
    
    # Initialize the model structure and state variables
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print("Opening MuJoCo interactive viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        while viewer.is_running():
            step_start = time.time()  # Track the wall-clock time right before the step

            mujoco.mj_step(model, data)
            viewer.sync()
            
            # --- REAL-TIME SYNCHRONISATION ---
            # Calculate how much wall-clock time is left for this physics timestep
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
if __name__ == "__main__":
    main()
