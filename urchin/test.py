
import time
import math
import mujoco
import mujoco.viewer


def main():
    # 1. Load the XML model file
    model_path = "/home/dexter/Documents/GitHub/Tactile-Urchin-Robot/urchin/icosabot.xml"
    print(f"Loading model from {model_path}...")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # ---------------------------------------------------------
    # SINE WAVE SETTINGS
    # ---------------------------------------------------------

    amplitude = 0.5       # Maximum actuator control value
    frequency = 0.1       # Hz - cycles per second

    # Phase difference between adjacent actuators.
    # 2*pi / number_of_actuators gives one complete wave
    # around the actuators.
    num_actuators = model.nu
    phase_offset = 2 * math.pi / num_actuators

    print(f"Number of actuators: {num_actuators}")
    print(f"Amplitude: {amplitude}")
    print(f"Frequency: {frequency} Hz")

    print("Opening MuJoCo interactive viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            step_start = time.time()

            # Use simulation time rather than wall-clock time
            t = data.time

            # -------------------------------------------------
            # DRIVE EVERY ACTUATOR WITH A SINE WAVE
            # -------------------------------------------------

            for i in range(model.nu):

                # Each actuator gets the same sine wave,
                # but shifted in phase.
                phase = i * phase_offset

                data.ctrl[i] = amplitude * math.sin(
                    2 * math.pi * frequency * t + phase
                )

            # Advance the MuJoCo simulation
            mujoco.mj_step(model, data)

            # Update viewer
            viewer.sync()

            # -------------------------------------------------
            # REAL-TIME SYNCHRONISATION
            # -------------------------------------------------

            time_until_next_step = (
                model.opt.timestep - (time.time() - step_start)
            )

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

