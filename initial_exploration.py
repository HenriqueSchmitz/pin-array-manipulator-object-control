import mujoco
import mujoco.viewer
import numpy as np
import time

def generate_xml(grid_size=10, pin_spacing=0.06):
    """Generates a MuJoCo XML string for a grid of pins."""
    pins_xml = ""
    actuators_xml = ""
    
    for i in range(grid_size):
        for j in range(grid_size):
            name = f"pin_{i}_{j}"
            x = (i - grid_size/2) * pin_spacing
            y = (j - grid_size/2) * pin_spacing
            
            # Define the body, joint, and geom for each pin
            pins_xml += f"""
            <body name="{name}" pos="{x} {y} 0">
                <joint name="{name}_joint" type="slide" axis="0 0 1" range="-0.1 0.1" damping="10"/>
                <geom type="box" size="0.025 0.025 0.05" rgba="0.8 0.8 0.8 1"/>
            </body>"""
            
            # Define a position actuator for each pin
            actuators_xml += f'<position name="{name}_act" joint="{name}_joint" kp="2000"/>'

    xml = f"""
    <mujoco>
        <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>
        
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
            
            {pins_xml}
            
            <body name="object" pos="0 0 0.2">
                <joint type="free"/>
                <geom type="sphere" size="0.1" rgba="1 0 0 1" mass="0.5"/>
            </body>
        </worldbody>

        <actuator>
            {actuators_xml}
        </actuator>
    </mujoco>
    """
    return xml

# 1. Load the model
model = mujoco.MjModel.from_xml_string(generate_xml(grid_size=10, pin_spacing=0.05))
data = mujoco.MjData(model)

# 2. Launch the viewer and run the simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    wave_phase = 0.0
    wave_speed = 0.003
    direction = 1
    while viewer.is_running():
        step_start = time.time()
        ball_pos = data.body('object').xpos

        wave_phase += wave_speed * direction
        
        # 3. Control logic: Create a moving wave pattern
        change_direction = False
        for i in range(10):
            for j in range(10):
                actuator_id = i * 10 + j
                
                # Use the accumulated phase + spatial offset (i + j)
                # This creates a diagonal traveling wave
                target_height = 0.25 * np.sin(wave_phase + (i + j) * 0.5)
                data.ctrl[actuator_id] = target_height
                if ((actuator_id == 90 and direction == 1) or (actuator_id == 9 and direction == -1)) and target_height >= 0.20:
                    change_direction = True

        if change_direction:
            direction *= -1
        
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Maintain real-time sync
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)