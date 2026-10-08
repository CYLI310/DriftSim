# From training to the car: Jetson Orin Nano, BeamNG, hardware

The pipeline for the final model:

1. **Train** the safe, self-adapting sliding policy (preset `safe-adaptive`) on the Jetson's GPU, a
   Mac (MPS) or any CUDA PC: in the GUI or with `driftsim-train`.
2. **Export** the best policy: `final/` + `final_model.zip` with NumPy, ONNX and TorchScript versions
   of the same network, checked against each other and driven through the on-car runtime in DriftSim.
3. **Test in BeamNG** (Windows PC): the exact exported model, runtime and safety filter on a car in an
   independent physics engine.
4. **Drive the real car** with the same runtime on the Jetson, through your hardware driver.

Steps 2 to 4 all use `rc_drift_sim.deploy`: `PolicyRuntime` (readings in, filtered commands out, NumPy
only) and the 50 Hz loop `driftsim-drive`, so what you test is what drives the car.

## What the model is

| part | what it does |
|---|---|
| task | hold a commanded drift: \|sideslip\| 22-38° and speed 1.2-1.8 m/s (set per run with `--beta/--speed`), from a parked start, either direction |
| unknown low grip | trained on three hard floors with the grip scaled 0.3-1.3x (peak μ ≈ 0.09-0.46 with plastic drift tires), dust, wear, stiffness, mass, motor, steering offset and 20-60 ms latency random; in 30 % of the episodes the grip jumps 0.6-1.6x halfway (a wet or dusty patch) |
| self-adaptation | the policy sees the last 16 readings (0.32 s); a grip-estimator network learns the friction coefficient from the same readings and feeds it to the policy and the safety filter; the critic sees the true state during training only |
| safety filter | `deploy/safety.py`, identical in training and on the car: command rate limits, counter-steer + throttle cut beyond 50-70° sideslip or a yaw rate above 2 x grip x g / speed, speed limit 3 m/s; interventions are penalized in training |
| sensors | gyro z, accelerometer x/y, body velocity x/y (optical flow, e.g. SparkFun OTOS), rear (driven) wheel speed; no front encoders |
| selection | every 25 iterations the policy drives a grip sweep (×0.25 ... ×1.05); the best score is `best.pt` |

## 1. Train on the Jetson Orin Nano

JetPack 6.2 ships Python 3.10; NVIDIA's PyTorch for it comes from the Jetson AI Lab index (the `.io`
domain replaced the old `.dev` one). On the Jetson:

```bash
bash scripts/jetson/setup_jetson.sh
```

which does, step by step:

```bash
sudo apt-get install -y python3-venv python3-pip libopenblas-dev
python3 -m venv ~/driftsim-venv && source ~/driftsim-venv/bin/activate
pip install --upgrade pip
pip install torch==2.8.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install -e ".[export]"
python -c "import torch; print(torch.cuda.is_available())"     # must print True
```

If `torch.cuda.is_available()` is False, a generic (non-Jetson) torch got installed: uninstall every
torch and reinstall from the index above. Use the highest power mode and fixed clocks while training
(`sudo nvpmodel -q` shows the mode, the MAXN SUPER mode on JetPack 6.2 is the fastest;
`sudo jetson_clocks`).

Train headless (survives a closed SSH session inside `tmux`):

```bash
driftsim-train --preset safe-adaptive --device cuda --out rl_runs/final --export
```

or with the GUI on the Jetson, viewed from your laptop through an SSH tunnel (the server only listens
on the Jetson's localhost):

```bash
driftsim-gui --no-browser                     # on the Jetson
ssh -L 8765:127.0.0.1:8765 user@jetson        # on the laptop, then open http://127.0.0.1:8765
```

RL training → Presets → **Use** "Safe adaptive sliding on unknown low-friction surfaces" → Start.
Settings saved from the GUI (Save JSON) also run headless: `driftsim-train --config settings.json`.

The preset is 60 M steps with 4,096 cars. Check the speed first with a short run
(`--steps 2e6`) and lower `--envs` if the 8 GB of shared memory runs out. On an Apple M4 (MPS) the
preset runs at about 15k steps/s (60 M steps in about an hour); the Jetson has not been measured here.

What to watch (RL runs page): the evaluation score and spin-outs on the grip sweep, how often the
safety filter takes over (should fall towards a few percent), and the grip estimate error.

## 2. Export the final model

GUI: RL runs → the run → **Export final model**. Command line:

```bash
driftsim-export rl_runs/final            # best.pt if present, else policy.pt
```

`final/policy.json` documents the inputs (order, scaling, history), targets, control period, safety
settings, car geometry and the evaluation. The export checks NumPy, TorchScript and ONNX against the
trained network and the GUI then drives the exported model through the runtime in DriftSim at three
grip levels (`final/sil.json`). Repeat that check by hand, also with a grip change:

```bash
driftsim-drive --model rl_runs/final/final --car sim --grip 0.4 --grip-change 0.6@5 --seconds 12
```

## 3. Test in BeamNG on your PC

BeamNG runs on Windows. The test drives the **exported model through the same runtime and safety
filter as the car**, in BeamNG's physics instead of DriftSim's.

**Set up the PC once** (Python 3.10-3.12 from python.org, then in the DriftSim folder):

```bash
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_windows.ps1
```

It makes `.venv`, installs PyTorch (the CUDA build if the PC has an NVIDIA GPU, so the PC can also
train), DriftSim with ONNX export and `beamngpy`, and looks for BeamNG in your Steam libraries
(`steamapps/common/BeamNG.drive`) and the usual BeamNG.tech folders. `beamngpy` must match the
game version (BeamNG 0.39 → beamngpy 1.36, 0.38 → 1.35.1, 0.37 → 1.34.1):
`.venv\Scripts\pip install beamngpy==<version>`.

**Get the model onto the PC.** Either copy the whole run folder from the Jetson (`rl_runs/<run>`,
including runs made with `driftsim-train`) into the PC's `DriftSim/rl_runs/`, or copy only
`final_model.zip`.

**Run the test from the GUI.** `scripts\windows\DriftSim GUI (Python).bat` → RL runs → your run →
Export final model (if not done) → **Test in BeamNG**: BeamNG starts by itself, the car spawns on
the flat grid, calibrates its steering, then the model drives; the card shows the live speed,
sideslip, commands, grip estimate and safety-filter activity and, at the end, a summary and the CSV
log (`final/drive_beamng.csv`).

**Or from the command line** (also works with just the zip):

```bash
.venv\Scripts\driftsim-drive --model final_model.zip --car beamng --seconds 20 --beta 30 --speed 1.5
```

Options: `--beamng-home` (if BeamNG is not found), `--vehicle` / `--part-config` (the car),
`--level`, `--scale`, `--steer-lock-deg`, `--throttle-gain`, `--gear`, `--wheel-key`.

About BeamNG.drive from Steam: `beamngpy` is made for BeamNG.tech (free for research and
academic use on request); the Steam BeamNG.drive works with it in part. The driver only uses the
basic calls (vehicle state, the Electrics sensor, `vehicle.control`, deterministic stepping). It
was tested here against a stand-in that mimics those calls with DriftSim physics, not against the
real game, so treat the first run as a shake-down. If `beamngpy` cannot start the game, start
BeamNG yourself with the launch options `-tcom -tport 25252` (Steam → BeamNG.drive → Properties →
Launch options) and add `--no-launch` (command line) to connect to it.

How the test works:

* **Scale.** BeamNG has no 1/10 RC car (community RC mods exist, but small cars are unstable in its
  physics). A full-size car is driven at the same Froude number: with k = car length / RC car length,
  speeds are divided by √k, yaw rates multiplied by √k, angles and accelerations are unchanged, and
  the control period becomes 20 ms × √k. `scale auto` reads the length from the vehicle's bounding
  box; `scale 1` drives an RC-size mod as it is.
* **Pick a rear-drive car with low grip.** A plastic-tire RC drift car behaves like a full-size car
  at about μ 0.3. BeamNGpy has no ground-friction setting, so choose a rear-drive configuration with
  slippery tires (drift, worn or winter tires on asphalt) in the vehicle's parts. On dry asphalt with
  road tires the car is far outside what the policy learned.
* **Steering** is calibrated at the start (a short steer test decides which BeamNG input turns left,
  whatever BeamNG's sign conventions). `steer lock` is the car's road-wheel angle at full input if it
  differs from the trained 35°. `throttle gain` scales the throttle (an engine is not an RC motor);
  `--gear 1` if the car does not move.
* **Wheel speed.** The driven-wheel speed comes from the Electrics value `wheelspeed`; the start-up
  message lists the wheel/speed values your version has (`--wheel-key` picks another).

What a pass means: the car holds a slide near the commanded sideslip without spinning and the filter
rarely acts. A fail means the policy does not transfer to these dynamics; it does not by itself mean
the real car fails, but read the log. Either way, identify the real car's parameters (system
identification) before the real run.

## 4. The real car

Write a driver with three methods (template: `deploy.car_loop.HardwareCar`):

```python
from rc_drift_sim.deploy.car_loop import CarInterface

class MyCar(CarInterface):
    def read(self):        # SI units, body frame (x forward, y left, counter-clockwise +)
        return dict(gyro_z=..., accel_x=..., accel_y=..., vel_x=..., vel_y=..., wheel_rear=...)
    def write(self, steer, throttle):   # [-1, 1]; steer +1 = full left lock
        ...                              # servo / ESC pulses (PCA9685 board or Jetson PWM)
    def armed(self):                     # dead-man switch: False -> neutral, policy restarts
        return transmitter_switch_on()
```

```bash
driftsim-drive --model final --car mycar:MyCar --beta 30 --speed 1.5 --seconds 15 \
    --car-config car.json --safety '{"v_max": 2.0}'
```

`car.json` holds the real car's `steer_max_deg` and `cg_to_front` for the safety filter. Start with
the wheels off the ground (check that +steer turns left and the commands are sane), then a large open
floor, low `v_max` and a hand on the dead-man switch. The loop keeps real time and counts overruns;
the NumPy network takes well under 1 ms per step.

Without DriftSim on the car: `policy_ts.pt` (`torch.jit.load`) or `policy.onnx` (onnxruntime /
TensorRT) map the observation to (action, grip); build the observation and apply the safety filter
exactly as `deploy/runtime.py` does (its README.txt lists the inputs).
