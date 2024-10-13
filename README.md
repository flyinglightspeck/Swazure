# Swazure
Swazure solves the missing sensor data using cooperation among FLSs.  It implements physical data independence by abstracting the physical characteristics of the sensors, making point cloud data independent of the sensor hardware.

Authors:  Hamed Alimohammadzadeh(halimoha@usc.edu) and Shahram Ghandeharizadeh (shahram@usc.edu)

## Run
Clone the repository and open it with Pycharm to set up a virtual environment and install the requirements.

Run `src/main.py` to generate results based on the default input parameters. Change the options as desired to run different solutions (shortest, averaging, move_blocking+, move_source+) for different shapes (chess, palm, dragon, kangaroo, skateboard, racecar).
```
usage: main.py [-h] [--shape {chess_408,palm_725,dragon_1147,kangaroo_972,skateboard_1372,racecar_3720}] [--alg {swazure}]
               [--solution {shortest,averaging,move_blocking,move_blocking+,move_source,move_source+}] [--scale SCALE] [--radius-beta RADIUS_BETA] [--radius RADIUS]
               [--sweet-range-min SWEET_RANGE_MIN] [--sweet-range-max SWEET_RANGE_MAX] [--decaying-range-min DECAYING_RANGE_MIN]
               [--decaying-range-max DECAYING_RANGE_MAX] [--accurate] [--dead-reckoning-angle DEAD_RECKONING_ANGLE] [--steps-threshold STEPS_THRESHOLD] [--weighted]
```

```
options:
  -h, --help            show this help message and exit
  --shape {chess_408,palm_725,dragon_1147,kangaroo_972,skateboard_1372,racecar_3720}
                        Name of the .xyz file in the src/assets directory.
  --alg {swazure}       Name of the algorithm to run.
  --solution {shortest,averaging,move_blocking,move_blocking+,move_source,move_source+}
                        Name of the solution to run. shortest: use the shortest shortest path, averaging: average the first two shotrtest paths when available,
                        move_blocking+: use the shortest path and move the blocking FLSs, move_source+: move the source to find a common sweet FLS.
  --scale SCALE         Scale factor for point cloud coordinates.
  --radius-beta RADIUS_BETA
                        Ratio of FLS radius to the minimum distance between FLSs.
  --radius RADIUS       Set radius explicitly (cm). If set to a non-zero value radius_beta will be ignored.
  --sweet-range-min SWEET_RANGE_MIN
                        Sweet range start. The maximum working range of the tracking device (cm).
  --sweet-range-max SWEET_RANGE_MAX
                        Sweet range end (cm).
  --decaying-range-min DECAYING_RANGE_MIN
                        Decaying range start (cm).
  --decaying-range-max DECAYING_RANGE_MAX
                        Decaying range end. The maximum working range of the tracking device (cm).
  --accurate            Use fully accurate measurements.
  --dead-reckoning-angle DEAD_RECKONING_ANGLE
                        Dead reckoning angle (degree).
  --steps-threshold STEPS_THRESHOLD
                        Maximum amount of steps the source FLS explores as a factor of its radius.
  --weighted            If passed, use euclidean distance as the weight in the shortest path computation. Otherwize use shortest hops.
```
