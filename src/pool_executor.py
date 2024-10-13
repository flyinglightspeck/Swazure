import concurrent.futures
import itertools
import subprocess
import logging


pool_size = 60

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_script(script_with_args):
    script, args = script_with_args
    logger.info(f"Starting {script}")
    try:
        result = subprocess.run(["python3", script] + args, check=True, capture_output=True, text=True)
        logger.info(f"Finished {script} with output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script} failed with error: {e.stderr}")
    except Exception as e:
        logger.error(f"An error occurred while running {script}: {str(e)}")


props_set = [
    # [
    #     {
    #         "keys": ["--solution"],
    #         "values": ["move_blocking"]
    #     },
    #     {
    #         "keys": ["--shape", "--scale"],
    #         "values": [
    #             {"--shape": "chess_408", "--scale": "13.6"},
    #             {"--shape": "palm_725", "--scale": "34"},
    #             {"--shape": "kangaroo_972", "--scale": "34"},
    #             {"--shape": "dragon_1147", "--scale": "40"},
    #             {"--shape": "skateboard_1372", "--scale": "34"},
    #             {"--shape": "racecar_3720", "--scale": "40"},
    #         ],
    #     },
    #     {
    #         "keys": ["--radius_beta"],
    #         "values": ["0.5", "0.4", "0.3", "0.2", "0.1", "0.01"]
    #     },
    #     {
    #         "keys": [" "],
    #         "values": ["--weighted"]
    #     },
    # ],
    [
        {
            "keys": ["--solution"],
            "values": ["move_blocking+"]
        },
        {
            "keys": ["--shape", "--scale"],
            "values": [
                {"--shape": "chess_408", "--scale": "13.6"},
                {"--shape": "palm_725", "--scale": "34"},
                {"--shape": "kangaroo_972", "--scale": "34"},
                {"--shape": "dragon_1147", "--scale": "40"},
                {"--shape": "skateboard_1372", "--scale": "34"},
            ],
        },
        {
            "keys": ["--radius_beta"],
            "values": ["0.4", "0.3"]
        },
        {
            "keys": [" "],
            "values": [" ", "--weighted"]
        },
    ],
    [
        {
            "keys": ["--solution"],
            "values": ["move_source+",]
        },
        {
            "keys": ["--shape", "--scale"],
            "values": [
                {"--shape": "chess_408", "--scale": "13.6"},
                {"--shape": "palm_725", "--scale": "34"},
                {"--shape": "kangaroo_972", "--scale": "34"},
                {"--shape": "dragon_1147", "--scale": "40"},
                {"--shape": "skateboard_1372", "--scale": "34"},
            ],
        },
        {
            "keys": ["--radius_beta"],
            "values": ["0.4", "0.3"]
        },
        {
            "keys": ["--steps_threshold"],
            "values": ["2", "10"]
        },
        {
            "keys": [" "],
            "values": ["--weighted", " "]
        },
    ],
]


def strip_and_add(args, arg):
    s_arg = arg.strip()
    if s_arg != "":
        args.append(arg)


if __name__ == "__main__":
    scripts = []
    for props in props_set:
        props_values = [p["values"] for p in props]
        combinations = list(itertools.product(*props_values))

        for j in range(len(combinations)):
            c = combinations[j]
            conf = []
            for i in range(len(c)):
                for k in props[i]["keys"]:
                    strip_and_add(conf, k)
                    if isinstance(c[i], dict):
                        strip_and_add(conf, c[i][k])
                    else:
                        strip_and_add(conf, c[i])
            scripts.append(["main.py", conf])

    logger.info(f"Running {len(scripts)} scripts in a pool of {pool_size} processes.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as executor:
        results = executor.map(run_script, scripts)

    logger.info("All scripts have completed.")
