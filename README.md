# Accelerating High-Capacity Ridepooling in Robo-Taxi Systems

This repository contains the code and data for the paper ['Accelerating High-Capacity Ridepooling in Robo-Taxi Systems'](https://arxiv.org/abs/2505.07776).

## Prerequisites
You will need to have a working Gurobi installation. If you are a student or academic, you can get a free license [here](https://www.gurobi.com/downloads/end-user-license-agreement-academic/). The code is built with python 3.10.14. Different versions of python may cause issues with some packages.

To install the required packages, run:
```bash
pip install -r requirements.txt
```
It is recommended to use a virtual environment. If you are using conda, you can create a new environment with:
```bashconda create -n {env_name} python=3.10
conda activate {env_name}
pip install -r requirements.txt
``` 
The pygurobi package is set to version 12.0.1 in the requirements.txt file. If you have a different version of Gurobi, you may need to change this.

## Data
The data used int the experiements is under the `data/` folder. The data is from the [NYC Taxi and Limousine Commission](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The request data is processed from the FHV trip record data on May 15, 2024. The road network data is from [OSMnx](https://osmnx.readthedocs.io/en/stable/). The processed map data is included in the `data/map/` folder.

To process the raw data on differnet days, you can refer to [this repository](https://github.com/mit-zardini-lab/scenario-generation.git) and follow the instructions in the README file.

## Running the Code
To run the baseline method, use `main.py'. The file accepts the following key arguments:
- `--DATAROOT`: Root directory to data files. (default: `./data`)
- `--REQUEST_DATA_FILE`: Path to request data file under DATAROOT. (default: `requests/requests.csv`)
- `VEHICLE_DATA_FILE`: Path to vehicle data file under DATAROOT. (default: `vehicles/vehicles.csv`)
- `--INITIAL_TIME`: Simulation start time in string format. (default: `00:00:00`)
- `--FINAL_TIME`: Simulation end time in string format. (default: `01:00:00`)
- `--INTERVAL`: Re-optimization interval in seconds. (default: `60`)
- `--CARSIZE`: Vehicle capacity. (default: 10)
- `--MAX_NEW`: Maximum number of new requests that can be assigned to a vehicle at each re-optimization. (default: 20)
- `--RTV_TIMELIMIT`: Time limit for generating the RTV at each re-optimization in seconds. (default: 0, which means no time limit)
- `--VEHICLE_LIMIT`: Fleet size. (default: 1000)
- `--LOG_FILE`: Path to log file. (default: `results.log`)

To run the basline method, use the following command:
```bash
python main.py --DATAROOT ./data --REQUEST_DATA_FILE requests/requests.csv --VEHICLE_DATA_FILE vehicles/vehicles.csv --INITIAL_TIME 08:00:00 --FINAL_TIME 09:00:00 --INTERVAL 60 --CARSIZE 10 --MAX_NEW 20 --RTV_TIMELIMIT 0 --VEHICLE_LIMIT 1000 --LOG_FILE results.log
```

To run the data-driven ILP method, use the argument `--ML` and set it to be 1. You will also need to provide the path to the pre-trained model using the argument `--MODEL_PATH`. The file accepts the following additional arguments:
- `--ML`: Whether to use the data-driven ILP method. (default:0) (set to 1 to use the data-driven ILP method)
- `--MODEL_PATH`: Path to the pre-trained model. (default: `weights/s2v/s2v_8_optimized.pt`)
To run the data-driven ILP method, use the following command:
```bash
python main.py --DATAROOT ./data --REQUEST_DATA_FILE requests/requests.csv --VEHICLE_DATA_FILE vehicles/vehicles.csv --INITIAL_TIME 08:00:00 --FINAL_TIME 09:00:00 --INTERVAL 60 --CARSIZE 10 --MAX_NEW 20 --RTV_TIMELIMIT 0 --VEHICLE_LIMIT 1000 --LOG_FILE results.log --ML 1 --MODEL_PATH weights/s2v/s2v_8_optimized.pt
```

To run the partition-based method, use the argument `--PARTITION`, `-VERSION`, and `PARTITION_K`. The file accepts the following additional arguments:
- `--PARTITION`: Partition algorithm to use. (default: None) (set to `METIS` to use the METIS partition algorithm, or `Modularity` to use the Modularity-based partition algorithm)
- `--PARTITION_K`: Number of partitions. (default: 3) (only matter for the METIS partition algorithm)
- `--VERSION`: Either to use the partition-based method. (default: 0)

To run the partition-based method with METIS, use the following command:
```bash
python main.py --DATAROOT ./data --REQUEST_DATA_FILE requests/requests.csv --VEHICLE_DATA_FILE vehicles/vehicles.csv --INITIAL_TIME 08:00:00 --FINAL_TIME 09:00:00 --INTERVAL 60 --CARSIZE 10 --MAX_NEW 20 --RTV_TIMELIMIT 0 --VEHICLE_LIMIT 1000 --LOG_FILE results.log --PARTITION METIS --PARTITION_K 3 --VERSION 1
```

To run the comibined method with METIS, use the following command:
```bash
python main.py --DATAROOT ./data --REQUEST_DATA_FILE requests/requests.csv --VEHICLE_DATA_FILE vehicles/vehicles.csv --INITIAL_TIME 08:00:00 --FINAL_TIME 09:00:00 --INTERVAL 60 --CARSIZE 10 --MAX_NEW 20 --RTV_TIMELIMIT 0 --VEHICLE_LIMIT 1000 --LOG_FILE results.log --ML 1 --MODEL_PATH weights/s2v/s2v_8_optimized.pt --PARTITION METIS --PARTITION_K 3 --VERSION 1
```


## Credits
This code is built upon the [OpenRidepoolSimulator](https://github.com/MAS-Research/OpenRidepoolSimulator). We thank the authors for their open-source code.