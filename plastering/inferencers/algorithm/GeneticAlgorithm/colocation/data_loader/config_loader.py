"""Loading a config file
"""
import pathlib
from copy import deepcopy
from typing import List

import dataclasses
from terminaltables import AsciiTable

import rapidjson as json


@dataclasses.dataclass
class ColocationConfig:
    """Configuration for running a colocation optimizer
    """
    task: str = 'strict_ga'
    job_name: str = 'untitled_job'
    seed: int = 1
    corr_matrix_path: str = ''
    output_path: str = './__output__/'
    total_room_count: int = 51
    total_type_count: int = 4
    selected_rooms: List[int] = dataclasses.field(default_factory=list)
    room_count: int = 51
    type_count: int = 4
    population_count: int = 300
    replaced_count: int = 150
    survivor_count: int = 20
    max_iteration: int = 5000
    crossing_over_rate: float = 0.0
    mutation_rate: float = 0.001
    mutation_weighted: bool = False
    searching_count: int = 0
    verbose: bool = True
    print_final_solution: bool = True
    visualize: bool = True
    show_figure: bool = True
    save_figure: bool = False
    save_results: bool = True
    plot_fitness_density: bool = False
    plot_fitness_accuracy: bool = True
    profile: bool = False

    def __str__(self):
        data = [['key', 'value']] + [[k, v] for k, v in vars(self).items()]
        return AsciiTable(data, title='Config').table

    def to_json(self):
        """Generate a json string

        Returns:
            str: a string of json
        """
        return json.dumps(dataclasses.asdict(self), indent=4)

    def copy(self) -> 'ColocationConfig':
        """Return a copy of self
        """
        return deepcopy(self)

    @property
    def base_file_name(self):
        """Get the base file path
        """
        return self.output_path + self.job_name + '/'

    def join_name(self, name: str):
        """Join the name based on a base name of the config
        """
        path = pathlib.Path(self.output_path).joinpath(
            pathlib.Path(self.job_name))
        return path.joinpath(name)


def load_config(config_file_path: str) -> ColocationConfig:
    """Load a configuration json file

    Args:
        config_file_path (str): Config file's path
    """
    with open(config_file_path, 'rb') as file:
        config = json.load(file)

    # mypy cannot recognize dataclass correctly
    config = ColocationConfig(**config)  # type: ignore

    return config
