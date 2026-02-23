from typing import Any
from typing import Union

class Material:
    def __init__(self, name: str, properties: dict[str, Union[int, float]]) -> None:
        '''Initializes the material class
        INPUTS:
            name: str -> name of the material
            properties: dict -> dictionary containing the material properties where key is string and value is int or float
        OUTPUTS:
            None
        '''
        
        # Material name
        self.name = name
        
        # Ensure all Material properties have been provided
        assert all([property in properties.keys() for property in ['E', 'rho', 'alpha']]), f"Material {name} is badly defined. Ensure that you have specfied its Young's modulus 'E', density 'rho' and thermal expansion coefficient 'alpha'"

        # Assign material properties
        self.E     = properties['E']
        self.rho   = properties['rho']
        self.alpha = properties['alpha']

    def __getitem__(self, key:str) -> Union[int, float]:
        '''Allow users to retrieve values from the Material's dictionary
        INPUTS:
            key: str -> key of the dictionary
        OUTPUTS:
            int | float -> value of the dictionary
        '''
        return self.__dict__[key]
    
    def __setitem__(self, key: Any, value: Any) -> None:
        '''Allow users to assign values to items in the Mesh's dictionary
        INPUTS:
            key: Any -> key of the dictionary
            value: Any -> value of the dictionary
        OUTPUTS:
            None
        '''
        self.__dict__[key] = value