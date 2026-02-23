import toml

from objects.material import Material


def read_materials(materials_file: str) -> list[str]:
    '''Read TOML material list and return a list with Material objects in the order in which the materials are listed in the TOML file
    INPUTS:
        materials_file: str -> path to the materials file
    OUTPUTS:
        list[str] -> list of Material objects
    '''

    # --------------------------------------------------------------------------
    
    material_definitions = toml.load(materials_file)
    material_dictionary  = {}
    
    for name, property_dictionary in material_definitions.items():

        material_dictionary[name] = Material(name, property_dictionary)

    return [material for material in material_dictionary.values()]


if __name__ == "__main__":

    materials_file = 'materials.toml'

    materials = read_materials(materials_file)
    
    print(materials)