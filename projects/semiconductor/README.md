## Description

Innovative materials design is needed to tackle some of the most important health, environmental, energy, social, and economic challenges of this century. In particular, improving the properties of materials that are intrinsically connected to the generation and utilization of energy is crucial if we are to mitigate environmental damage due to a growing global demand. Transparent conductors are an important class of compounds that are both electrically conductive and have a low absorption in the visible range, which are typically competing properties. A combination of both of these characteristics is key for the operation of a variety of technological devices such as photovoltaic cells, light-emitting diodes for flat-panel displays, transistors, sensors, touch screens, and lasers. However, only a small number of compounds are currently known to display both transparency and conductivity suitable enough to be used as transparent conducting materials.

Aluminum (Al), gallium (Ga), indium (In) sesquioxides are some of the most promising transparent conductors because of a combination of both large bandgap energies, which leads to optical transparency over the visible range, and high conductivities. These materials are also chemically stable and relatively inexpensive to produce. Alloying of these binary compounds in ternary or quaternary mixtures could enable the design of a new material at a specific composition with improved properties over what is current possible. These alloys are described by the formula ; where x, y, and z can vary but are limited by the constraint x+y+z = 1. 

## Task

The task for this competition is to predict two target properties:

- Formation energy (an important indicator of the stability of a material)
- Bandgap energy (an important property for optoelectronic applications)

## Data Description
High-quality data are provided for 3,000 materials that show promise as transparent conductors. The following information has been included:

Spacegroup (a label identifying the symmetry of the material)
Total number of Al, Ga, In and O atoms in the unit cell )
Relative compositions of Al, Ga, and In (x, y, z)
Lattice vectors and angles: lv1, lv2, lv3 (which are lengths given in units of angstroms ( meters) and α, β, γ (which are angles in degrees between 0° and 360°)
A domain expert will understand the physical meaning of the above information but those with a data mining background may simply use the data as input for their models.


## File Descriptions
Note: For each line of the CSV file, the corresponding spatial positions of all of the atoms in the unit cell (expressed in Cartesian coordinates) are provided as a separate file.

train.csv - contains a set of materials for which the bandgap and formation energies are provided

test.csv - contains the set of materials for which you must predict the bandgap and formation energies

/{train|test}/{id}/geometry.xyz - files with spatial information about the material. The file name corresponds to the id in the respective csv files.

### Disclaimer: The data provided belongs to Nomad Competition hosted in Kaggle. The data is provided for educational purposes only.