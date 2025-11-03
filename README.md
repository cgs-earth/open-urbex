[![flake8 ⚙️](https://github.com/cgs-earth/urbex/actions/workflows/flake8.yml/badge.svg)](https://github.com/cgs-earth/urbex/actions/workflows/flake8.yml)
[![black](https://github.com/cgs-earth/urbex/actions/workflows/black.yml/badge.svg)](https://github.com/cgs-earth/urbex/actions/workflows/black.yml)
[![tests](https://github.com/cgs-earth/urbex/actions/workflows/test.yml/badge.svg)](https://github.com/cgs-earth/urbex/actions/workflows/test.yml)

# urbex
(or - the repository formally known as "The Atlas of Urban Expansion")

Author: Jeff Allenby

Attending Metaphysician: Margo Atkinson

### Introduction
The Atlas of Urban Expansion originated in 2014 through a partnership with the Lincoln Institute of Land Policy, New York University, and UN-Habitat. For three time periods, 1990, circa 2000, and 2014, 200 urban areas around the world were classified by a variety of metrics including overall extent, average population density, fragmentation, compactness, shares of infill, extension, leapfrogging, and more (www.atlasofurbanexpansion.org). This groundbreaking effort was limited by costly and time consuming manual classification of Landsat satellite imagery, and it has not been significantly updated since.

### Current Approach
Understanding and mapping urban extent remains a critical challenge in urban studies and planning. Traditional methodologies for delineating urban areas, such as the Atlas of Urban Expansion and Functional Urban Areas (FUAs), rely heavily on manual interpretation of satellite imagery and administrative data. While these approaches have provided valuable insights into urban form and growth, they present significant limitations in terms of reproducibility, temporal currency, and predictive capability.

The manual interpretation of urban boundaries introduces subjectivity into the analysis process. Different analysts or temporal inconsistencies in imagery (time of year or differences in precipitation) may draw wildly different conclusions from the same landscape over time, leading to inconsistencies in urban extent delineation. Furthermore, the time-intensive nature of reviewing these semi-automated processes means that updates are infrequent and often lag behind actual urban development patterns. This temporal gap becomes particularly problematic in rapidly urbanizing regions where development patterns can change significantly between analysis periods.

Remote sensing approaches have attempted to address these limitations through automated classification of urban areas. However, these methods often struggle to capture the nuanced patterns of urban development, particularly in non-Western cities or areas with complex land use patterns. Additionally, remote sensing analysis requires specialized data and expertise, limiting its widespread application and replication.

This analysis presents a novel methodology that leverages readily available open datasets to create an objective, reproducible, and predictive understanding of urban development patterns. Our approach combines building footprints, transportation networks, and terrain data from OpenStreetMap and Overture Maps with innovative analytical techniques adapted from ecological science. By treating buildings as species and urban areas as habitats, we apply presence-only observation modeling techniques (MAXENT) to understand and predict urban development patterns.

This methodology offers several key innovations. First, it eliminates subjectivity by establishing quantitative thresholds for urban pattern classification based on the relationship between building density and infrastructure provision. Second, it identifies informal settlements through an objective analysis of the disparity between building density and road network coverage. Third, it adapts ecological habitat modeling techniques to predict future urban growth based on the environmental and infrastructure characteristics of existing development.

The resulting analysis provides a comprehensive framework for understanding urban form that can be regularly updated as new data becomes available. The automated nature of the process ensures consistency across different urban contexts while maintaining methodological rigor. Furthermore, the predictive capabilities of the model offer valuable insights for urban planning and policy development, allowing cities to better anticipate and prepare for future growth patterns.


## Setting Up ubex:
The python environment that this repository runs on has been tested on Windows and Ubuntu (Linux) operating systems. It has been lightly tested on Mac OS (Intel Chip). Instructions are almost identical with the exception of one extra step for Mac OS - noted below.

The urbex repository uses uv as its Python package and project manager ([uv docs](https://docs.astral.sh/uv/)). One of the benefits is that uv attempts to make a universal environment file to allow for the same environment to be used across operating systems. With a few exceptions - uv makes this really easy even for geospatial coding! Due to these few exceptions, the Linux and Windows environments differ slightly in the support that the required packages need to run. So far the only impact is that we have to keep the lockfile "unlocked" when unit testing via GitHub Actions. One additonal note - uv does not play well with conda but is highly compatible with pip.

### Install Steps
1. Clone urbex repo from main branch.
2. Install uv ([uv docs](https://docs.astral.sh/uv/))
3. Reccomended Installs
    a. VS Code
    b. GitHub Desktop
4. MaxEnt (https://biodiversityinformatics.amnh.org/open_source/maxent/)
5. Once you have everything downloaded, open the urbex repository in your IDE. Open a terminal in your IDE and type: ```uv sync```. At this point, your python environment should download its required packages and set itself up.

** Macs differ here! **
The only difference that has been found so far is that Mac users need to install the python package "gdal" with [Homebrew](https://brew.sh/). GDAL has wheels specifically for Windows and Linux that work and are specified in the pyproject.toml file - pre-installing GDAL into your python environment is not required on Linux and Windows operating systems. Once you have done this, try ```uv sync``` again - it should work!

6. You can now use "urbex" as your python environment in your IDE or python session and you are ready to run the code!


## Repository Notes
- urbex is using the Black formatter to enforce code standards. It is mostly compatible with flake8 (PEP8 linter) and other linters but please refer to their website for tricks to fully get there if you have linter installed as well as Black.
- Even if you do not have Black installed in your IDE, every push to the repository triggers Black to run in a GitHub Action and therefore all work will be automatically formatted.
- The repository goal for unit testing is 80% coverage. All new functions must have a test written when submitting a PR.
- This repository is private - please do not share it without express permission from the repository admin (matkinson@lincolninst.edu).
- While currently private, the intention is to create public releases and have this repository become public. To that note - DO NOT STORE ANY SECRET INFORMATION IN FILES IN THIS REPOSITORY. It will be searchable forever no matter how many times you try to delete it from the interwebs. Secret information includes API keys, passwords, SSNs, birthdays, etc.Leave your identifying information at home please!
- Using this repository does require an internet connection to download the data required. If you have run it previously and have all the data in the right places, you won't need the internet.
##### Data Inputs
- Building footprints – Overture Maps Foundation
- Roads and transportation features (such as train and bus stations) - Open Street Map
- Water features, including rivers and lakes – Open Street Map
- Elevation – SRTM 30m global data
- Slope – SRTM 30m global data
- Urban extent and population – Global Human Settlement Layer

## Contribution Guidelines
Welcome! Please see CONTRIBUTING.md!

### Finally - the urbex repository is the IP of the Lincoln Institute of Land Policy. Any contributions will be considered the sole IP of LILP.