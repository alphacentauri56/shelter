import os
#import math
import string
import numpy as np
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.io import ascii
from astropy import constants as const
from astropy.time import Time
from astropy.table import Table
import matplotlib.pyplot as plt
#import gaia_query as gaia
from pathlib import Path

import time
start_time = time.time()

########################
# --- LOADING DATA --- #
########################

def get_directory():
    try:
        # Try to use __file__ for regular scripts
        file_path = Path(__file__)
        return file_path.parent
    except NameError:
        # Fallback for Jupyter notebooks
        try:
            import ipynbname
            notebook_path = ipynbname.path()
            return notebook_path.parent
        except Exception as e:
            print("Could not determine the file path:", e)
            return None
        
def file_load(file):
    # Define working directory & read path
    raw_data_path = Path('.')  # Tuple of all subdirectories below CWD
    csv_location = str(list(raw_data_path.glob("**/" + file))[0])
    return pd.read_csv(csv_location)

def create_folder(folder):
    is_exist = os.path.exists(folder)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(folder)

def fill_values(planet_table, planet_data, required=None):
    # Gather required data, prioritizing values with the smallest uncertainty
    if required is not None:
        for param in required:
            try:
                # Define uncertainty column names
                err1_col = f"{param}err1"
                err2_col = f"{param}err2"

                if isinstance(planet_table[param], np.ma.MaskedArray):
                    # Handle masked columns
                    if np.ma.is_masked(planet_data[param]):
                        # Compute uncertainties for all rows
                        uncertainties = []
                        for row in planet_table:
                            if not np.ma.is_masked(row[param]):
                                err1 = abs(row.get(err1_col, float('inf')))
                                err2 = abs(row.get(err2_col, float('inf')))
                                total_uncertainty = err1 + err2
                                uncertainties.append((total_uncertainty, row[param], err1, err2))
                        
                        # Select the value with the smallest uncertainty
                        if uncertainties:
                            best_value = min(uncertainties, key=lambda x: x[0])
                            planet_data[param] = best_value[1]
                            planet_data[err1_col] = best_value[2]
                            planet_data[err2_col] = best_value[3]

                else:
                    # Handle unmasked columns
                    if np.isnan(planet_data[param]):  # For unmasked arrays
                        # Compute uncertainties for all rows
                        uncertainties = []
                        for row in planet_table:
                            if not np.isnan(row[param]):
                                err1 = abs(row.get(err1_col, float('inf')))
                                err2 = abs(row.get(err2_col, float('inf')))
                                total_uncertainty = err1 + err2
                                uncertainties.append((total_uncertainty, row[param], err1, err2))
                        
                        # Select the value with the smallest uncertainty
                        if uncertainties:
                            best_value = min(uncertainties, key=lambda x: x[0])
                            planet_data[param] = best_value[1]
                            planet_data[err1_col] = best_value[2]
                            planet_data[err2_col] = best_value[3]

            except KeyError:
                print(f"Cannot retrieve {param} for planet {planet_data['pl_letter']}")

    return planet_data

def merge_tables(id, input_data, filepath=None):
    if 'KIC' or 'KOI' or 'Kepler' in id:
        mission = 'Kepler'
    if 'TOI' in id:
        mission = 'TESS'

    data = input_data.to_pandas()

    if mission == 'Kepler':
        file = 'KOI list.csv'
        file_df = file_load(file)
        file_df = file_df[file_df['koi_disposition'] != 'FALSE POSITIVE']
        file_df = file_df[file_df.kepid == int(id.lstrip('KIC '))].reset_index()

        # Base dictionary to map columns
        column_mapping = {
            "pl_orbsmax": "koi_sma",
            "pl_ratdor": "koi_dor",
            "pl_orbper": "koi_period",
            "pl_tranmid": "koi_time0",
            "pl_ratror": "koi_ror",
            "pl_orbeccen": "koi_eccen",
            "pl_orbincl": "koi_incl",
            "pl_imppar": "koi_impact",
            "pl_trandep": "koi_depth",
            "pl_trandur": "koi_duration",
            "pl_orblper": "koi_longp",
            "st_mass": "koi_smass",
            "st_rad": "koi_srad",
            "st_dens": "koi_srho",
            "st_teff": "koi_steff",
        }

        # Extend the dictionary to include error mappings
        extended_mapping = {}
        for key, value in column_mapping.items():
            extended_mapping[f"{key}err1"] = f"{value}_err1"
            extended_mapping[f"{key}err2"] = f"{value}_err2"

        # Combine the original mapping with the extended error mappings
        column_mapping.update(extended_mapping)
                          
        column_mapping.update({"ra": "ra",
            "dec": "dec",
            "hostname": "kepid",
            "pl_name": "kepoi_name",
        })

    if mission == 'TESS':
        file = 'TOI list.csv'
        file_df = file_load(file)
        file_df = file_df[file_df['tfopwg_disp'] != 'FP']
        file_df = file_df[file_df.toipfx == int(id.lstrip('TOI '))].reset_index()

        column_mapping = {
            "pl_orbper": "pl_orbper",
            "pl_tranmid": "pl_tranmid",
            "pl_trandep": "pl_trandep",
            "pl_trandur": "pl_trandurh",
            "st_mass": "st_mass",
            "st_rad": "st_rad",
            "st_teff": "st_teff",
        }

        # Extend the dictionary to include error mappings
        extended_mapping = {}
        for key, value in column_mapping.items():
            extended_mapping[f"{key}err1"] = f"{value}err1"
            extended_mapping[f"{key}err2"] = f"{value}err2"

        # Combine the original mapping with the extended error mappings
        column_mapping.update(extended_mapping)
                          
        column_mapping.update({"ra": "ra",
            "dec": "dec",
            "hostname": "toipfx",
            "pl_name": "toi",
        })

    # Rename columns of file_df to match data
    file_df = file_df.rename(columns={v: k for k, v in column_mapping.items()})

    common_cols = [col for col in set(file_df.columns).intersection(data.columns)]
    file_df = file_df[common_cols]

    file_df['default_flag'] = np.ones(len(file_df))
    letters = list(string.ascii_lowercase)
    letters.pop(0)
    if len(data) > 0:
        data_letters = set(data['pl_letter'])
        data['disposition'] = 'confirmed'
        letters = [letter for letter in letters if letter not in data_letters]

    # Append rows from file_df to data based on the period criteria
    per = data.dropna(subset=['pl_orbper'])['pl_orbper'].values  # Orbital periods in 'data'
    pererr = data.dropna(subset=['pl_orbper'])['pl_orbpererr1'].values
    
    if len(per) > 0:  # Check only if 'data' has periods
        for index, row in file_df.iterrows():
            pl_orbper = row['pl_orbper']
            pl_orbpererr1 = row.get('pl_orbpererr1', 0)  # Error in orbital period

            # Debugging: Print information about the matching process
            # print(f"Row {index}: pl_orbper={pl_orbper}, closest_per={per}, pererr={pl_orbpererr1}")

            # Check for the closest match
            pindex = np.argmin(np.abs(per - pl_orbper))
            diff = np.abs(per[pindex] - pl_orbper)
            threshold = pl_orbper*0.01

            print(f"Closest match diff: {diff}, threshold: {threshold}")

            # Skip this row if the period matches closely
            if diff < threshold:
                print(f"Skipping row {index} due to close match.")
                continue
            
            # Only append if the row doesn't match
            if letters:  # Ensure there are letters available
                row['pl_letter'] = letters.pop(0)  # Assign a new letter
                row['disposition'] = 'candidate'
                data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)

    else:
        # Directly append all rows if 'data' has no periods
        print("No periods found in data; appending all rows from file_df.")
        file_df['pl_letter'] = letters[:len(file_df)]
        file_df['disposition'] = 'candidate'
        letters = letters[len(file_df):]  # Update letters list
        data = pd.concat([data, file_df], ignore_index=True)

    data = Table.from_pandas(data)

    if filepath is None:
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, 'data_dump/archive_queries')

    create_folder(filepath)
    file_loc = os.path.join(filepath, f'exoarchiv_{id}.csv')

    return data

def query_archive(id, index=None, default=True, filepath=None, required_params=None, table_id='ps', save=True, overwrite=False):
    if filepath is None:
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, 'data_dump/archive_queries')
    create_folder(filepath)
    file_loc = os.path.join(filepath, f'exoarchiv_{id}.csv')
    is_exist = os.path.exists(file_loc)

    def _whole_word_match(name, id):
        # Match if name == id, or name starts with id + ' ', or ends with ' ' + id, or contains ' ' + id + ' '
        return (
            name == id or
            name.startswith(id + ' ') or
            name.endswith(' ' + id) or
            (' ' + id + ' ') in name
        )

    table = None
    if (not is_exist) or overwrite:
        # Try exact query first
        table_data = NasaExoplanetArchive.query_object(id, table=table_id)
        if len(table_data) == 0:
            # If no results, try substring search
            all_data = NasaExoplanetArchive.query_criteria(table=table_id, select='*')
            # Filter for whole word matches
            matches = [row for row in all_data if _whole_word_match(str(row['hostname']), id)]
            if len(matches) == 0:
                print(f"No results found for '{id}' in the archive.")
                return None
            table_data = Table(rows=matches, names=all_data.colnames)
        if save:
            ascii.write(table_data, file_loc, format='csv', overwrite=True)
        table = table_data
    else:
        table = ascii.read(file_loc, format='csv')

    if default:
        try:
            index = np.where(table['default_flag'] == 1)[0][0]
        except IndexError:
            print('No default parameter sets available')
            return None

    if index == 'latest':
        data = table[0]
    elif index == 'oldest':
        data = table[-1]
    elif index == 'all':
        data = table
    elif index == None:
        data = table
    else:
        data = table[index]

    # Gather required data, checking additional datasets if necessary
    data = fill_values(table, data, required_params)    

    return data

def print_query(id):
    print(query_archive(id))

########################
# --- DICTIONARIES --- #
########################

planet_params = {
    "semiamplitude": "pl_rvamp",
    "semimajor_axis": "pl_orbsmax",
    "scaled_semimajor_axis": "pl_ratdor",
    "period": "pl_orbper",
    "mass": "pl_bmasse",
    "radius": "pl_rade",
    "scaled_radius": "pl_ratror",
    "density": "pl_dens",
    "eccentricity": "pl_orbeccen",
    "inclination": "pl_orbincl",
    "insolation_flux": "pl_insol",
    "time_of_midtransit": "pl_tranmid",
    "temperature": "pl_eqt",
    "impact_parameter": "pl_imppar",
    "depth": "pl_trandep",
    "duration": "pl_trandur",
    "arg_periastron": "pl_orblper",
    "time_periastron": "pl_orbtper",
    "star_temperature": "st_teff",
    "star_mass": "st_mass",
    "star_radius": "st_rad",
    "star_density": "st_dens",
    "ra": "ra",
    "dec": "dec",
}

planet_names = {
    "star_name": "hostname",
    "name": "pl_name",
    "letter": "pl_letter",
    "time_ref": "pl_tsystemref",
    "publication": "pl_refname",
    "disposition": "disposition",
}

star_params = {
    "temperature": "st_teff",
    "radius": "st_rad",
    "mass": "st_mass",
    "density": "st_dens",
    "metallicity": "st_met",
    "luminosity": "st_lum",
    "ra": "ra",
    "dec": "dec",
}

star_names = {
    "name": "hostname",    
}

planet_aliases = {
    "semiamplitude": ["K"],
    "period": ["P", "per"],
    "time_of_midtransit": ["t0", "t_0", "midtransit_time", "transit_time", "time_of_transit"],
    "semimajor_axis": ["a", "sma"],
    "scaled_semimajor_axis": ["a_R", "ssma"],
    "mass": ["m"],
    "radius": ["r"],
    "scaled_radius": ["r_R", "p"],
    "density": ["rho", "dens"],
    "eccentricity": ["e", "ecc"],
    "inclination": ["i", "inc"],
    "insolation_flux": ["S", "insol"],
    "temperature": ["T_eq"],
    "impact_parameter": ["b", "imppar", "imp"],
    "depth": ["delta", "D"],
    "duration": ["W", "dur", "T_14"],
    "arg_periastron": ["omega", "w"],
    "time_periastron": ["t_p"],
    "star_temperature": ["T_star"],
    "star_mass": ["M_star"],
    "star_radius": ["R_star"],
    "star_density": ["rho_star"],
    "ra": ["RA"],
    "dec": ["DEC"],
    "dispostion": ["disp"]
}

star_aliases = {
    "temperature": ["T", "T_star"],
    "radius": ["R", "R_star"],
    "mass": ["M", "M_star"],
    "density": ["rho", "rho_star"],
    "metallicity": ["met"],
    "luminosity": ["L", "L_star"],
    "ra": ["RA"],
    "dec": ["DEC"],
}

# Define uncertainty suffix aliases
uncertainty_suffixes = {
    "upper": ["u", "err1"],
    "lower": ["l", "err2"],
}

def retrieve_params(object, dict, data, generate_uncertainties=True):
    keys = dict.keys()
    for key in keys:
        try:
            header = dict[key]
            value = data[header]
            if np.ma.is_masked(value):
                value = np.array([value])[0]
            if np.isnan(value):
                value = None

            setattr(object, key, value)

            if generate_uncertainties:
                # Dynamically generate uncertainty attributes (_upper and _lower)
                for suffix, uncertainty_key in [("upper", f"{header}err1"), ("lower", f"{header}err2")]:
                    uncertainty_value = None

                    if uncertainty_key in data.colnames:
                        uncertainty_value = data[uncertainty_key]
                        if np.ma.is_masked(uncertainty_value):
                            uncertainty_value = np.array([uncertainty_value])[0]
                        if np.isnan(uncertainty_value):
                            uncertainty_value = None

                    setattr(object, f"{key}{suffix}", uncertainty_value)
        except KeyError:
            print(f'{dict[key]} not in data headers, skipping.')
            continue

def retrieve_names(object, dict, data):
    keys = dict.keys()
    for key in keys:
        try:
            header = dict[key]
            setattr(object, key, data[header])
        except KeyError:
            print(f'{dict[key]} not in data headers, skipping.')
            continue

def setup_system(system_name, index=0, default=True, filepath=None,
                 required=[
                     'pl_orbper', 'pl_tranmid', 'pl_trandur', 'pl_ratdor', 'pl_ratror',
                     'pl_orbpererr1', 'pl_tranmiderr1', 'pl_trandurerr1', 'pl_ratdorerr1', 'pl_ratrorerr1',
                     'pl_orbpererr2', 'pl_tranmiderr2', 'pl_trandurerr2', 'pl_ratdorerr2', 'pl_ratrorerr2'
                 ],
                 candidate_data=False, table_id='ps', include_planets=None, save=True):
    '''
    Initializes a system with stars and planets based on archival data.

    Parameters:
        system_name (str): Name of the system.
        index (int or list): Index or list of indices to select specific data entries for each planet.
        default (bool): Whether to use the default data entries as specified in the archive.
        include_planets (list, optional): A list of planet letters (e.g., ['b', 'c']) or indices (e.g., [0, 2]) to include.
                                          If None, all available planets are included.
        save (bool): Whether to save the query result to disk.
    Returns:
        System: Initialized system object containing star and planet data.
    '''
    setup = System(system_name)

    data = query_archive(system_name, index='all', default=False, filepath=filepath, table_id=table_id, save=save)
    if data is None:
        print(f"Error: No data could be retrieved for system '{system_name}'.")
        return None
    if candidate_data:
        data = merge_tables(system_name, data, filepath=filepath)
    else:
        data['disposition'] = 'confirmed'

    letters = list(set(data['pl_letter']))

    # Filter planets based on include_planets
    if include_planets is not None:
        if all(isinstance(p, int) for p in include_planets):  # Index-based selection
            letters = [letters[i] for i in include_planets if i < len(letters)]
        elif all(isinstance(p, str) for p in include_planets):  # Letter-based selection
            letters = [p for p in letters if p in include_planets]

    nstars = []
    nplanets = 0

    if table_id == 'ps':
        # Determine the index and period for each planet
        planet_info = []
        for letter in letters:
            planet_table = data[data['pl_letter'] == letter]
            # Determine which index to use for this planet
            if default:
                planet_idx_array = np.where(planet_table['default_flag'] == 1)[0]
                planet_idx = int(planet_idx_array[0]) if len(planet_idx_array) > 0 else 0
            else:
                try:
                    planet_idx = index[len(planet_info)]
                except Exception:
                    planet_idx = index[-1] if isinstance(index, list) else index
            planet_data = planet_table[planet_idx]
            period = planet_data['pl_orbper']
            planet_info.append((letter, planet_idx, period))

        # Sort planets by their orbital period
        planet_info.sort(key=lambda x: x[2] if not np.isnan(x[2]) else float('inf'))

        # Process planets in sorted order
        for letter, planet_idx, _ in planet_info:
            planet_table = data[data['pl_letter'] == letter]
            planet_data = planet_table[planet_idx]
            # Gather required data, prioritizing values with the smallest uncertainty
            planet_data = fill_values(planet_table, planet_data, required)
            nstar = planet_data['sy_snum'] if 'sy_snum' in planet_data.colnames else 1
            if np.ma.is_masked(nstar):
                nstar = np.array([nstar])[0]
            nstars.append(nstar)
            nplanets += 1
            planet_obj = setup.create_data_planet(data=planet_data)
            if planet_obj:
                if hasattr(planet_obj, 'patch_data'):
                    planet_obj.patch_data()

        setup.nstars = max(set(nstars), key=nstars.count) if nstars else 1
        setup.nplanets = nplanets

    if table_id == 'pscomppars':
        df = data.to_pandas().sort_values(by='pl_orbper')
        for _, planet_data in df.iterrows():
            planet_obj = setup.create_data_planet(data=planet_data)
            if planet_obj and hasattr(planet_obj, 'patch_data'):
                planet_obj.patch_data()

    # Create the star data after processing planets
    setup.create_data_star(id=system_name, index=int(index if isinstance(index, int) else index[-1]), default=default, filepath=filepath)

    return setup

#############################
# --- PARAMETER CLASSES --- #
#############################

class Distribution:
    def __init__(self, samples):
        self.samples = np.array(samples)

    def median(self):
        return np.percentile(self.samples, 50)

    def credible_interval(self, low=16, high=84):
        return (
            np.percentile(self.samples, low),
            np.percentile(self.samples, 50),
            np.percentile(self.samples, high),
        )

    def __repr__(self):
        low, med, high = self.credible_interval()
        return f"Distribution(median={med:.3g}, 16th={low:.3g}, 84th={high:.3g})"

class Parameter:
    ''' class for storing object parameter values and posterior distributions '''
    def __init__(self, value=None, upper=None, lower=None, aliases=None, distribution=None):
        self.value = value
        self.upper = upper
        self.lower = lower
        self.aliases = aliases or []
        self._distribution = None

        if distribution is not None:
            self.distribution = distribution

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, dist):
        if not isinstance(dist, Distribution):
            raise ValueError("distribution must be a Distribution instance")
        self._distribution = dist
        low, med, high = dist.credible_interval()
        self.value = med
        self.lower = med - low
        self.upper = high - med

    @property
    def samples(self):
        return self._distribution.samples if self._distribution else None

    def set_samples(self, samples):
        self.distribution = Distribution(samples)

    def __repr__(self):
        return f"Parameter(value={self.value}, +{self.upper}, -{self.lower})"

class ParameterContainer:
    '''
    A class to manage parameters, with support for aliases and uncertainty suffixes.
    Intended as a base class for object classes, e.g. Star, Planet.
    '''
    def __init__(self):
        self._parameters = {}
        self._aliases = {}

        self._suffix_map = {}
        for canonical, alts in uncertainty_suffixes.items():
            attr = canonical.lstrip('_')
            self._suffix_map[canonical] = attr
            for alt in alts:
                self._suffix_map[alt] = attr

    def set_param(self, name, value, upper=None, lower=None, aliases=None):
        param = Parameter(value, upper, lower, aliases=[name] + (aliases or []))
        self._parameters[name] = param
        for alias in param.aliases:
            self._aliases[alias] = name
            for suffix in self._suffix_map:
                self._aliases[f"{alias}{suffix}"] = f"{name}{suffix}"

    def get_param(self, name):
        real_name = self._aliases.get(name, name)
        base, suffix_attr = self._split_suffix(real_name)
        param = self._parameters.get(base)
        if not param:
            return None
        return getattr(param, suffix_attr) if suffix_attr else param

    def _split_suffix(self, name):
        for suffix, attr in self._suffix_map.items():
            if name.endswith(suffix):
                return name[:-len(suffix)], attr
        return name, None

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return self.get_param(name)

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('set_param', 'get_param'):
            super().__setattr__(name, value)
            return

        if '_aliases' in self.__dict__:
            real_name = self._aliases.get(name)
            if real_name:
                base, suffix_attr = self._split_suffix(real_name)
                param = self._parameters.get(base)
                if param:
                    if suffix_attr:
                        setattr(param, suffix_attr, value)
                    else:
                        param.value = value
                    return

        super().__setattr__(name, value)

    def set_posterior_samples(self, param_name, samples):
        param = self._parameters.setdefault(param_name, Parameter())
        param.set_samples(samples)

    def get_parameter_summary(self, param_name):
        param = self._parameters.get(param_name)
        if param:
            return {
                "value": param.value,
                "lower": param.lower,
                "upper": param.upper
            }
        return None

########################
# --- SYSTEM CLASS --- #
########################

class System(ParameterContainer):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.stars = []
        self.planets = []

    def add_star(self, star):
        self.stars.append(star)
        star.system = self

    def add_planet(self, planet):
        self.planets.append(planet)
        planet.system = self

    def create_custom_star(self, data):
        star = Star(data.get("name", "Unnamed Star"))
        for key, value in data.items():
            setattr(star, key, value)

        if star.name in [s.name for s in self.stars]:
            return None

        self.add_star(star)
        return star

    def create_custom_planet(self, data):
        planet = Planet(data.get("name", "Unnamed Planet"))
        for key, value in data.items():
            setattr(planet, key, value)

        if planet.name in [p.name for p in self.planets]:
            return None

        self.add_planet(planet)
        return planet

    def create_data_star(self, id=None, data=None, index=None, default=True, filepath=None):
        star = Star()
        if data is None:
            data = query_archive(id, index, default, filepath=filepath)
        retrieve_params(star, star_params, data)
        retrieve_names(star, star_names, data)

        if star.name in [s.name for s in self.stars]:
            return None

        self.add_star(star)
        return star

    def create_data_planet(self, id=None, data=None, index=None, default=True, filepath=None):
        planet = Planet()
        if data is None:
            data = query_archive(id, index, default, filepath=filepath)
        retrieve_params(planet, planet_params, data)
        retrieve_names(planet, planet_names, data)

        if planet.name in [p.name for p in self.planets]:
            return None

        self.add_planet(planet)
        return planet

class Star(ParameterContainer):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.system = None
        self.planets = []

        for std, aliases in star_aliases.items():
            for alias in aliases:
                self._aliases[alias] = std
                for suffix in self._suffix_map:
                    self._aliases[f"{alias}{suffix}"] = f"{std}{suffix}"

    def add_planet(self, planet):
        self.planets.append(planet)
        planet.host_stars.append(self)

class Planet(ParameterContainer):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.system = None
        self.host_stars = []  # Can be one or more stars

        for std, aliases in planet_aliases.items():
            for alias in aliases:
                self._aliases[alias] = std
                for suffix in self._suffix_map:
                    self._aliases[f"{alias}{suffix}"] = f"{std}{suffix}"
