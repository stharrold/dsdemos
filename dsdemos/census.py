#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Utilities for working with data from the United States Census Bureau.[^cb]

Notes:
    * Acronyms:
        ACS: American Community Survey[^acs]
        PUMS: Public Use Microdata Sample[^pums], specific to ACS

References:
    [^cb]: http://www.census.gov/
    [^acs]: https://www.census.gov/programs-surveys/acs/
    [^pums]: https://www.census.gov/programs-surveys/acs/technical-documentation/pums.html

"""


# Import standard packages.
import collections
import os
import pdb
# Import installed packages.
# Import local packages.
import dsdemos.utils as utils


def parse_pumsdatadict(path:str) -> collections.OrderedDict:
    r"""Parse ACS PUMS Data Dictionaries.
    
    Args:
        path (str): Path to downloaded data dictionary.
        
    Returns:
        ddict (collections.OrderedDict): Parsed data dictionary with original
            key order preserved.
    
    Raises:
        FileNotFoundError: Raised if `path` does not exist.

    Notes:
        * Only some data dictionaries have been tested.[^urls]
        * Values are all strings. No data types are inferred from the
            original file.
        * Example structure of returned `ddict`:
            ddict['title'] = '2013 ACS PUMS DATA DICTIONARY'
            ddict['date'] = 'August 7, 2015'
            ddict['record_types']['HOUSING RECORD']['RT']\
                ['length'] = '1'
                ['description'] = 'Record Type'
                ['var_codes']['H'] = 'Housing Record or Group Quarters Unit'
            ddict['record_types']['HOUSING RECORD'][...]
            ddict['record_types']['PERSON RECORD'][...]
            ddict['notes'] =
                ['Note for both Industry and Occupation lists...',
                 '*  In cases where the SOC occupation code ends...',
                 ...]
    
    References:
        [^urls]: http://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/
            PUMSDataDict2013.txt
            PUMS_Data_Dictionary_2009-2013.txt
    
    """
    # Check arguments.
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Path does not exist:\n{path}".format(path=path))
    # Parse data dictionary.
    # Note:
    # * Data dictionary keys and values are "codes for variables",
    #   using the ACS terminology,
    #   https://www.census.gov/programs-surveys/acs/technical-documentation/pums/documentation.html
    # * The data dictionary is not all encoded in UTF-8. Replace encoding
    #   errors when found.
    # * Catch instances of inconsistently formatted data.
    ddict = collections.OrderedDict()
    with open(path, encoding='utf-8', errors='replace') as fobj:
        # Data dictionary name is line 1.
        ddict['title'] = fobj.readline().strip()
        # Data dictionary date is line 2.
        ddict['date'] = fobj.readline().strip()    
        # Initialize flags to catch lines.
        (catch_var_name, catch_var_desc,
         catch_var_code, catch_var_note) = (None, )*4
        var_name = None
        var_name_last = 'PWGTP80' # Necessary for unformatted end-of-file notes.
        for line in fobj:
            # Replace tabs with 4 spaces
            line = line.replace('\t', ' '*4).rstrip()
            # Record type is section header 'HOUSING RECORD' or 'PERSON RECORD'.
            if (line.strip() == 'HOUSING RECORD'
                or line.strip() == 'PERSON RECORD'):
                record_type = line.strip()
                if 'record_types' not in ddict:
                    ddict['record_types'] = collections.OrderedDict()
                ddict['record_types'][record_type] = collections.OrderedDict()
            # A newline precedes a variable name.
            # A newline follows the last variable code.
            elif line == '':
                # Example inconsistent format case:
                # WGTP54     5
                #     Housing Weight replicate 54
                #
                #           -9999..09999 .Integer weight of housing unit
                if (catch_var_code
                    and 'var_codes' not in ddict['record_types'][record_type][var_name]):
                    pass
                # Terminate the previous variable block and look for the next
                # variable name, unless past last variable name.
                else:
                    catch_var_code = False
                    catch_var_note = False
                    if var_name != var_name_last:
                        catch_var_name = True
            # Variable name is 1 line with 0 space indent.
            # Variable name is followed by variable description.
            # Variable note is optional.
            # Variable note is preceded by newline.
            # Variable note is 1+ lines.
            # Variable note is followed by newline.
            elif (catch_var_name and not line.startswith(' ') 
                and var_name != var_name_last):
                # Example: "Note: Public use microdata areas (PUMAs) ..."
                if line.lower().startswith('note:'):
                    var_note = line.strip() # type(var_note) == str
                    if 'notes' not in ddict['record_types'][record_type][var_name]:
                        ddict['record_types'][record_type][var_name]['notes'] = list()
                    # Append a new note.
                    ddict['record_types'][record_type][var_name]['notes'].append(var_note)
                    catch_var_note = True
                # Example: """
                # Note: Public Use Microdata Areas (PUMAs) designate areas ...
                # population.  Use with ST for unique code. PUMA00 applies ...
                # ...
                # """
                elif catch_var_note:
                    var_note = line.strip() # type(var_note) == str
                    if 'notes' not in ddict['record_types'][record_type][var_name]:
                        ddict['record_types'][record_type][var_name]['notes'] = list()
                    # Concatenate to most recent note.
                    ddict['record_types'][record_type][var_name]['notes'][-1] += ' '+var_note
                # Example: "NWAB       1 (UNEDITED - See 'Employment Status Recode' (ESR))"
                else:
                    # type(var_note) == list
                    (var_name, var_len, *var_note) = line.strip().split(maxsplit=2)
                    ddict['record_types'][record_type][var_name] = collections.OrderedDict()
                    ddict['record_types'][record_type][var_name]['length'] = var_len
                    # Append a new note if exists.
                    if len(var_note) > 0:
                        if 'notes' not in ddict['record_types'][record_type][var_name]:
                            ddict['record_types'][record_type][var_name]['notes'] = list()
                        ddict['record_types'][record_type][var_name]['notes'].append(var_note[0])
                    catch_var_name = False
                    catch_var_desc = True
                    var_desc_indent = None
            # Variable description is 1+ lines with 1+ space indent.
            # Variable description is followed by variable code(s).
            # Variable code(s) is 1+ line with larger whitespace indent
            # than variable description. Example:"""
            # PUMA00     5      
            #     Public use microdata area code (PUMA) based on Census 2000 definition for data
            #     collected prior to 2012. Use in combination with PUMA10.          
            #           00100..08200 .Public use microdata area codes 
            #                   77777 .Combination of 01801, 01802, and 01905 in Louisiana
            # 	          -0009 .Code classification is Not Applicable because data 
            #                         .collected in 2012 or later            
            # """
            # The last variable code is followed by a newline.
            elif (catch_var_desc or catch_var_code) and line.startswith(' '):
                indent = len(line) - len(line.lstrip())
                # For line 1 of variable description.
                if catch_var_desc and var_desc_indent is None:
                    var_desc_indent = indent
                    var_desc = line.strip()
                    ddict['record_types'][record_type][var_name]['description'] = var_desc
                # For lines 2+ of variable description.
                elif catch_var_desc and indent <= var_desc_indent:
                    var_desc = line.strip()
                    ddict['record_types'][record_type][var_name]['description'] += ' '+var_desc
                # For lines 1+ of variable codes.
                else:
                    catch_var_desc = False
                    catch_var_code = True
                    is_valid_code = None
                    if not line.strip().startswith('.'):
                        # Example case: "01 .One person record (one person in household or"
                        if ' .' in line:
                            (var_code, var_code_desc) = line.strip().split(
                                sep=' .', maxsplit=1)
                            is_valid_code = True
                        # Example inconsistent format case:"""
                        #            bbbb. N/A (age less than 15 years; never married)
                        # """
                        elif '. ' in line:
                            (var_code, var_code_desc) = line.strip().split(
                                sep='. ', maxsplit=1)
                            is_valid_code = True
                        else:
                            raise AssertionError(
                                "Program error. Line unaccounted for:\n" +
                                "{line}".format(line=line))
                        if is_valid_code:
                            if 'var_codes' not in ddict['record_types'][record_type][var_name]:
                                ddict['record_types'][record_type][var_name]['var_codes'] = collections.OrderedDict()
                            ddict['record_types'][record_type][var_name]['var_codes'][var_code] = var_code_desc
                    # Example case: ".any person in group quarters)"
                    else:
                        var_code_desc = line.strip().lstrip('.')
                        ddict['record_types'][record_type][var_name]['var_codes'][var_code] += ' '+var_code_desc
            # Example inconsistent format case:"""
            # ADJHSG     7      
            # Adjustment factor for housing dollar amounts (6 implied decimal places)
            # """
            elif (catch_var_desc and
                'description' not in ddict['record_types'][record_type][var_name]):
                var_desc = line.strip()
                ddict['record_types'][record_type][var_name]['description'] = var_desc
                catch_var_desc = False
                catch_var_code = True
            # Example inconsistent format case:"""
            # WGTP10     5
            #     Housing Weight replicate 10
            #           -9999..09999 .Integer weight of housing unit
            # WGTP11     5
            #     Housing Weight replicate 11
            #           -9999..09999 .Integer weight of housing unit
            # """
            elif ((var_name == 'WGTP10' and 'WGTP11' in line)
                or (var_name == 'YOEP12' and 'ANC' in line)):
                # type(var_note) == list
                (var_name, var_len, *var_note) = line.strip().split(maxsplit=2)
                ddict['record_types'][record_type][var_name] = collections.OrderedDict()
                ddict['record_types'][record_type][var_name]['length'] = var_len
                if len(var_note) > 0:
                    if 'notes' not in ddict['record_types'][record_type][var_name]:
                        ddict['record_types'][record_type][var_name]['notes'] = list()
                    ddict['record_types'][record_type][var_name]['notes'].append(var_note[0])
                catch_var_name = False
                catch_var_desc = True
                var_desc_indent = None
            else:
                if (catch_var_name, catch_var_desc,
                    catch_var_code, catch_var_note) != (False, )*4:
                    raise AssertionError(
                        "Program error. All flags to catch lines should be set " +
                        "to `False` by end-of-file.")
                if var_name != var_name_last:
                    raise AssertionError(
                        "Program error. End-of-file notes should only be read "+
                        "after `var_name_last` has been processed.")
                if 'notes' not in ddict:
                    ddict['notes'] = list()
                ddict['notes'].append(line)
    return ddict
