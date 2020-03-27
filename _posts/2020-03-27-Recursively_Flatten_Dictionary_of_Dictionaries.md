---
layout: post
title: "Recursively Flatten Dictionary of Dictionaries"
date: 2020-03-27 10:46:08 -0500
---

# Recursively Flatten Dictionary of Dictionaries

Over the last few weeks, I've been working on my MLB Stats project. I've been parsing the MLB stats API in order to store the information in a SQL database using the SQLAlchemy ORM framework. I wanted to share a function that I wrote that's proved particularly useful throughout the process. 

## The Problem

I'm trying to store the information in a normalized database, but the API will uses nested dictionaries. For example, this dictionary returned from the API that represents my home team's ballpark, Yankee Stadium: 

```python
{'id': 3313,
 'name': 'Yankee Stadium',
 'link': '/api/v1/venues/3313',
 'location': {'city': 'Bronx',
  'state': 'New York',
  'stateAbbrev': 'NY',
  'defaultCoordinates': {'latitude': 40.82919482, 'longitude': -73.9264977}},
 'timeZone': {'id': 'America/New_York', 'offset': -5, 'tz': 'EST'},
 'fieldInfo': {'capacity': 47309,
  'turfType': 'Grass',
  'roofType': 'Open',
  'leftLine': 318,
  'leftCenter': 399,
  'center': 408,
  'rightCenter': 385,
  'rightLine': 314}}
```

I'd like all of this information summed up in a normalized 'venue' table. I think it'd be unreasonable to have seperate tables for every nested dictionary. That would mean having seperate tables for 'fieldinfo', 'location', and 'timeZone' that all share the same primary key. 

## My Solution

```python
def flatten_dicts(dictionary):
    """
    recursively flatten a dictionary of dictionaries
    """
    #base case 
    if dict not in [type(x) for x in dictionary.values()]:
        return dictionary
    else:
        for key, value in dictionary.items():
            if type(value)==dict:
                temp_dict = dictionary.pop(key)
                for k,v in temp_dict.items():
                    dictionary[f"{key}_{k}"]=v
                return flatten_dicts(dictionary)
```
In plain english: if none of the values in the dictionary are of type 'dict', return the dictionary. Otherwise, we pop the nested dict, thereby removing it from the original dictionary and returning it in the same line. Then, we add the values from the nested dict to the orignial dict with the original key in the prefix. Here is the result of this function on the 'venue' record for Yankee Stadium. 

```python
{'id': 3313,
 'name': 'Yankee Stadium',
 'link': '/api/v1/venues/3313',
 'location_city': 'Bronx',
 'location_state': 'New York',
 'location_stateAbbrev': 'NY',
 'timeZone_id': 'America/New_York',
 'timeZone_offset': -5,
 'timeZone_tz': 'EST',
 'fieldInfo_capacity': 47309,
 'fieldInfo_turfType': 'Grass',
 'fieldInfo_roofType': 'Open',
 'fieldInfo_leftLine': 318,
 'fieldInfo_leftCenter': 399,
 'fieldInfo_center': 408,
 'fieldInfo_rightCenter': 385,
 'fieldInfo_rightLine': 314,
 'location_defaultCoordinates_latitude': 40.82919482,
 'location_defaultCoordinates_longitude': -73.9264977}
```

Be warned: dict.pop() changes the dictionary in place. So if you want to retain the orignal dictionary for later use, you'll have to deepcopy it before calling this function. 
