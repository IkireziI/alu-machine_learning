#!/usr/bin/env python3
"""Return list of ships"""

import requests


def availableShips(passengerCount):
    """Return list of ships

    Args:
        passengerCount (int): Minimum number of passengers required.
    
    Returns:
        list: Names of ships that can carry at least `passengerCount` passengers.
    """

    url = 'https://swapi-api.alx-tools.com/api/starships'
    output = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break  # Stop if request fails

        data = response.json()
        for ship in data['results']:
            passengers = ship['passengers'].replace(',', '')  # Remove commas

            try:
                if int(passengers) >= passengerCount:
                    output.append(ship['name'])
            except ValueError:
                pass  # Skip ships with "unknown" passenger counts

        url = data.get('next')  # Get next page URL, if available

    return output


# Example usage:
# print(availableShips(100))
