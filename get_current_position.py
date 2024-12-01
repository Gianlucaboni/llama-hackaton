import googlemaps
import os
import json


def get_current_position() -> dict:
    """
    Retrieves the current geographic position using the Google Maps API.

    This function uses a geolocation request to determine the latitude and longitude
    of the device and then performs a reverse geocoding operation to obtain the
    formatted address.

    Returns:
        dict: A dictionary containing the following keys:
            - "lat" (float): The latitude of the current position.
            - "lng" (float): The longitude of the current position.
            - "address" (str): The formatted address corresponding to the location.
    """
    cwd = os.path.dirname(__file__)
    cred_path = os.path.join(cwd, "credentials", "credentials_google.json")
    with open(cred_path, "r") as f:
        creds = json.load(f)
        KEY_API_GOOGLE_MAPS = creds["KEY_API_GOOGLE_MAPS"]

    gmaps = googlemaps.Client(key=KEY_API_GOOGLE_MAPS)
    response = gmaps.geolocate()
    latitude = response["location"]["lat"]
    longitude = response["location"]["lng"]

    result = gmaps.reverse_geocode((latitude, longitude))
    position = {
        "lat": latitude,
        "lng": longitude,
        "address": result[0]["formatted_address"],
    }
    return position
