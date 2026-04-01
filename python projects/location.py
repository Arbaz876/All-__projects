import requests

def get_ip_location(ip_address):
    response = requests.get(f'http://ip-api.com/json/{ip_address}')
    return response.json()

# Returns city/region/country but not exact address (accuracy varies)
