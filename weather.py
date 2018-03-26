import pyowm
from config import owm_key


class Weather:
    def __init__(self):
        self.owm = pyowm.OWM(owm_key)

    def test(self):
        weather = self.owm.weather_at_place('Provo,USA').get_weather()
        print(weather.get_temperature('fahrenheit'))

Weather().test()
