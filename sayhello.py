import pyttsx3
import datetime


class Sayhello:

    def __init__(self, volume, rate):
        self.engine = pyttsx3.init()  # object creation
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[4].id)
        self.engine.setProperty('volume', volume)  # 1.0
        self.engine.setProperty('rate', rate)  # 150

    def sayagain(self, last):
        now = datetime.datetime.now()
        difference = now - last
        datetime.timedelta(0, 8, 562000)
        seconds_in_day = 24 * 60 * 60
        time = divmod(difference.days * seconds_in_day + difference.seconds, 60)
        if time[0] == 0:
            self.engine.say('Hallo Again')
            self.engine.say('Ich habe Sie das letze mal vor ' + str(time[1]) + ' Sekunden gesehen')
        else:
            self.engine.say('Hallo Again')
            self.engine.say('Ich habe Sie das letze mal vor ' + str(time[0]))
            self.engine.say(' Minuten und ' + str(time[1]) + 'Sekunden gesehen')
        self.engine.runAndWait()
        self.engine.stop()

    def sayhello(self):
        self.engine.say('Hallo herzlich willkommen! Schön dass Sie hier sind.')
        self.engine.runAndWait()
        self.engine.stop()
