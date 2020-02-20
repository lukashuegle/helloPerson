import win32com.client as wincl
import datetime
import time as timetime
import threading


class Sayhello:

    def __init__(self, volume, rate):
        self.engine = wincl.Dispatch("SAPI.SpVoice")
        self.last_again = timetime.time() - 100

    def sayagain_async(self, last):
        t1 = threading.Thread(target=self.sayagain, args=(last,))
        t1.start()
    
    def sayhello_async(self):
        t1 = threading.Thread(target=self.sayhello)
        t1.start()

    def sayagain(self, last):
        if (timetime.time() - self.last_again) >= 5:
            self.last_again = timetime.time()
            now = datetime.datetime.now()
            difference = now - last
            datetime.timedelta(0, 8, 562000)
            seconds_in_day = 24 * 60 * 60
            time = divmod(difference.days * seconds_in_day + difference.seconds, 60)
            if time[0] == 0:
                self.engine.Speak('Hallo')
                self.engine.Speak('Ich habe Sie das letze mal vor ' + str(time[1]) + ' Sekunden gesehen')
            else:
                self.engine.Speak('Hallo')
                self.engine.Speak('Ich habe Sie das letze mal vor ' + str(time[0]))
                self.engine.Speak(' Minuten und ' + str(time[1]) + 'Sekunden gesehen')

    def sayhello(self):
        self.engine.Speak('Hallo herzlich willkommen! Schön dass Sie hier sind.')

