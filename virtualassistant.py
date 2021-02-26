import speech_recognition as sr
import pyttsx3 as pytt
import pywhatkit
import datetime
import wikipedia
import pyjokes

listener = sr.Recognizer()
alexa = pytt.init()
alexa.setProperty("rate", 160)
voices = alexa.getProperty("voices")


def talk(text):
    alexa.say(text)
    alexa.runAndWait()


def takecommand():
    try:
        with sr.Microphone() as source:
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.upper()
            if "ALEXA" in command:
                command = command.replace("ALEXA", "")
    except:
        pass
    return command


def runalexa():
    command = takecommand()
    if "PLAY" in command:
        song = command.replace("PLAY", "")
        talk("PLAYING")
        pywhatkit.playonyt("PLAYING " + song)
    elif "TIME" in command:
        time = datetime.datetime.now().time().strftime("%I:%M %p")
        talk(f"CURRENT TIME IS {time}")
    elif "FIND" in command:
        find = command.replace("SEARCH", "")
        info = wikipedia.summary(find, sentences=2)
        talk(info)
    elif "JOKE" in command:
        talk(pyjokes.get_joke(category="all"))
    elif "BYE" in command:
        talk("GOOD BYE! SEE YOU AGAIN!")
        exit()
    else:
        talk("PLEASE SAY THAT AGAIN!")


talk("I'M ALEXA! I'M LISTENING TO YOU")

while True:
    runalexa()
