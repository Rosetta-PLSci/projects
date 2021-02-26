import wikipedia
import googlesearch
import webbrowser
import time
import requests
import random
from bs4 import BeautifulSoup

while True:
    try:
        whatSearch = input("WHAT DO YOU WANT TO SEARCH: ").upper()
        wheretosave = input("WHERE DO YOU WANT TO SAVE(TYPE FULL EXTENSION): ")
        whatnamedoc = input("WHAT IS THE FILE NAME: ").upper()
        print("IT'S SEARCHING\n")
        summarywhat = wikipedia.summary(whatSearch, sentences=10).replace(".", "\n")
        photoswhat = wikipedia.page(whatSearch).images
        urllist = [url for url in
                   googlesearch.search(whatSearch, tld="com", lang="en", num=20, start=0, stop=20, pause=3.0)]

        with open(wheretosave + whatnamedoc + "(pic)" + ".txt", "w") as fileobject:
            for i in photoswhat:
                fileobject.write(i + "\n")

        for i in photoswhat:
            values = str(random.randint(1, 2000))
            url = requests.get(f"{i}")
            with open(wheretosave + f"{whatnamedoc}{values}.jpg", "wb") as f:
                f.write(url.content)
        try:
            with open(wheretosave + f"{whatnamedoc}(info).txt", "w", encoding="utf-8") as f:
                for u in urllist:
                    requestsurl = requests.get(u)
                    newcontent = requestsurl.content
                    soup = BeautifulSoup(newcontent, "html.parser")
                    s = soup.find_all("p")
                    for info in s:
                        f.write(info.text)
        except Exception:
            pass

        with open(wheretosave + whatnamedoc + "(sum)" + ".txt", "w", encoding="utf-8") as fileobject1:
            fileobject1.write(summarywhat)

        with open(wheretosave + whatnamedoc + "(link)" + ".txt", "w") as fileobject3:
            for link in urllist:
                fileobject3.write(link + "\n")
        print()
        print("PROCESS COMPLETED SUCCESSFULLY\n")

        while True:
            answerContinue = input("DO YOU WANT TO CHECK THE LINK/(Y)ES OR (N)O: ").upper()
            if answerContinue == "Y":
                print("1) PICTURES\n2) LINKS OF GOOGLE\n3) SUMMARY\n")
                answerOption = input("TYPE YOUR NUMBER WHERE YOU WANT TO GO: ")
                if answerOption == "1":
                    print("GOING!")
                    time.sleep(3)
                    for goPicURL in photoswhat:
                        time.sleep(0.7)
                        webbrowser.open(goPicURL)
                    continue
                elif answerOption == "2":
                    print("GOING!")
                    time.sleep(3)
                    for goGoURL in urllist:
                        time.sleep(0.7)
                        webbrowser.open(goGoURL)
                    continue
                elif answerOption == "3":
                    time.sleep(1)
                    print(summarywhat)
                    continue
                else:
                    print("THAT IS NOT OPTION, CHOOSE 1-2-3")
            elif answerContinue != "Y" and answerContinue != "N":
                print("THAT IS NOT OPTION, CHOOSE Y-N")
            elif answerContinue == "N":
                print("PROCESS HAS BEEN ENDED")
                break
        exit()


    except Exception:

        print("SOMETHING IS WRONG! PLEASE CHECK ALL PARAMETERS")
        continue
